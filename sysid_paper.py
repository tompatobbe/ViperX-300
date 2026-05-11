#!/usr/bin/env python3
"""
sysid_paper.py — System Identification for ViperX-300 6-DOF Arm
================================================================
Implements the methodology from:
  "Physically feasible dynamic model identification and constrained control
   of robotic arms: A case study on the ViperX-300 6-DoF robotic manipulator"

Pipeline:
  1. Modified DH kinematics (paper Table 1 — Khalil/modified convention)
  2. Regressor matrix W(q, dq, ddq): shape (6, 78)  [13 params × 6 links]
  3. Base parameter reduction via QR with column pivoting
  4. Data loading + zero-phase Butterworth LPF (filtfilt, fc=10 Hz)
  5. Per-joint torque normalisation
  6. Constrained SLSQP optimisation for physical feasibility
  7. REL validation metric + plots

Parameter vector φ per link (13 entries):
  [m, m·cx, m·cy, m·cz, Jxx, Jxy, Jxz, Jyy, Jyz, Jzz, Fv, Fc, F0]
  where cx,cy,cz = CoM coordinates in local frame, J = inertia at origin

Physical feasibility constraints per link:
  - m > 0
  - Pseudo-inertia matrix Σ = [[Jxx+Jyy+Jzz/2, Jxy, Jxz, m·cx],
                               [Jxy, Jxx+Jyy+Jzz/2, Jyz, m·cy],
                               [Jxz, Jyz, Jxx+Jyy+Jzz/2, m·cz],
                               [m·cx, m·cy, m·cz, m]] PSD
  - Fv >= 0, Fc >= 0

pip install numpy scipy matplotlib pandas
"""

import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import warnings
from typing import Tuple, List, Optional

# =============================================================================
# 1. Standard (Craig) DH kinematics
#    Convention: T_{i-1,i} = Rz(θ_i) · Tz(d_i) · Tx(a_i) · Rx(α_i)
# =============================================================================

# DH parameters: [alpha_prev, a_prev, d_i, theta_offset_i]
#   Joint 2 offset = -0.437π rad  (~-78.7°, shoulder mount angle)
#   Joint 3 offset = -0.063π rad  (~-11.3°, elbow mount angle)
L1 = 0.12675   # base height [m]
L2 = 0.30594   # upper arm  [m]
L3 = 0.21981   # forearm    [m]  (L3+L4 = 0.30002 per paper)
L4 = 0.08021
L5 = 0.07000   # wrist pitch offset
L6 = 0.13658   # end-effector

DH_PARAMS = np.array([
    # alpha_prev   a_prev    d_i          theta_offset
    [0.0,          0.0,      L1,           0.0          ],   # joint 1
    [3*np.pi/2,    0.0,      0.0,         -0.437*np.pi  ],   # joint 2
    [0.0,          L2,       0.0,         -0.063*np.pi  ],   # joint 3
    [3*np.pi/2,    0.0,      L3+L4,        0.0          ],   # joint 4
    [  np.pi/2,    0.0,      0.0,          0.0          ],   # joint 5
    [3*np.pi/2,    0.0,      L6,           0.0          ],   # joint 6
], dtype=float)

N_JOINTS   = 6
N_PARAMS   = 13   # per link
N_PARAMS_T = N_JOINTS * N_PARAMS   # 78 total


def _dh_transform(alpha: float, a: float, d: float, theta: float) -> np.ndarray:
    """Standard (Craig) DH transform T_{i-1,i} = Rz(θ)·Tz(d)·Tx(a)·Rx(α)."""
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ])


def forward_kinematics(q: np.ndarray) -> List[np.ndarray]:
    """
    Compute transforms T_0^i for i=0..N_JOINTS.
    Returns list of 7 (4×4) matrices: T[0] = I (world), T[1]..T[6] = joint frames.
    """
    T = [np.eye(4)]
    for i in range(N_JOINTS):
        alpha, a, d, theta_off = DH_PARAMS[i]
        Ti = _dh_transform(alpha, a, d, q[i] + theta_off)
        T.append(T[-1] @ Ti)
    return T


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])


# =============================================================================
# 2. Regressor matrix  W(q, dq, ddq)  shape (6, 78)
#
#   Exploit linearity: τ = W·φ.
#   Column k of W = inverse_dynamics evaluated with φ = e_k (unit vector).
#
#   Per-link inverse dynamics using Newton-Euler in link-frame parameters.
#   Implementation: numeric Newton-Euler forward/backward pass
#   with inertia at the link *origin* (not CoM).
# =============================================================================

def _inertia_at_origin(phi_link: np.ndarray) -> np.ndarray:
    """
    Build 3×3 inertia tensor at link origin from φ_link[0..9].
    J_O = J_CoM + m·[c×]^T[c×]  is NOT used here because φ already
    stores J at origin (Jxx = Ixx_O, etc.).
    """
    Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = phi_link[4:10]
    return np.array([
        [Jxx, Jxy, Jxz],
        [Jxy, Jyy, Jyz],
        [Jxz, Jyz, Jzz],
    ])


def inverse_dynamics_phi(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray,
                          phi: np.ndarray) -> np.ndarray:
    """
    Recursive Newton-Euler inverse dynamics.
    phi : (78,) parameter vector [m, m·cx, m·cy, m·cz, Jxx..Jzz, Fv, Fc, F0] × 6 links
    Returns joint torques (6,).
    """
    T = forward_kinematics(q)
    R = [T[i][:3, :3] for i in range(N_JOINTS + 1)]   # R[0]=I, R[1]..R[6]

    g0 = np.array([0, 0, 9.81])   # gravity in world frame (z up, g positive)

    # --- Forward pass: propagate angular vel / accel, linear accel ----------
    omega   = [np.zeros(3)] * (N_JOINTS + 1)   # omega[i] in frame i
    domega  = [np.zeros(3)] * (N_JOINTS + 1)
    ddp     = [np.zeros(3)] * (N_JOINTS + 1)   # linear accel of frame origin
    ddp[0]  = g0                                # include gravity (d'Alembert)

    for i in range(1, N_JOINTS + 1):
        Ri  = R[i]                    # R_0^i
        Rp  = R[i-1]                  # R_0^{i-1}
        Rrel = Rp.T @ Ri              # R_{i-1}^i

        zi_1 = Rp[:, 2]              # z-axis of frame i-1 in world
        zi_1_loc = Rrel.T @ np.array([0, 0, 1])  # z_{i-1} in frame i

        omega_prev  = omega[i-1]
        domega_prev = domega[i-1]

        # Transform previous omega/domega to current frame
        omega_loc  = Rrel.T @ omega_prev
        domega_loc = Rrel.T @ domega_prev

        omega[i]  = omega_loc  + dq[i-1]  * zi_1_loc
        domega[i] = domega_loc + ddq[i-1] * zi_1_loc + np.cross(omega_loc, dq[i-1]*zi_1_loc)

        # Origin position of frame i in frame i coords
        p_i_w = T[i][:3, 3]
        p_im1_w = T[i-1][:3, 3]
        r_loc = Ri.T @ (p_i_w - p_im1_w)    # r_{i-1,i} in frame i

        ddp_prev_loc = Ri.T @ (Rp @ ddp[i-1])
        ddp[i] = ddp_prev_loc + np.cross(domega[i], r_loc) + np.cross(omega[i], np.cross(omega[i], r_loc))

    # --- Backward pass: compute forces / torques -----------------------------
    f_ext = [np.zeros(3)] * (N_JOINTS + 1)
    n_ext = [np.zeros(3)] * (N_JOINTS + 1)
    tau   = np.zeros(N_JOINTS)

    for i in range(N_JOINTS, 0, -1):
        idx    = i - 1                         # 0-indexed link
        phi_i  = phi[idx * N_PARAMS : (idx + 1) * N_PARAMS]
        mi     = phi_i[0]
        mci    = phi_i[1:4]                    # m·[cx, cy, cz] in frame i
        Ji     = _inertia_at_origin(phi_i)     # 3×3 at origin
        Fv_i   = phi_i[10]
        Fc_i   = phi_i[11]
        F0_i   = phi_i[12]

        Ri = R[i]

        # Linear acceleration of CoM in frame i
        ddp_com = ddp[i] + np.cross(domega[i], mci / (mi + 1e-12)) + \
                  np.cross(omega[i], np.cross(omega[i], mci / (mi + 1e-12)))

        # Force on link i
        fi = mi * ddp_com

        # Moment contribution at origin
        ni = Ji @ domega[i] + np.cross(omega[i], Ji @ omega[i]) + np.cross(mci, ddp[i])

        # Next link's wrench in frame i
        if i < N_JOINTS:
            Ri_next = R[i+1]
            Rrel_next = Ri.T @ Ri_next          # R_i^{i+1}
            p_next_w  = T[i+1][:3, 3]
            p_i_w     = T[i][:3, 3]
            r_next    = Ri.T @ (p_next_w - p_i_w)  # r_{i,i+1} in frame i

            f_next = Rrel_next @ f_ext[i]
            n_next = Rrel_next @ n_ext[i] + np.cross(r_next, f_next)
        else:
            f_next = np.zeros(3)
            n_next = np.zeros(3)

        f_ext[i] = fi - f_next
        n_ext[i] = ni + n_next

        # Joint torque = z-component of moment in frame i
        tau[idx] = n_ext[i][2] + Fv_i * dq[idx] + Fc_i * np.sign(dq[idx]) + F0_i

    return tau


def regressor(q: np.ndarray, dq: np.ndarray, ddq: np.ndarray) -> np.ndarray:
    """
    Build the (6, 78) regressor matrix W such that W @ phi = tau.
    Each column k = inverse_dynamics with phi = e_k.
    """
    W = np.zeros((N_JOINTS, N_PARAMS_T))
    e = np.zeros(N_PARAMS_T)
    for k in range(N_PARAMS_T):
        e[:] = 0.0
        e[k] = 1.0
        W[:, k] = inverse_dynamics_phi(q, dq, ddq, e)
    return W


# =============================================================================
# 3. Base parameter reduction via QR with column pivoting
# =============================================================================

def find_base_parameters(W_stacked: np.ndarray,
                          tol: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Given stacked regressor (M*6, 78), find base parameter columns via
    QR decomposition with column pivoting.

    Returns:
        base_cols : (p,) column indices of base parameters in original φ
        L_matrix  : (78, p) matrix such that W_stacked ≈ W_base @ L.T
                    i.e. φ_full = L @ φ_base (regrouping matrix)
    """
    Q, R, P = la.qr(W_stacked, pivoting=True, mode='economic')   # W[:, P] = Q @ R
    rank = np.sum(np.abs(np.diag(R)) > tol * np.abs(R[0, 0]))
    base_cols = np.sort(P[:rank])

    # Compute L such that W_full = W_base @ L^T
    # i.e. W_full[:, j] = Σ_k L[j,k] * W_base[:, k]
    W_base = W_stacked[:, base_cols]
    L = np.linalg.lstsq(W_base, W_stacked, rcond=None)[0].T   # (78, p)

    return base_cols, L


# =============================================================================
# 4. Data loading + zero-phase Butterworth LPF (paper: fc = 10 Hz)
# =============================================================================

STALL_TORQUE = np.array([4.1, 10.6, 10.6, 4.1, 4.1, 4.1])   # Nm (same as dynamic_model.py)
EFFORT_SCALE = STALL_TORQUE / 100.0   # raw load unit = % of stall torque

ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def load_and_filter(csv_path: str,
                    fs: float = 50.0,
                    fc: float = 10.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load CSV, apply zero-phase Butterworth LPF (paper: fc=10 Hz), then
    compute velocity and acceleration via central finite differences.

    Returns:
        t    : (N,)   time array
        q    : (N, 6) joint positions   [rad]
        dq   : (N, 6) joint velocities  [rad/s]
        tau  : (N, 6) joint torques     [Nm]
    """
    df = pd.read_csv(csv_path)
    t  = df["time"].values.astype(float)
    t  = t - t[0]

    pos_cols    = [f"{j}_pos" for j in ARM_JOINTS]
    effort_cols = [f"{j}_effort" for j in ARM_JOINTS]

    q_raw   = df[pos_cols].values.astype(float)
    eff_raw = df[effort_cols].values.astype(float)

    # Convert effort to Nm
    tau_raw = eff_raw * EFFORT_SCALE[np.newaxis, :]

    # Auto-detect actual sample rate from timestamps; warn if it differs from --fs
    fs_actual = 1.0 / float(np.median(np.diff(t)))
    if abs(fs_actual - fs) > 2.0:
        print(f"    [warn] --fs={fs:.1f} Hz but CSV contains {fs_actual:.1f} Hz data "
              f"— using detected rate for filter design")
    fs = fs_actual

    # Design zero-phase Butterworth filter (4th order → effectively 8th after filtfilt)
    nyq    = fs / 2.0
    Wn     = min(fc / nyq, 0.99)
    b, a   = sig.butter(4, Wn, btype='low')

    q_filt   = np.vstack([sig.filtfilt(b, a, q_raw[:, j])   for j in range(N_JOINTS)]).T
    tau_filt = np.vstack([sig.filtfilt(b, a, tau_raw[:, j]) for j in range(N_JOINTS)]).T

    # Velocity via central finite differences on filtered positions
    dt  = np.gradient(t)
    dq  = np.gradient(q_filt, axis=0) / dt[:, np.newaxis]
    dq  = np.vstack([sig.filtfilt(b, a, dq[:, j]) for j in range(N_JOINTS)]).T

    # Acceleration
    ddq = np.gradient(dq, axis=0) / dt[:, np.newaxis]
    ddq = np.vstack([sig.filtfilt(b, a, ddq[:, j]) for j in range(N_JOINTS)]).T

    return t, q_filt, dq, ddq, tau_filt


# =============================================================================
# 5. Per-joint torque normalization
# =============================================================================

def normalize_rows(W: np.ndarray, tau: np.ndarray,
                   tau_max: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale each joint row by 1/max|τ_j| so dominant joints don't skew the fit.
    Returns normalized W, tau, and the scale vector used.
    """
    if tau_max is None:
        tau_max = np.maximum(np.max(np.abs(tau), axis=0), 1e-6)
    scale = 1.0 / tau_max
    W_n   = W   * scale[np.newaxis, :]   # (6N, 78) — each row block scaled
    tau_n = tau * scale[np.newaxis, :]
    return W_n, tau_n, tau_max


# =============================================================================
# 6. Physical feasibility constraints
# =============================================================================

def pseudo_inertia(phi_link: np.ndarray) -> np.ndarray:
    """
    4×4 pseudo-inertia matrix Σ for a single link.
    Σ is PSD iff the link parameters are physically feasible.

    Σ = [[ (Jyy+Jzz-Jxx)/2,  -Jxy,            -Jxz,           m·cx ],
         [ -Jxy,              (Jxx+Jzz-Jyy)/2,  -Jyz,          m·cy ],
         [ -Jxz,              -Jyz,             (Jxx+Jyy-Jzz)/2, m·cz],
         [ m·cx,               m·cy,             m·cz,           m   ]]
    """
    m, mcx, mcy, mcz, Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = phi_link[:10]
    return np.array([
        [(Jyy+Jzz-Jxx)/2,  -Jxy,             -Jxz,            mcx],
        [-Jxy,              (Jxx+Jzz-Jyy)/2,  -Jyz,            mcy],
        [-Jxz,              -Jyz,              (Jxx+Jyy-Jzz)/2, mcz],
        [mcx,                mcy,               mcz,             m  ],
    ])


def feasibility_constraints(phi: np.ndarray) -> List[dict]:
    """Build scipy constraint list for physical feasibility of all 6 links."""
    constraints = []

    for i in range(N_JOINTS):
        idx = i * N_PARAMS

        # Mass positivity
        j = idx  # local to closure
        constraints.append({
            'type': 'ineq',
            'fun': lambda phi, j=j: phi[j] - 1e-4,   # m > 1e-4
        })

        # PSD pseudo-inertia: min eigenvalue >= small margin (avoids boundary issues in SLSQP)
        _PSD_MARGIN = 1e-6

        def min_eig(phi, idx=idx, margin=_PSD_MARGIN):
            phi_link = phi[idx: idx + N_PARAMS]
            Sigma = pseudo_inertia(phi_link)
            return np.min(np.linalg.eigvalsh(Sigma)) - margin

        constraints.append({'type': 'ineq', 'fun': min_eig})

        # Non-negative viscous friction
        constraints.append({
            'type': 'ineq',
            'fun': lambda phi, j=idx+10: phi[j],   # Fv >= 0
        })

        # Non-negative Coulomb friction
        constraints.append({
            'type': 'ineq',
            'fun': lambda phi, j=idx+11: phi[j],   # Fc >= 0
        })

    return constraints


# =============================================================================
# 7. Constrained optimisation
# =============================================================================

def initial_phi_guess() -> np.ndarray:
    """
    Initial guess using paper's identified friction values (Table 3) and
    small nominal inertial values.
    """
    Fv  = [0.126, 0.101, 0.0403, 0.0234, 0.0166, 0.00158]
    Fc  = [0.239, 0.412, 0.261,  0.0303, 0.0608, 0.00295]
    F0  = [-0.0529, -0.203, -0.766, 0.0119, -0.0127, -0.00469]

    # Rough link masses from ViperX-300 specs (kg)
    masses = [0.8, 0.8, 0.5, 0.3, 0.2, 0.1]

    phi0 = np.zeros(N_PARAMS_T)
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        m   = masses[i]
        phi0[idx + 0]  = m          # mass
        # m·cx = m·cy = m·cz = 0 initially
        phi0[idx + 4]  = m * 0.01   # Jxx (small but positive)
        phi0[idx + 7]  = m * 0.01   # Jyy
        phi0[idx + 9]  = m * 0.01   # Jzz
        phi0[idx + 10] = Fv[i]
        phi0[idx + 11] = Fc[i]
        phi0[idx + 12] = F0[i]

    return phi0


def identify(W_stacked: np.ndarray,
             tau_stacked: np.ndarray,
             phi0: Optional[np.ndarray] = None,
             w1: float = 1.0,
             w2: float = 1e-3,
             verbose: bool = True) -> np.ndarray:
    """
    Constrained identification:
       min w1 * ||W·φ - τ||² + w2 * ||φ - φ0||²
    subject to physical feasibility constraints.

    W_stacked  : (M, 78) stacked regressor over all samples
    tau_stacked: (M,)    stacked torque measurements (all joints concatenated)
    Returns identified φ (78,).
    """
    if phi0 is None:
        phi0 = initial_phi_guess()

    def objective(phi):
        residual = W_stacked @ phi - tau_stacked
        reg      = phi - phi0
        return w1 * residual @ residual + w2 * reg @ reg

    def gradient(phi):
        residual = W_stacked @ phi - tau_stacked
        return 2 * w1 * W_stacked.T @ residual + 2 * w2 * (phi - phi0)

    constraints = feasibility_constraints(phi0)

    if verbose:
        print(f"Starting constrained optimisation — {N_PARAMS_T} parameters, "
              f"{W_stacked.shape[0]} equations...")

    result = opt.minimize(
        objective, phi0, jac=gradient,
        method='SLSQP',
        constraints=constraints,
        options={'maxiter': 2000, 'ftol': 1e-9, 'disp': verbose},
    )

    if verbose:
        print(f"Optimisation status: {result.message}")
        print(f"Final cost: {result.fun:.6f}")

    return result.x


# =============================================================================
# 8. REL error metric
# =============================================================================
 
def rel_metric(tau_true: np.ndarray, tau_pred: np.ndarray) -> np.ndarray:
    """
    Relative Error Lenient per joint.
    REL_j = (1/N) Σ_t |τ_true[t,j] - τ_pred[t,j]| / max(|τ_true[t,j]|, |τ_pred[t,j]|)
    """
    denom = np.maximum(np.abs(tau_true), np.abs(tau_pred))
    denom = np.where(denom < 1e-6, 1e-6, denom)
    return np.mean(np.abs(tau_true - tau_pred) / denom, axis=0)


# =============================================================================
# 9. Excitation trajectory (sum of sinusoids, optimise condition number)
# =============================================================================

def excitation_trajectory(t: np.ndarray,
                           n_harmonics: int = 5,
                           seed: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a sum-of-sinusoids excitation trajectory per joint.
    q_j(t) = Σ_k [ a_jk sin(k·ωf·t) + b_jk cos(k·ωf·t) ]
    Coefficients are randomly initialised then optionally optimised
    to minimize the condition number of the stacked regressor.

    For the optimisation step the user should run this with actual hardware
    and then refine. Here we return a random but kinematically bounded trajectory.
    """
    rng  = np.random.default_rng(seed)
    wf   = 0.5 * np.pi   # fundamental frequency [rad/s] (~0.25 Hz)

    # Joint limits for ViperX-300 (rad)
    q_lim = np.array([
        [-3.14, 3.14],   # waist
        [-1.88, 1.99],   # shoulder
        [-2.15, 1.57],   # elbow
        [-3.14, 3.14],   # forearm_roll
        [-1.75, 2.15],   # wrist_angle
        [-3.14, 3.14],   # wrist_rotate
    ])

    q   = np.zeros((len(t), N_JOINTS))
    dq  = np.zeros_like(q)
    ddq = np.zeros_like(q)

    for j in range(N_JOINTS):
        span   = (q_lim[j, 1] - q_lim[j, 0]) * 0.35   # use 35% of range
        center = (q_lim[j, 0] + q_lim[j, 1]) / 2

        a = rng.uniform(-1, 1, n_harmonics)
        b = rng.uniform(-1, 1, n_harmonics)
        # Normalise so amplitude ≈ span
        scale = span / (np.sum(np.abs(a)) + np.sum(np.abs(b)) + 1e-9)
        a *= scale
        b *= scale

        for k_idx, k in enumerate(range(1, n_harmonics + 1)):
            kw     = k * wf
            q[:,  j] += a[k_idx] * np.sin(kw * t) + b[k_idx] * np.cos(kw * t)
            dq[:, j] += a[k_idx] * kw * np.cos(kw * t) - b[k_idx] * kw * np.sin(kw * t)
            ddq[:,j] -= a[k_idx] * kw**2 * np.sin(kw * t) + b[k_idx] * kw**2 * np.cos(kw * t)

        q[:, j] += center

    return q, dq, ddq


# =============================================================================
# 10. Full pipeline
# =============================================================================

def run_identification(csv_path: str,
                       fs: float = 50.0,
                       fc_lpf: float = 10.0,
                       verbose: bool = True,
                       plot: bool = True) -> np.ndarray:
    """
    End-to-end system identification from a CSV data file.

    Returns identified parameter vector phi (78,).
    """
    print("=" * 60)
    print("ViperX-300 System Identification — Paper Method")
    print("=" * 60)

    # --- Load & filter -------------------------------------------------------
    print("\n[1] Loading and filtering data...")
    t, q, dq, ddq, tau_meas = load_and_filter(csv_path, fs=fs, fc=fc_lpf)
    N = len(t)
    print(f"    Samples: {N},  duration: {t[-1]-t[0]:.2f} s")

    # --- Build stacked regressor --------------------------------------------
    print("\n[2] Building stacked regressor W (may take a while)...")
    W_list   = []
    tau_list = []
    for n in range(N):
        Wn = regressor(q[n], dq[n], ddq[n])   # (6, 78)
        W_list.append(Wn)
        tau_list.append(tau_meas[n])

    W_stacked_3d  = np.stack(W_list, axis=0)  # (N, 6, 78)
    tau_stacked_3d = np.stack(tau_list, axis=0)  # (N, 6)

    # Reshape to (N*6, 78) and (N*6,)
    W_full  = W_stacked_3d.reshape(N * N_JOINTS, N_PARAMS_T)
    tau_full = tau_stacked_3d.reshape(N * N_JOINTS)

    print(f"    W shape: {W_full.shape}")

    # --- Per-joint normalisation -------------------------------------------
    print("\n[3] Normalising by max torque per joint...")
    tau_max  = np.max(np.abs(tau_meas), axis=0)
    tau_max  = np.maximum(tau_max, 1e-3)
    scale_vec = np.tile(1.0 / tau_max, N)   # (N*6,) repeat for each sample
    W_norm   = W_full   * scale_vec[:, np.newaxis]
    tau_norm = tau_full * scale_vec
    print(f"    τ_max per joint [Nm]: {tau_max.round(3)}")

    # --- Base parameter reduction ------------------------------------------
    print("\n[4] Finding base parameters via QR...")
    base_cols, L_mat = find_base_parameters(W_norm)
    p = len(base_cols)
    print(f"    Full params: {N_PARAMS_T},  Base params: {p}")
    print(f"    Base columns (φ indices): {base_cols}")

    # --- Unconstrained least-squares baseline ----------------------------
    phi_ls, res, rnk, sv = np.linalg.lstsq(W_norm, tau_norm, rcond=None)
    tau_pred_ls = (W_full @ phi_ls).reshape(N, N_JOINTS)
    rel_ls = rel_metric(tau_meas, tau_pred_ls)
    print(f"\n    Unconstrained lstsq REL per joint: {rel_ls.round(4)}")
    print(f"    Mean REL: {rel_ls.mean():.4f}")

    # --- Constrained optimisation ------------------------------------------
    print("\n[5] Constrained identification (SLSQP)...")
    phi_init = initial_phi_guess()
    phi_id   = identify(W_norm, tau_norm, phi0=phi_init, verbose=verbose)

    # --- Evaluate -----------------------------------------------------------
    tau_pred_id = (W_full @ phi_id).reshape(N, N_JOINTS)
    rel_id      = rel_metric(tau_meas, tau_pred_id)
    print(f"\n[6] Constrained REL per joint: {rel_id.round(4)}")
    print(f"    Mean REL: {rel_id.mean():.4f}")

    # --- Print identified parameters ---------------------------------------
    print("\n[7] Identified friction parameters:")
    print(f"{'Joint':<12} {'Fv [Nm·s/rad]':>15} {'Fc [Nm]':>10} {'F0 [Nm]':>10}")
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        print(f"  {ARM_JOINTS[i]:<10} {phi_id[idx+10]:>15.4f} {phi_id[idx+11]:>10.4f} {phi_id[idx+12]:>10.4f}")

    # --- Check feasibility -------------------------------------------------
    print("\n[8] Feasibility check:")
    all_ok = True
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        phi_link = phi_id[idx: idx + N_PARAMS]
        m_ok     = phi_link[0] > 0
        Sigma    = pseudo_inertia(phi_link)
        eigs     = np.linalg.eigvalsh(Sigma)
        psd_ok   = np.all(eigs >= -1e-4)   # 1e-4 tolerance for floating-point noise
        fv_ok    = phi_link[10] >= 0
        fc_ok    = phi_link[11] >= 0
        ok       = m_ok and psd_ok and fv_ok and fc_ok
        all_ok   = all_ok and ok
        status   = "OK" if ok else "FAIL"
        print(f"  Link {i+1} ({ARM_JOINTS[i]}): m={phi_link[0]:.4f}, "
              f"min_eig(Σ)={eigs.min():.4f}, Fv={phi_link[10]:.4f}, Fc={phi_link[11]:.4f}  [{status}]")
    print(f"  All feasibility constraints satisfied: {all_ok}")

    # --- Plots --------------------------------------------------------------
    if plot:
        _plot_results(t, tau_meas, tau_pred_id, rel_id)

    return phi_id


def _plot_results(t: np.ndarray, tau_meas: np.ndarray,
                  tau_pred: np.ndarray, rel: np.ndarray) -> None:
    fig, axes = plt.subplots(N_JOINTS, 1, figsize=(12, 10), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, tau_meas[:, j], label='Measured', linewidth=0.8)
        ax.plot(t, tau_pred[:, j], '--', label='Predicted', linewidth=0.8)
        ax.set_ylabel(f"τ_{j+1} [Nm]")
        ax.set_title(f"{ARM_JOINTS[j]}   REL={rel[j]:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("ViperX-300 System Identification — Paper Method", fontsize=11)
    plt.tight_layout()
    plt.show()


# =============================================================================
# 11. Excitation trajectory demo
# =============================================================================

def demo_excitation(duration: float = 30.0, fs: float = 200.0) -> None:
    """Generate and plot a sample excitation trajectory."""
    t = np.linspace(0, duration, int(duration * fs))
    q, dq, ddq = excitation_trajectory(t)

    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    labels = [f"q_{j+1}" for j in range(N_JOINTS)]
    for j in range(N_JOINTS):
        axes[0].plot(t, np.degrees(q[:, j]),   label=labels[j])
        axes[1].plot(t, np.degrees(dq[:, j]),  label=labels[j])
        axes[2].plot(t, np.degrees(ddq[:, j]), label=labels[j])
    for ax, ylabel in zip(axes, ['Position [°]', 'Velocity [°/s]', 'Accel [°/s²]']):
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("Excitation Trajectory — Sum of Sinusoids", fontsize=11)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="ViperX-300 system identification following the paper method")
    parser.add_argument("csv", nargs="?", default="data/sysid_run1.csv",
                        help="CSV file from collect_arm_data.py (default: data/arm_data.csv)")
    parser.add_argument("--fs",    type=float, default=50.0,
                        help="Sampling rate Hz (default: 50)")
    parser.add_argument("--fc",    type=float, default=10.0,
                        help="LPF cutoff Hz (default: 10)")
    parser.add_argument("--demo-traj", action="store_true",
                        help="Plot a sample excitation trajectory and exit")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip plots")
    args = parser.parse_args()

    if args.demo_traj:
        demo_excitation()
    else:
        phi = run_identification(args.csv, fs=args.fs, fc_lpf=args.fc,
                                 plot=not args.no_plot)
        np.save("phi_identified.npy", phi)
        print("\nSaved identified parameters to phi_identified.npy")
