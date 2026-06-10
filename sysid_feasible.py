#!/usr/bin/env python3
"""
sysid_feasible.py — System Identification for ViperX-300 6-DOF Arm
====================================================================
Based on sysid_fast.py with complete physical feasibility constraints
as described in Momani & Hosseinzadeh (2025).

  Additional constraints over sysid_fast.py:
    1. Triangle inequalities on principal moments of I^c (eqs. 16d–f):
         λ₁ + λ₂ ≥ λ₃,  λ₂ + λ₃ ≥ λ₁,  λ₁ + λ₃ ≥ λ₂
       where λᵢ are eigenvalues of the inertia tensor at the centre of mass
       (I^c), computed via the parallel axis theorem.
       NOTE: sysid_fast.py enforced pseudo-inertia PD, which gives triangle
       inequalities on J (inertia at the link frame origin), not on I^c. This
       file enforces both.
    2. First mass moment sign constraints (eq. 16i, ViperX-300 geometry):
         m₂y₂ > 0,  m₃y₃ > 0,  m₄y₄ < 0,  m₅y₅ > 0,  m₆y₆ > 0
       (joints 2–6 in 1-indexed paper notation = links 1–5 in 0-indexed code)

pip install numpy scipy matplotlib pandas
"""

import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib.pyplot as plt
import pandas as pd
import os

import pipeline_artifacts

PIPELINE_NAME    = "sysid_feasible"
PIPELINE_VERSION = "1.4"   # 1.4: effort columns are mA; τ = (mA/1000)·2.409·motors (was wrongly %-of-stall)

# =============================================================================
# 1. DH kinematics  (unchanged from sysid_fast.py)
# =============================================================================

L1 = 0.12675
L2 = 0.30594
L3 = 0.21981
L4 = 0.08021
L5 = 0.07000
L6 = 0.13658

DH_PARAMS = np.array([
    [0.0,          0.0,      L1,           0.0          ],
    [3*np.pi/2,    0.0,      0.0,         -0.437*np.pi  ],
    [0.0,          L2,       0.0,         -0.063*np.pi  ],
    [3*np.pi/2,    0.0,      L3+L4,        0.0          ],
    [  np.pi/2,    0.0,      0.0,          0.0          ],
    [3*np.pi/2,    0.0,      0.0,          0.0          ],   # d_6 = 0 (fixed)
], dtype=float)

N_JOINTS   = 6
N_PARAMS   = 13
N_PARAMS_T = N_JOINTS * N_PARAMS   # 78


def _dh_transform(alpha, a, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ])


def forward_kinematics(q):
    T = [np.eye(4)]
    for i in range(N_JOINTS):
        alpha, a, d, theta_off = DH_PARAMS[i]
        T.append(T[-1] @ _dh_transform(alpha, a, d, q[i] + theta_off))
    return T


# =============================================================================
# 2a. NE forward pass  (unchanged)
# =============================================================================

def _ne_forward_pass(q, dq, ddq, T, R):
    g0 = np.array([0.0, 0.0, 9.81])
    omega  = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    domega = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    ddp    = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    ddp[0] = g0

    for i in range(1, N_JOINTS + 1):
        Rp   = R[i-1]; Ri = R[i]
        Rrel = Rp.T @ Ri
        zi_1_loc   = Rrel.T @ np.array([0.0, 0.0, 1.0])
        omega_loc  = Rrel.T @ omega[i-1]
        domega_loc = Rrel.T @ domega[i-1]
        omega[i]   = omega_loc + dq[i-1] * zi_1_loc
        domega[i]  = domega_loc + ddq[i-1] * zi_1_loc + np.cross(omega_loc, dq[i-1] * zi_1_loc)
        r_loc = Ri.T @ (T[i][:3, 3] - T[i-1][:3, 3])
        ddp[i] = (Ri.T @ (Rp @ ddp[i-1])
                  + np.cross(domega[i], r_loc)
                  + np.cross(omega[i], np.cross(omega[i], r_loc)))

    return omega, domega, ddp


# =============================================================================
# 2b. Vectorised NE backward pass  (unchanged)
# =============================================================================

def regressor_fast(q, dq, ddq):
    T = forward_kinematics(q)
    R = [T[i][:3, :3] for i in range(N_JOINTS + 1)]
    omega, domega, ddp = _ne_forward_pass(q, dq, ddq, T, R)

    P = N_PARAMS_T
    f_mat = np.zeros((3, P))
    n_mat = np.zeros((3, P))
    W     = np.zeros((N_JOINTS, P))

    EX = np.array([1.0, 0.0, 0.0])
    EY = np.array([0.0, 1.0, 0.0])
    EZ = np.array([0.0, 0.0, 1.0])

    for i in range(N_JOINTS, 0, -1):
        li   = i - 1
        base = li * N_PARAMS

        ω = omega[i]; dω = domega[i]; d = ddp[i]

        fi_cols = np.zeros((3, N_PARAMS))
        fi_cols[:, 0] = d
        for k, e in enumerate((EX, EY, EZ)):
            fi_cols[:, 1+k] = np.cross(dω, e) + np.cross(ω, np.cross(ω, e))

        ni_cols = np.zeros((3, N_PARAMS))
        for k, e in enumerate((EX, EY, EZ)):
            ni_cols[:, 1+k] = np.cross(e, d)

        Jdw = np.array([dω[0], 0.0,   0.0  ]); Jw = np.array([ω[0], 0.0,  0.0  ])
        ni_cols[:, 4] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([dω[1], dω[0], 0.0  ]); Jw = np.array([ω[1], ω[0], 0.0  ])
        ni_cols[:, 5] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([dω[2], 0.0,   dω[0]]); Jw = np.array([ω[2], 0.0,  ω[0] ])
        ni_cols[:, 6] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([0.0,   dω[1], 0.0  ]); Jw = np.array([0.0,  ω[1], 0.0  ])
        ni_cols[:, 7] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([0.0,   dω[2], dω[1]]); Jw = np.array([0.0,  ω[2], ω[1] ])
        ni_cols[:, 8] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([0.0,   0.0,   dω[2]]); Jw = np.array([0.0,  0.0,  ω[2] ])
        ni_cols[:, 9] = Jdw + np.cross(ω, Jw)

        fi_full = np.zeros((3, P)); fi_full[:, base:base+N_PARAMS] = fi_cols
        ni_full = np.zeros((3, P)); ni_full[:, base:base+N_PARAMS] = ni_cols

        if i < N_JOINTS:
            Rrel_next = R[i].T @ R[i+1]
            r_next    = R[i].T @ (T[i+1][:3, 3] - T[i][:3, 3])
            f_from_next = Rrel_next @ f_mat
            n_from_next = Rrel_next @ n_mat + np.cross(r_next, f_from_next.T).T
        else:
            f_from_next = np.zeros((3, P))
            n_from_next = np.zeros((3, P))

        f_mat = fi_full + f_from_next
        n_mat = ni_full + n_from_next

        W[li, :] = n_mat[2, :]
        W[li, base+10] += dq[li]
        W[li, base+11] += np.sign(dq[li])
        W[li, base+12] += 1.0

    return W


# =============================================================================
# 2c. Scalar NE (kept for validation)
# =============================================================================

def _inertia_at_origin(phi_link):
    Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = phi_link[4:10]
    return np.array([[Jxx, Jxy, Jxz], [Jxy, Jyy, Jyz], [Jxz, Jyz, Jzz]])


def inverse_dynamics_phi(q, dq, ddq, phi):
    T = forward_kinematics(q)
    R = [T[i][:3, :3] for i in range(N_JOINTS + 1)]
    omega, domega, ddp = _ne_forward_pass(q, dq, ddq, T, R)

    f_ext = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    n_ext = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    tau   = np.zeros(N_JOINTS)

    for i in range(N_JOINTS, 0, -1):
        idx   = i - 1
        phi_i = phi[idx*N_PARAMS:(idx+1)*N_PARAMS]
        mi    = phi_i[0]; mci = phi_i[1:4]; Ji = _inertia_at_origin(phi_i)

        fi = mi * ddp[i] + np.cross(domega[i], mci) + np.cross(omega[i], np.cross(omega[i], mci))
        ni = Ji @ domega[i] + np.cross(omega[i], Ji @ omega[i]) + np.cross(mci, ddp[i])

        if i < N_JOINTS:
            Rrel_next = R[i].T @ R[i+1]
            r_next    = R[i].T @ (T[i+1][:3, 3] - T[i][:3, 3])
            f_next = Rrel_next @ f_ext[i+1]
            n_next = Rrel_next @ n_ext[i+1] + np.cross(r_next, f_next)
        else:
            f_next = np.zeros(3); n_next = np.zeros(3)

        f_ext[i] = fi + f_next
        n_ext[i] = ni + n_next
        tau[idx]  = n_ext[i][2] + phi_i[10]*dq[idx] + phi_i[11]*np.sign(dq[idx]) + phi_i[12]

    return tau


# =============================================================================
# 3. Base parameter reduction  (unchanged)
# =============================================================================

def find_base_parameters(W_stacked, tol=1e-10):
    Q, R_mat, P = la.qr(W_stacked, pivoting=True, mode='economic')
    rank = np.sum(np.abs(np.diag(R_mat)) > tol * np.abs(R_mat[0, 0]))
    base_cols = np.sort(P[:rank])
    W_base = W_stacked[:, base_cols]
    L = np.linalg.lstsq(W_base, W_stacked, rcond=None)[0].T
    return base_cols, L


# =============================================================================
# 4. Data loading + filtering  (unchanged)
# =============================================================================

# Effort→torque conversion.
# The `*_effort` columns are the Dynamixel "present current" reported by the
# interbotix driver, in milliamps (mA), for a single (master) motor. Motor
# torque is proportional to current:  τ = k_t · I  (k_t = 2.409 Nm/A for the
# XM540-W270, gearbox included — see control/trq.py).
#   τ[Nm] = (effort[mA] / 1000) · k_t · (motors on that joint)
# The shoulder and elbow are dual-motor joints (a master + a mirrored shadow
# motor); the driver reports only the master's current, so their joint torque
# is twice the per-motor torque. This 2× on the shoulder was confirmed
# empirically by regressing raw effort against the URDF gravity torque.
TORQUE_CONSTANT  = 2.409                              # Nm/A  (XM540-W270, geared)
MOTORS_PER_JOINT = np.array([1, 2, 2, 1, 1, 1])       # waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
EFFORT_SCALE     = (TORQUE_CONSTANT / 1000.0) * MOTORS_PER_JOINT   # mA → Nm
ARM_JOINTS   = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def load_and_filter(csv_path, fs=50.0, fc=10.0, stride=1):
    df = pd.read_csv(csv_path)
    t  = df["time"].values.astype(float); t -= t[0]

    q_raw   = df[[f"{j}_pos"    for j in ARM_JOINTS]].values.astype(float)
    tau_raw = df[[f"{j}_effort" for j in ARM_JOINTS]].values.astype(float) * EFFORT_SCALE

    fs_actual = 1.0 / float(np.median(np.diff(t)))
    if abs(fs_actual - fs) > 2.0:
        print(f"    [warn] --fs={fs:.1f} but CSV is {fs_actual:.1f} Hz")
    fs = fs_actual

    nyq  = fs / 2.0
    Wn   = min(fc / nyq, 0.99)
    b, a = sig.butter(4, Wn, btype='low')

    q_filt   = np.vstack([sig.filtfilt(b, a, q_raw[:,j])   for j in range(N_JOINTS)]).T
    tau_filt = np.vstack([sig.filtfilt(b, a, tau_raw[:,j]) for j in range(N_JOINTS)]).T
    dt  = np.gradient(t)
    dq  = np.gradient(q_filt, axis=0) / dt[:, np.newaxis]
    dq  = np.vstack([sig.filtfilt(b, a, dq[:,j]) for j in range(N_JOINTS)]).T
    ddq = np.gradient(dq, axis=0) / dt[:, np.newaxis]
    ddq = np.vstack([sig.filtfilt(b, a, ddq[:,j]) for j in range(N_JOINTS)]).T

    if stride > 1:
        t, q_filt, dq, ddq, tau_filt = (
            t[::stride], q_filt[::stride], dq[::stride], ddq[::stride], tau_filt[::stride]
        )
    return t, q_filt, dq, ddq, tau_filt


# =============================================================================
# 5. Feasibility helpers
# =============================================================================

def pseudo_inertia(phi_link):
    m, mcx, mcy, mcz, Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = phi_link[:10]
    return np.array([
        [(Jyy+Jzz-Jxx)/2,  -Jxy,             -Jxz,            mcx],
        [-Jxy,              (Jxx+Jzz-Jyy)/2,  -Jyz,            mcy],
        [-Jxz,              -Jyz,              (Jxx+Jyy-Jzz)/2, mcz],
        [mcx,                mcy,               mcz,             m  ],
    ])


def inertia_at_com(phi_link):
    """
    Inertia tensor I^c at the centre of mass via the parallel axis theorem:
        I^c = J - (1/m) * (|mc|² I₃ - mc mcᵀ)
    where mc = m·C is the first mass moment vector and J is the inertia tensor
    expressed at the link frame origin.
    """
    m   = max(phi_link[0], 1e-8)   # guard against division by zero
    mc  = phi_link[1:4]
    J   = _inertia_at_origin(phi_link)
    mc_sq = float(np.dot(mc, mc))
    return J - (1.0 / m) * (mc_sq * np.eye(3) - np.outer(mc, mc))


def triangle_ineq_values(phi_link):
    """
    Returns three values (each must be ≥ 0) encoding the triangle inequalities
    on the principal moments of I^c (eqs. 16d–f in the paper):
        λ₁ + λ₂ - λ₃ ≥ 0
        λ₂ + λ₃ - λ₁ ≥ 0
        λ₁ + λ₃ - λ₂ ≥ 0
    with eigenvalues sorted in ascending order (λ₁ ≤ λ₂ ≤ λ₃).
    """
    I_c = inertia_at_com(phi_link)
    lam = np.sort(np.linalg.eigvalsh(I_c))
    l1, l2, l3 = lam
    return np.array([l1 + l2 - l3, l2 + l3 - l1, l1 + l3 - l2])


# Sign conventions for mcy (= m·yCoM) derived from ViperX-300 frame geometry
# (paper Section 3.4 / eq. 16i).  Format: (link_0idx, required_sign)
#   Paper notation  →  0-indexed link  →  mcy index in phi
#   m₂y₂ > 0       →  link 1          →  phi[1*13 + 2]
#   m₃y₃ > 0       →  link 2          →  phi[2*13 + 2]
#   m₄y₄ < 0       →  link 3          →  phi[3*13 + 2]  (sign=-1)
#   m₅y₅ > 0       →  link 4          →  phi[4*13 + 2]
#   m₆y₆ > 0       →  link 5          →  phi[5*13 + 2]
_MCY_SIGN_CONSTRAINTS = [
    (1, +1),
    (2, +1),
    (3, -1),
    (4, +1),
    (5, +1),
]


def feasibility_constraints(phi, with_ic_pd=True):
    """
    Physical feasibility constraints (each 'ineq' entry must be ≥ 0).

    Per-link always-on (6 × 4 = 24 constraints):
      • mass > 0
      • pseudo-inertia 4×4 PD  (min eigenvalue ≥ ε)
      • Fv ≥ 0
      • Fc ≥ 0

    Per-link I^c PD — only when with_ic_pd=True (6 × 1 = 6 constraints, eq. 16c):
      • min eigenvalue of I^c ≥ ε
        Stage 1 omits this so F0/friction converge first; Stage 2 adds it.

    Per-link triangle inequalities on I^c (6 × 3 = 18 constraints, eqs. 16d–f):
      • λ₁(I^c) + λ₂(I^c) ≥ λ₃(I^c)
      • λ₂(I^c) + λ₃(I^c) ≥ λ₁(I^c)
      • λ₁(I^c) + λ₃(I^c) ≥ λ₂(I^c)

    First mass moment sign constraints (5 constraints, eq. 16i).
    """
    constraints = []

    for i in range(N_JOINTS):
        idx = i * N_PARAMS

        # mass positivity
        constraints.append({'type': 'ineq', 'fun': lambda phi, j=idx: phi[j] - 1e-4})

        # pseudo-inertia 4×4 positive definite (covers J at frame origin)
        def min_eig_pseudo(phi, idx=idx):
            return np.min(np.linalg.eigvalsh(pseudo_inertia(phi[idx:idx+N_PARAMS]))) - 1e-6
        constraints.append({'type': 'ineq', 'fun': min_eig_pseudo})

        # friction non-negative
        constraints.append({'type': 'ineq', 'fun': lambda phi, j=idx+10: phi[j]})
        constraints.append({'type': 'ineq', 'fun': lambda phi, j=idx+11: phi[j]})

        # paper eq. 16c: I^c ≻ 0 — added only in Stage 2
        if with_ic_pd:
            def min_eig_ic(phi, idx=idx):
                return np.min(np.linalg.eigvalsh(inertia_at_com(phi[idx:idx+N_PARAMS]))) - 1e-6
            constraints.append({'type': 'ineq', 'fun': min_eig_ic})

        # triangle inequalities on principal moments of I^c (eqs. 16d–f)
        for k in range(3):
            def tri_ineq(phi, idx=idx, k=k):
                return triangle_ineq_values(phi[idx:idx+N_PARAMS])[k]
            constraints.append({'type': 'ineq', 'fun': tri_ineq})

    # --- NEW: first mass moment sign constraints (eq. 16i) ---
    for link_idx, sign in _MCY_SIGN_CONSTRAINTS:
        mcy_idx = link_idx * N_PARAMS + 2
        constraints.append({
            'type': 'ineq',
            'fun': lambda phi, j=mcy_idx, s=sign: s * phi[j] - 1e-5,
        })

    return constraints


# =============================================================================
# 6. Initial guess  (updated: mcy seeded to satisfy sign constraints)
# =============================================================================

def initial_phi_guess():
    Fv = [0.126, 0.101, 0.0403, 0.0234, 0.0166, 0.00158]
    Fc = [0.239, 0.412, 0.261,  0.0303, 0.0608, 0.00295]
    F0 = [-0.0529, -0.203, -0.766, 0.0119, -0.0127, -0.00469]
    masses = [0.8, 0.8, 0.5, 0.3, 0.2, 0.1]

    # mcy seeds: must respect sign constraints from eq. 16i.
    # Link 0 has no sign constraint so 0 is fine.
    mcy_seeds = [0.0, 0.01, 0.01, -0.01, 0.01, 0.01]

    phi0 = np.zeros(N_PARAMS_T)
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        m = masses[i]
        phi0[idx+0]  = m
        phi0[idx+2]  = mcy_seeds[i]          # mcy seeded for feasibility
        phi0[idx+4]  = m * 0.01              # Jxx
        phi0[idx+7]  = m * 0.01              # Jyy
        phi0[idx+9]  = m * 0.01              # Jzz
        phi0[idx+10] = Fv[i]
        phi0[idx+11] = Fc[i]
        phi0[idx+12] = F0[i]
    return phi0


# =============================================================================
# 7. Optimisation
# =============================================================================

def rel_metric(tau_true, tau_pred):
    denom = np.maximum(np.abs(tau_true), np.abs(tau_pred))
    denom = np.where(denom < 1e-6, 1e-6, denom)
    return np.mean(np.abs(tau_true - tau_pred) / denom, axis=0)


def identify(W_base, tau_stacked, L_mat, phi0=None, w1=1.0, w2=5e-3,
             method='SLSQP', verbose=True, with_ic_pd=True):
    """
    Paper eq. 16: joint optimisation over (phi_b, phi).

    min  w1*||W_base @ phi_b - tau||^2 + w2*||phi_b - L.T @ phi||^2
    s.t. feasibility constraints on phi (standard parameters)

    W_base     : (N_eq, rank)       full-rank base regressor
    L_mat      : (N_PARAMS_T, rank) L s.t. W_full ≈ W_base @ L.T
    phi0       : (N_PARAMS_T,)      initial standard-parameter guess (warm-start)
    with_ic_pd : include I^c ≻ 0 constraint (paper eq. 16c).
                 Set False for Stage 1 so F0/friction converge without fighting
                 the inertia PD constraint; True for Stage 2 warm-started from
                 Stage 1's solution.

    Returns phi (N_PARAMS_T,) — standard parameters.
    """
    rank = W_base.shape[1]

    if phi0 is None:
        phi0 = initial_phi_guess()

    # Start with phi_b consistent with phi0 so the coupling term begins at zero.
    phi_b0 = L_mat.T @ phi0
    x0 = np.concatenate([phi_b0, phi0])

    def objective(x):
        phi_b = x[:rank]
        phi   = x[rank:]
        r = W_base @ phi_b - tau_stacked
        c = phi_b - L_mat.T @ phi
        return w1 * float(r @ r) + w2 * float(c @ c)

    def gradient(x):
        phi_b = x[:rank]
        phi   = x[rank:]
        r = W_base @ phi_b - tau_stacked
        c = phi_b - L_mat.T @ phi
        g_phi_b = 2*w1 * (W_base.T @ r) + 2*w2 * c
        g_phi   = -2*w2 * (L_mat @ c)
        return np.concatenate([g_phi_b, g_phi])

    # Feasibility constraints act on phi = x[rank:] only.
    raw_cons = feasibility_constraints(None, with_ic_pd=with_ic_pd)

    def wrap(fun):
        return lambda x: fun(x[rank:])

    if method == 'trust-constr':
        from scipy.optimize import NonlinearConstraint
        cons = [NonlinearConstraint(wrap(c['fun']), 0.0, np.inf, jac='2-point')
                for c in raw_cons]
        opts = {'maxiter': 2000, 'gtol': 1e-4, 'verbose': 0}
    else:
        cons = [{'type': c['type'], 'fun': wrap(c['fun'])} for c in raw_cons]
        # eps=1e-4: larger FD step stabilises gradients of eigenvalue constraints.
        opts = {'maxiter': 1000, 'ftol': 1e-6, 'disp': False, 'eps': 1e-4}

    ic_label = '+I^c PD' if with_ic_pd else 'no I^c PD'
    if verbose:
        print(f"  Starting {method} ({len(raw_cons)} constraints [{ic_label}], "
              f"joint variable: {rank} base + {N_PARAMS_T} standard = {rank+N_PARAMS_T})...")

    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        result = opt.minimize(objective, x0, jac=gradient, method=method,
                              constraints=cons, options=opts)

    if verbose:
        status = 'converged' if result.success else f'stopped ({result.message})'
        print(f"  {status} — cost: {result.fun:.3f}")
    return result.x[rank:]   # return standard parameters phi


def identify_sdp(W_base, tau_stacked, L_mat, w1=1.0, w2=5e-3, verbose=True,
                 entropic_gamma=0.0, solver=None,
                 ref_mass=0.5, ref_inertia=1e-3):
    """
    Convex SDP form of the paper's identification problem (eq. 16), solved with
    CVXPY. This is the paper-faithful route (the paper uses YALMIP, i.e. an SDP
    solver): the physical-consistency condition is the *linear matrix inequality*
    that the 4×4 pseudo-inertia P_i(phi) is positive semidefinite (Wensing,
    Kim & Slotine 2017 — paper ref [53]). That single LMI subsumes mass>0,
    I^c ≻ 0 (16c) AND the triangle inequalities (16d–f). With convex constraints,
    eq. 16 has a unique global optimum, so it avoids both NLP failure modes
    (trust-constr stalling, SLSQP escaping feasibility).

        min  w1·||W_base φ_b − τ||² + w2·||φ_b − Lᵀ φ||²
        s.t. P_i(φ) ⪰ 0  ∀i           (physical consistency, 16b–f)
             Fv_i ≥ 0, Fc_i ≥ 0  ∀i    (16g–h)
             sign(m_i·y_i) constraints (16i)

    No reference/CAD model is used — feasibility is resolved purely by physics,
    so the method applies to a robot that has no pre-existing dynamic model.

    The data term is rewritten via the Gram matrix (WᵀW, Wᵀτ) so the problem size
    is independent of the number of samples.

    `entropic_gamma > 0` adds the **model-free log-det (Bregman) divergence
    regulariser** (Wensing, Kim & Slotine 2017 — ref [53]):

        +γ·Σ_i [ tr(P0⁻¹·P_i) − log det P_i ]       (convex, ≥0, min at P_i = P0)

    The base parameters fix what the data determines; the *standard* parameters
    are a non-unique realisation, and the unregularised SDP collapses the
    unidentifiable masses toward zero (PSD but degenerate), which blocks URDF
    export. The reference P0 = diag(ref_inertia·I₃, ref_mass) is a **generic
    isotropic blob — the SAME uninformative shape for every link, NOT the CAD
    model**, so the demonstration stays "from data alone". Both terms matter:
    `−log det P` (concave entropy; minimising its negative is convex → SDP stays
    convex) lifts masses *off* zero, and `tr(P0⁻¹·P)` *bounds them from above*
    (pure `−log det P` is unbounded and explodes the free masses → ∞). The
    minimiser P_i = P0 means free masses settle near `ref_mass` and CoMs near 0.
    γ=0 reproduces the exact paper Eq. 16; pick the smallest γ that gives
    non-degenerate masses so the torque fit (REL) is essentially unperturbed.

    Returns phi (N_PARAMS_T,) — standard parameters.
    """
    try:
        import cvxpy as cp
    except ImportError as e:
        raise ImportError(
            "method='cvxpy' needs cvxpy with an SDP-capable backend (SCS/Clarabel).\n"
            "  Install with:  pip install cvxpy"
        ) from e

    rank = W_base.shape[1]
    G = W_base.T @ W_base
    G = 0.5 * (G + G.T)                       # symmetrise against float drift
    b = W_base.T @ tau_stacked
    const = float(tau_stacked @ tau_stacked)

    phi_b = cp.Variable(rank)
    phi   = cp.Variable(N_PARAMS_T)

    data_term = cp.quad_form(phi_b, cp.psd_wrap(G)) - 2.0 * (b @ phi_b) + const
    coupling  = cp.sum_squares(phi_b - L_mat.T @ phi)

    EPS = 1e-6
    cons    = []
    P_list  = []
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        m             = phi[idx + 0]
        mcx, mcy, mcz = phi[idx + 1], phi[idx + 2], phi[idx + 3]
        Jxx, Jxy, Jxz = phi[idx + 4], phi[idx + 5], phi[idx + 6]
        Jyy, Jyz, Jzz = phi[idx + 7], phi[idx + 8], phi[idx + 9]
        # 4×4 pseudo-inertia — must match pseudo_inertia(); affine & symmetric.
        P = cp.bmat([
            [(Jyy + Jzz - Jxx) / 2, -Jxy,                  -Jxz,                  mcx],
            [-Jxy,                  (Jxx + Jzz - Jyy) / 2,  -Jyz,                  mcy],
            [-Jxz,                  -Jyz,                   (Jxx + Jyy - Jzz) / 2, mcz],
            [mcx,                    mcy,                    mcz,                  m  ],
        ])
        cons.append(P >> EPS * np.eye(4))     # physical-consistency LMI
        cons.append(phi[idx + 10] >= 0)       # Fv ≥ 0
        cons.append(phi[idx + 11] >= 0)       # Fc ≥ 0
        P_list.append(P)

    for link_idx, sign in _MCY_SIGN_CONSTRAINTS:
        cons.append(sign * phi[link_idx * N_PARAMS + 2] >= 1e-5)

    obj_expr = w1 * data_term + w2 * coupling
    if entropic_gamma > 0:
        # Bounded log-det (Bregman) divergence to a GENERIC isotropic reference
        # P0 = diag(ref_inertia, ref_inertia, ref_inertia, ref_mass) — same blob
        # for every link, NOT the CAD model. tr(P0⁻¹·P) bounds masses from above
        # (the part pure −log det P lacked); −log det P lifts them off zero.
        P0_inv = np.array([1.0 / ref_inertia] * 3 + [1.0 / ref_mass])
        div = cp.sum([cp.sum(cp.multiply(P0_inv, cp.diag(P))) - cp.log_det(P)
                      for P in P_list])
        obj_expr = obj_expr + entropic_gamma * div
    objective = cp.Minimize(obj_expr)

    prob = cp.Problem(objective, cons)
    if verbose:
        reg = (f", log-det div γ={entropic_gamma:g} → P0=diag({ref_inertia:g}·I₃,{ref_mass:g})"
               if entropic_gamma > 0 else "")
        print(f"  Solving SDP ({N_JOINTS} 4×4 LMIs + {len(cons) - N_JOINTS} linear, "
              f"{rank} base + {N_PARAMS_T} standard vars{reg})...")
    prob.solve(**({} if solver is None else {"solver": solver}))
    if phi.value is None:
        raise RuntimeError(f"SDP did not solve (status={prob.status}).")
    if verbose:
        used = getattr(prob.solver_stats, "solver_name", "?")
        print(f"  status: {prob.status} — cost: {prob.value:.3f} (solver: {used})")
    return np.asarray(phi.value).ravel()


# =============================================================================
# 8. Full pipeline
# =============================================================================

def run_identification(csv_path, fs=50.0, fc_lpf=10.0, stride=1,
                       verbose=True, plot=True, method='trust-constr',
                       w1=1.0, w2=5e-3, entropic_gamma=0.0, solver=None,
                       ref_mass=0.5, ref_inertia=1e-3):
    print("=" * 60)
    print(f"ViperX-300 SysID — Full feasibility (stride={stride})")
    print("=" * 60)

    print("\n[1] Loading and filtering data...")
    t, q, dq, ddq, tau_meas = load_and_filter(csv_path, fs=fs, fc=fc_lpf, stride=stride)
    N = len(t)
    print(f"    Samples: {N},  duration: {t[-1]-t[0]:.2f} s")

    print("\n[2] Building stacked regressor W (fast NE)...")
    W_rows = np.empty((N * N_JOINTS, N_PARAMS_T))
    for n in range(N):
        if verbose and n % max(1, N//10) == 0:
            print(f"    sample {n}/{N}", end="\r")
        W_rows[n*N_JOINTS:(n+1)*N_JOINTS, :] = regressor_fast(q[n], dq[n], ddq[n])
    print()
    tau_full = tau_meas.reshape(N * N_JOINTS)
    print(f"    W shape: {W_rows.shape}")

    print("\n[3] Normalising...")
    tau_max   = np.maximum(np.max(np.abs(tau_meas), axis=0), 1e-3)
    scale_vec = np.tile(1.0 / tau_max, N)
    W_norm    = W_rows    * scale_vec[:, np.newaxis]
    tau_norm  = tau_full  * scale_vec
    print(f"    τ_max per joint [Nm]: {tau_max.round(3)}")

    print("\n[4] Finding base parameters via QR...")
    base_cols, L_mat = find_base_parameters(W_norm)
    print(f"    Full: {N_PARAMS_T},  Base: {len(base_cols)}")

    phi_ls, *_ = np.linalg.lstsq(W_norm, tau_norm, rcond=None)
    rel_ls = rel_metric(tau_meas, (W_rows @ phi_ls).reshape(N, N_JOINTS))
    print(f"\n    Unconstrained lstsq REL: {rel_ls.round(4)}  mean={rel_ls.mean():.4f}")

    W_base_norm = W_norm[:, base_cols]

    if method == 'cvxpy':
        # --- Convex SDP solve (paper-faithful): single global optimum ---
        print(f"\n[5] Identifying — convex SDP via CVXPY...")
        phi_id = identify_sdp(W_base_norm, tau_norm, L_mat,
                              w1=w1, w2=w2, verbose=verbose,
                              entropic_gamma=entropic_gamma, solver=solver,
                              ref_mass=ref_mass, ref_inertia=ref_inertia)
    else:
        # --- Stage 1: no I^c PD — anchors F0 and friction values ---
        print(f"\n[5a] Stage 1 — {method}, without I^c PD (anchors F0/friction)...")
        phi_s1 = identify(W_base_norm, tau_norm, L_mat, phi0=initial_phi_guess(),
                          verbose=verbose, method=method, w1=w1, w2=w2,
                          with_ic_pd=False)

        tau_pred_s1 = (W_rows @ phi_s1).reshape(N, N_JOINTS)
        rel_s1 = rel_metric(tau_meas, tau_pred_s1)
        print(f"    Stage 1 REL: {rel_s1.round(4)}  mean={rel_s1.mean():.4f}")

        # --- Stage 2: add I^c PD, warm-started from Stage 1 ---
        print(f"\n[5b] Stage 2 — {method}, +I^c PD, warm-started from Stage 1...")
        phi_id = identify(W_base_norm, tau_norm, L_mat, phi0=phi_s1,
                          verbose=verbose, method=method, w1=w1, w2=w2,
                          with_ic_pd=True)

    tau_pred_id = (W_rows @ phi_id).reshape(N, N_JOINTS)
    rel_id = rel_metric(tau_meas, tau_pred_id)
    print(f"\n[6] Final REL: {rel_id.round(4)}  mean={rel_id.mean():.4f}")

    print("\n[7] Identified friction parameters:")
    print(f"{'Joint':<14} {'Fv':>10} {'Fc':>10} {'F0':>10}")
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        print(f"  {ARM_JOINTS[i]:<12} {phi_id[idx+10]:>10.4f} {phi_id[idx+11]:>10.4f} {phi_id[idx+12]:>10.4f}")

    print("\n[8] Feasibility check:")
    all_ok = True
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        pl  = phi_id[idx:idx+N_PARAMS]

        eigs_pseudo = np.linalg.eigvalsh(pseudo_inertia(pl))
        tri         = triangle_ineq_values(pl)
        I_c         = inertia_at_com(pl)
        eigs_ic     = np.sort(np.linalg.eigvalsh(I_c))

        mass_ok  = pl[0] > 0
        psd_ok   = np.all(eigs_pseudo >= -1e-4)
        ic_pd_ok = eigs_ic.min() >= -1e-4          # paper eq. 16c: I^c ≻ 0
        tri_ok   = np.all(tri >= -1e-4)
        fric_ok  = pl[10] >= -1e-4 and pl[11] >= -1e-4   # same slack as the PSD checks
        link_ok  = mass_ok and psd_ok and ic_pd_ok and tri_ok and fric_ok
        all_ok  &= link_ok

        print(f"  Link {i+1} ({ARM_JOINTS[i]}):")
        print(f"    m={pl[0]:.4f}  mcy={pl[2]:.5f}  "
              f"pseudo_min_eig={eigs_pseudo.min():.4f}  "
              f"Fv={pl[10]:.4f}  Fc={pl[11]:.4f}  [{'OK' if link_ok else 'FAIL'}]")
        print(f"    I^c eigenvalues: {eigs_ic.round(5)}  min={eigs_ic.min():.5f}  "
              f"[{'OK' if ic_pd_ok else 'FAIL (16c)'}]")
        print(f"    triangle slack: {tri.round(5)}  [{'OK' if tri_ok else 'FAIL (16d-f)'}]")

    print("\n  mcy sign constraints (eq. 16i):")
    for link_idx, sign in _MCY_SIGN_CONSTRAINTS:
        mcy = phi_id[link_idx * N_PARAMS + 2]
        satisfied = sign * mcy > 0
        all_ok &= satisfied
        label = f"m{link_idx+1}·y{link_idx+1} {'>' if sign>0 else '<'} 0"
        print(f"    Link {link_idx+1}: {label}  mcy={mcy:.5f}  [{'OK' if satisfied else 'FAIL'}]")

    print(f"\n  Overall: [{'ALL OK' if all_ok else 'SOME FAILED'}]")

    if plot:
        _plot_results(t, tau_meas, tau_pred_id, rel_id)
    return phi_id


def _plot_results(t, tau_meas, tau_pred, rel):
    fig, axes = plt.subplots(N_JOINTS, 1, figsize=(12, 10), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, tau_meas[:, j], label='Measured', lw=0.8)
        ax.plot(t, tau_pred[:, j], '--', label='Predicted', lw=0.8)
        ax.set_ylabel(f"τ_{j+1} [Nm]")
        ax.set_title(f"{ARM_JOINTS[j]}  REL={rel[j]:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("ViperX-300 SysID — Full feasibility", fontsize=11)
    plt.tight_layout(); plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="ViperX-300 sysid — full feasibility")
    parser.add_argument("csv",       nargs="?", default="data/sysid_run1.csv")
    parser.add_argument("--fs",      type=float, default=50.0,   help="Sample rate Hz")
    parser.add_argument("--fc",      type=float, default=10.0,   help="LPF cutoff Hz")
    parser.add_argument("--stride",  type=int,   default=1,      help="Subsample stride")
    parser.add_argument("--w1",      type=float, default=1.0,    help="Data-fit weight")
    parser.add_argument("--w2",      type=float, default=5e-3,   help="Coupling weight")
    parser.add_argument("--method",  default="trust-constr",
                        choices=["trust-constr", "SLSQP", "cvxpy"],
                        help="Optimizer: trust-constr / SLSQP (SciPy NLP) or "
                             "cvxpy (convex SDP, paper-faithful; needs `pip install cvxpy`)")
    parser.add_argument("--entropic", type=float, default=0.0,
                        help="Log-det (Bregman) divergence regulariser weight γ "
                             "(cvxpy only; 0 = exact paper Eq. 16). >0 gives "
                             "non-degenerate link masses for URDF export by pulling "
                             "each pseudo-inertia toward a generic blob P0 — "
                             "model-free, no CAD prior. Use the smallest γ that "
                             "lifts masses off zero while REL stays ≈ baseline.")
    parser.add_argument("--ref-mass", type=float, default=0.5,
                        help="Generic reference link mass [kg] for the log-det "
                             "divergence target P0 (same blob for every link).")
    parser.add_argument("--ref-inertia", type=float, default=1e-3,
                        help="Generic reference link inertia [kg·m²] for P0.")
    parser.add_argument("--solver", default=None,
                        help="Override CVXPY solver (e.g. CLARABEL, SCS). "
                             "CLARABEL (interior-point) is more accurate near the "
                             "feasibility margins than the default SCS.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--force",   action="store_true",
                        help="Recompute even if an identical artifact already exists")
    parser.add_argument("--outputs-dir", default=None,
                        help="Override output root directory (default: outputs/)")
    parser.add_argument("--migrate-legacy", action="store_true",
                        help="Import old npy/ files into the new outputs/ structure and exit")
    args = parser.parse_args()

    if args.migrate_legacy:
        n = pipeline_artifacts.migrate_legacy(outputs_root=args.outputs_dir or "outputs")
        print(f"Migrated {n} file(s) from npy/ → outputs/legacy/")
        sys.exit(0)

    # Build the config dict — every parameter that affects the output numerics.
    config = {
        "fs":     args.fs,
        "fc_lpf": args.fc,
        "stride": args.stride,
        "method": args.method,
        "w1":     args.w1,
        "w2":     args.w2,
        "entropic":    args.entropic,
        "solver":      args.solver,
        "ref_mass":    args.ref_mass,
        "ref_inertia": args.ref_inertia,
        # Effort→torque conversion (see EFFORT_SCALE). Recorded here so the
        # provenance sidecar and config hash capture the units; a change to the
        # torque constant or motor counts now yields a distinct artifact.
        "torque_constant":  TORQUE_CONSTANT,
        "motors_per_joint": MOTORS_PER_JOINT.tolist(),
    }

    # Cache-hit check: skip expensive computation if the artifact already exists.
    if not args.force:
        cached = pipeline_artifacts.load_artifact(
            args.csv, PIPELINE_NAME, PIPELINE_VERSION, config, args.outputs_dir
        )
        if cached is not None:
            npy_p, _ = pipeline_artifacts.artifact_path(
                args.csv, PIPELINE_NAME, PIPELINE_VERSION, config, args.outputs_dir
            )
            print(f"\n[cache] Artifact already exists — skipping computation.")
            print(f"  {npy_p}")
            print("  Pass --force to recompute.")
            sys.exit(0)

    phi = run_identification(
        args.csv,
        fs=args.fs, fc_lpf=args.fc, stride=args.stride,
        plot=not args.no_plot, method=args.method,
        w1=args.w1, w2=args.w2,
        entropic_gamma=args.entropic, solver=args.solver,
        ref_mass=args.ref_mass, ref_inertia=args.ref_inertia,
    )

    npy_path, json_path = pipeline_artifacts.save_artifact(
        phi, args.csv, PIPELINE_NAME, PIPELINE_VERSION, config,
        outputs_root=args.outputs_dir,
    )
    print(f"\nSaved  →  {npy_path}")
    print(f"Sidecar →  {json_path}")
