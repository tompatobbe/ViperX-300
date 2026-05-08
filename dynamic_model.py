"""
dynamic_model.py — Full 6-DOF Lagrangian dynamic model for ViperX-300s
=======================================================================
Implements the complete equations of motion:

    M(q) q̈  +  C(q, q̇) q̇  +  g(q)  =  τ

where
  M(q)       — 6×6 configuration-dependent mass / inertia matrix
  C(q, q̇)   — 6×6 Coriolis + centrifugal matrix (Christoffel symbols)
  g(q)       — 6×1 gravity torque vector

Workflow
--------
1. Forward kinematics via DH parameters (from Visualizer_org.py).
2. Per-link CoM Jacobians  →  M(q).
3. Numerical Christoffel symbols  →  C(q, q̇).
4. Numerical gradient of potential energy  →  g(q).
5. Load CSV data, smooth with Savitzky-Golay, finite-difference to get q̇ / q̈.
6. Evaluate τ_pred = M q̈ + C q̇ + g for every sample.
7. Identify per-joint scale + friction via least squares.
8. Plot predicted vs measured.

Usage
-----
  python dynamic_model.py                    # self-test at zero config
  python dynamic_model.py arm_data.csv       # validate + identify from data
  python dynamic_model.py arm_data.csv --no-id   # skip identification

Dependencies
------------
  pip install numpy pandas scipy matplotlib

Notes
-----
- Effort (torque) in the CSV is the Dynamixel "Present Load" value in 0.1 %
  units of stall torque.  STALL_TORQUE below converts it to Nm.  Measure the
  actual stall current and supply voltage on your robot to refine these numbers.
- CoM positions and inertia tensors are taken directly from vx300s.urdf.
  The inertial <origin rpy="..."> rotation is applied to each tensor.
- The DH theta_offset encodes the difference between the URDF zero-angle
  definition and the DH frame; CoM vectors are expressed in the URDF link
  frame and may be slightly misaligned with the DH frame.  System
  identification corrects for residual errors in all inertial parameters.
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter

# ══════════════════════════════════════════════════════════════════════════════
# 1.  ROBOT GEOMETRY — DH Parameters
# ══════════════════════════════════════════════════════════════════════════════
# Source: Visualizer_org.py  (Standard / Craig DH convention)
# Transform per joint:  T_i = Rz(θᵢ + θ_off) · Tz(d) · Tx(a) · Rx(α)
#
# Columns:  [  α (°)    a (m)      d (m)    θ_offset (°)  ]

DH = np.array([
    [ 90.0,  0.00000,  0.12675,   0.00 ],   # 1  Waist
    [  0.0,  0.30594,  0.00000,  78.66 ],   # 2  Shoulder
    [ 90.0,  0.30000,  0.00000, -90.00 ],   # 3  Elbow
    [  0.0,  0.00000,  0.05000, -90.00 ],   # 4  Forearm Roll
    [ 90.0,  0.00000,  0.00000,  90.00 ],   # 5  Wrist Angle
    [-90.0,  0.00000,  0.05000, -90.00 ],   # 6  Wrist Rotate
])

N_DOF        = 6
G_ACC        = 9.81   # m s⁻²
JOINT_NAMES  = ["Waist", "Shoulder", "Elbow",
                "Forearm Roll", "Wrist Angle", "Wrist Rotate"]

# ══════════════════════════════════════════════════════════════════════════════
# 2.  INERTIAL PARAMETERS  (extracted from vx300s.urdf)
# ══════════════════════════════════════════════════════════════════════════════

def _rpy_to_R(r, p, y):
    """Intrinsic RPY → 3×3 rotation matrix."""
    Rx = np.array([[1,      0,       0     ],
                   [0,  np.cos(r), -np.sin(r)],
                   [0,  np.sin(r),  np.cos(r)]])
    Ry = np.array([[ np.cos(p), 0, np.sin(p)],
                   [     0,     1,     0    ],
                   [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0],
                   [np.sin(y),  np.cos(y), 0],
                   [    0,          0,     1]])
    return Rz @ Ry @ Rx

def _I(ixx, ixy, ixz, iyy, iyz, izz):
    return np.array([[ixx, ixy, ixz],
                     [ixy, iyy, iyz],
                     [ixz, iyz, izz]])

def _rot_I(I_urdf, rpy):
    """Rotate an inertia tensor from URDF frame to link frame: R I Rᵀ."""
    R = _rpy_to_R(*rpy)
    return R @ I_urdf @ R.T

_PI = np.pi

# Each dict: mass [kg], com in link frame [m], I at CoM in link frame [kg m²].
# The URDF inertial <origin rpy="..."> rotation is applied to each tensor.
LINKS = [
    {   # 1 — shoulder_link  (moved by waist joint)
        "mass": 0.969034,
        "com":  np.array([-5.348e-2, -5.626e-4,  2.060e-2]),
        "I":    _rot_I(_I(6.024e-3,  4.713e-5,  3.851e-6,
                          1.700e-3, -8.415e-5,  7.162e-3), (0, 0, _PI/2)),
    },
    {   # 2 — upper_arm_link
        "mass": 0.798614,
        "com":  np.array([ 2.592e-4, -3.355e-6,  1.161e-2]),
        "I":    _rot_I(_I(9.388e-4, -1.000e-9, -1.910e-8,
                          1.138e-3,  5.957e-6,  1.201e-3), (0, 0, _PI/2)),
    },
    {   # 3 — upper_forearm_link
        "mass": 0.792592,
        "com":  np.array([ 2.070e-2,  0.000e+0,  2.265e-1]),
        "I":    _rot_I(_I(8.925e-3,  0.000e+0,  0.000e+0,
                          8.937e-3,  1.201e-3,  9.357e-4), (0, 0, _PI/2)),
    },
    {   # 4 — lower_forearm_link  (forearm roll)
        "mass": 0.322228,
        "com":  np.array([ 1.057e-1,  0.000e+0,  0.000e+0]),
        "I":    _I(1.524e-4, -1.883e-5, -8.406e-6,
                   1.342e-3,  1.256e-6,  1.441e-3),
    },
    {   # 5 — wrist_link  (wrist angle)
        "mass": 0.414823,
        "com":  np.array([ 5.135e-2, -6.805e-3,  0.000e+0]),
        "I":    _I(1.753e-4, -8.528e-5,  0.000e+0,
                   5.269e-4,  0.000e+0,  5.911e-4),
    },
    {   # 6 — gripper_link  (wrist rotate)
        "mass": 0.115395,
        "com":  np.array([ 4.674e-2,  7.670e-6, -1.057e-2]),
        "I":    _rot_I(_I(4.631e-5,  1.950e-8,  2.300e-9,
                          4.514e-5,  4.200e-6,  5.270e-5), (0, _PI, _PI/2)),
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# 3.  KINEMATICS
# ══════════════════════════════════════════════════════════════════════════════

def _dh_T(alpha_deg, a, d, theta_deg):
    """Standard DH homogeneous transform: T = Rz(θ) Tz(d) Tx(a) Rx(α)."""
    θ = np.radians(theta_deg)
    α = np.radians(alpha_deg)
    cθ, sθ = np.cos(θ), np.sin(θ)
    cα, sα = np.cos(α), np.sin(α)
    return np.array([
        [ cθ, -sθ*cα,  sθ*sα, a*cθ],
        [ sθ,  cθ*cα, -cθ*sα, a*sθ],
        [  0,     sα,     cα,    d ],
        [  0,      0,      0,    1 ],
    ])

def forward_kinematics(q):
    """
    q: joint angles in radians (6,)
    Returns T_list (7,): T[0] = eye(4) (world), T[i] = T_{0→i} for i=1..6.
    """
    T = [np.eye(4)]
    for i in range(N_DOF):
        α, a, d, θ_off = DH[i]
        T.append(T[-1] @ _dh_T(α, a, d, np.degrees(q[i]) + θ_off))
    return T

def _com_world(q, T, links=None):
    """(6, 3) array of CoM positions in world frame."""
    if links is None:
        links = LINKS
    return np.array([
        T[i+1][:3, 3] + T[i+1][:3, :3] @ links[i]["com"]
        for i in range(N_DOF)
    ])

# ══════════════════════════════════════════════════════════════════════════════
# 4.  LINK CoM JACOBIANS
# ══════════════════════════════════════════════════════════════════════════════

def _link_jacobians(q, T, links=None):
    """
    Returns Jv[i] (3×6) and Jw[i] (3×6) for the CoM of link i.
    For revolute joints, only columns 0..i are non-zero:
      Jv[:,j] = z_j × (p_ci − o_j)   (linear)
      Jw[:,j] = z_j                    (angular)
    """
    pcs = _com_world(q, T, links)
    Jv = [np.zeros((3, N_DOF)) for _ in range(N_DOF)]
    Jw = [np.zeros((3, N_DOF)) for _ in range(N_DOF)]
    for i in range(N_DOF):
        for j in range(i + 1):
            z_j = T[j][:3, 2]
            o_j = T[j][:3, 3]
            Jv[i][:, j] = np.cross(z_j, pcs[i] - o_j)
            Jw[i][:, j] = z_j
    return Jv, Jw

# ══════════════════════════════════════════════════════════════════════════════
# 5.  MASS MATRIX  M(q)
# ══════════════════════════════════════════════════════════════════════════════

def mass_matrix(q, links=None):
    """
    M(q) = Σᵢ [ mᵢ Jvᵢᵀ Jvᵢ  +  Jwᵢᵀ (Rᵢ Iᵢ Rᵢᵀ) Jwᵢ ]
    Returns symmetric 6×6 matrix.
    """
    if links is None:
        links = LINKS
    T    = forward_kinematics(q)
    Jv, Jw = _link_jacobians(q, T, links)
    M = np.zeros((N_DOF, N_DOF))
    for i in range(N_DOF):
        R_i   = T[i+1][:3, :3]
        I_w   = R_i @ links[i]["I"] @ R_i.T   # inertia rotated to world frame
        M    += links[i]["mass"] * Jv[i].T @ Jv[i] + Jw[i].T @ I_w @ Jw[i]
    return M

# ══════════════════════════════════════════════════════════════════════════════
# 6.  CORIOLIS / CENTRIFUGAL MATRIX  C(q, q̇)
# ══════════════════════════════════════════════════════════════════════════════

def coriolis_matrix(q, dq, links=None, eps=1e-7):
    """
    Christoffel symbol formulation (numerical ∂M/∂qₖ):

      C[k,j] = Σᵢ ½ ( ∂M[k,j]/∂qᵢ + ∂M[k,i]/∂qⱼ − ∂M[i,j]/∂qₖ ) dqᵢ

    τ_coriolis = C(q, q̇) @ q̇
    """
    if links is None:
        links = LINKS
    M0   = mass_matrix(q, links)
    dMdq = np.zeros((N_DOF, N_DOF, N_DOF))   # dMdq[k] = ∂M/∂q_k
    for k in range(N_DOF):
        qp       = q.copy(); qp[k] += eps
        dMdq[k]  = (mass_matrix(qp, links) - M0) / eps

    C = np.zeros((N_DOF, N_DOF))
    for k in range(N_DOF):
        for j in range(N_DOF):
            for i in range(N_DOF):
                C[k, j] += 0.5 * (dMdq[i, k, j]
                                 + dMdq[j, k, i]
                                 - dMdq[k, i, j]) * dq[i]
    return C

# ══════════════════════════════════════════════════════════════════════════════
# 7.  GRAVITY VECTOR  g(q) = ∂P/∂q
# ══════════════════════════════════════════════════════════════════════════════

def _potential(q, links):
    T   = forward_kinematics(q)
    pcs = _com_world(q, T, links)
    return G_ACC * sum(links[i]["mass"] * pcs[i, 2] for i in range(N_DOF))

def gravity_vector(q, links=None, eps=1e-7):
    """
    g(q) = ∂P/∂q   (numerical gradient of gravitational potential energy).
    Only the z-coordinate of each CoM contributes: P = g Σ mᵢ zᵢ(q).
    """
    if links is None:
        links = LINKS
    P0 = _potential(q, links)
    g  = np.zeros(N_DOF)
    for j in range(N_DOF):
        qp    = q.copy(); qp[j] += eps
        g[j]  = (_potential(qp, links) - P0) / eps
    return g

# ══════════════════════════════════════════════════════════════════════════════
# 8.  INVERSE DYNAMICS
# ══════════════════════════════════════════════════════════════════════════════

def inverse_dynamics(q, dq, ddq, links=None):
    """
    Predicted joint torques:  τ = M(q) q̈ + C(q, q̇) q̇ + g(q)

    Args
    ----
    q, dq, ddq : (6,) arrays — joint position, velocity, acceleration [rad, rad/s, rad/s²]
    links      : optional list of link dicts (defaults to LINKS from URDF)

    Returns
    -------
    tau : (6,) predicted torques [Nm]
    """
    if links is None:
        links = LINKS
    M = mass_matrix(q, links)
    C = coriolis_matrix(q, dq, links)
    g = gravity_vector(q, links)
    return M @ ddq + C @ dq + g

# ══════════════════════════════════════════════════════════════════════════════
# 9.  TORQUE UNIT CONVERSION  (Dynamixel "Present Load" → Nm)
# ══════════════════════════════════════════════════════════════════════════════
# Dynamixel Protocol 2.0 Present_Load: signed integer in 0.1 % units of stall
# torque.  Range −1000 … +1000.  Multiply by (stall_torque / 1000) to get Nm.
#
# Stall torques at 24 V (Robotis spec sheets):
#   XM430-W350  →  4.1 Nm   (waist, forearm roll, wrist angle, wrist rotate)
#   XM540-W270  → 10.6 Nm   (shoulder ×2, elbow)
#
# Tune STALL_TORQUE if your supply voltage or motor variant differs.

STALL_TORQUE = np.array([4.1, 10.6, 10.6, 4.1, 4.1, 4.1])  # per joint [Nm]

def effort_to_Nm(effort_raw):
    """Convert (N, 6) Dynamixel load values to joint torques in Nm."""
    return effort_raw * (STALL_TORQUE / 1000.0)

# ══════════════════════════════════════════════════════════════════════════════
# 10.  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

_ARM_JOINTS = ["waist", "shoulder", "elbow",
               "forearm_roll", "wrist_angle", "wrist_rotate"]

def load_data(csv_path, sg_window=21, sg_poly=3):
    """
    Load arm_data CSV, differentiate positions twice with Savitzky-Golay
    smoothing to obtain q̇ and q̈.

    Savitzky-Golay preserves peak heights better than a plain low-pass filter,
    which is important for accurate acceleration estimation.

    Returns
    -------
    q    : (N, 6) joint angles [rad]
    dq   : (N, 6) joint velocities [rad/s]
    ddq  : (N, 6) joint accelerations [rad/s²]
    tau  : (N, 6) measured joint torques [Nm]
    t    : (N,)   timestamps [s]
    """
    df  = pd.read_csv(csv_path)
    t   = df["time"].values.astype(float)
    N   = len(t)

    # Savitzky-Golay window must be odd and ≤ N
    win = sg_window
    if win > N:
        win = N if N % 2 == 1 else N - 1
    if win < sg_poly + 2:
        raise ValueError(f"Not enough samples ({N}) for SG filter "
                         f"(poly={sg_poly}, window={win})")

    q_raw   = np.column_stack([df[f"{j}_pos"]    for j in _ARM_JOINTS])
    tau_raw = np.column_stack([df[f"{j}_effort"] for j in _ARM_JOINTS])

    # Smooth positions
    q = np.column_stack([savgol_filter(q_raw[:, i], win, sg_poly)
                         for i in range(N_DOF)])

    # First derivative → velocity
    dt      = np.diff(t)
    _pad    = lambda x: np.vstack([x[:1], x])   # repeat first row so shape stays (N,6)
    dq_raw  = _pad(np.diff(q, axis=0) / dt[:, None])
    dq      = np.column_stack([savgol_filter(dq_raw[:, i], win, sg_poly)
                                for i in range(N_DOF)])

    # Second derivative → acceleration
    ddq_raw = _pad(np.diff(dq, axis=0) / dt[:, None])
    ddq     = np.column_stack([savgol_filter(ddq_raw[:, i], win, sg_poly)
                                for i in range(N_DOF)])

    # Smooth torques and convert units
    tau_smooth = np.column_stack([savgol_filter(tau_raw[:, i], win, sg_poly)
                                  for i in range(N_DOF)])
    tau = effort_to_Nm(tau_smooth)

    return q, dq, ddq, tau, t

# ══════════════════════════════════════════════════════════════════════════════
# 11.  SYSTEM IDENTIFICATION
# ══════════════════════════════════════════════════════════════════════════════
# Per-joint affine model:
#
#   τ_meas[j] ≈  scale[j] * τ_model[j]
#              + fc[j] * sign(q̇[j])       ← Coulomb friction
#              + fv[j] * q̇[j]             ← viscous friction
#
# This is linear in [scale, fc, fv] → ordinary least squares per joint.
# A scale ≈ 1 means the URDF inertial parameters are accurate.
# A scale ≠ 1 suggests the Dynamixel load-to-Nm conversion needs adjustment
# or the inertial parameters need refinement.

def identify_friction_and_scale(tau_model, tau_meas, dq):
    """
    Identifies per-joint: torque scale, Coulomb friction coefficient, viscous
    friction coefficient via least squares.

    Args
    ----
    tau_model : (N, 6)  torques predicted by the dynamic model  [Nm]
    tau_meas  : (N, 6)  measured torques (after unit conversion) [Nm]
    dq        : (N, 6)  joint velocities [rad/s]

    Returns
    -------
    scale : (6,) multiplicative scale on model torque
    fc    : (6,) Coulomb friction [Nm]
    fv    : (6,) viscous friction [Nm·s/rad]
    tau_id: (N, 6) torques from the identified model
    """
    scale = np.zeros(N_DOF)
    fc    = np.zeros(N_DOF)
    fv    = np.zeros(N_DOF)
    tau_id = np.zeros_like(tau_meas)

    for j in range(N_DOF):
        # Regressor columns: [τ_model, sign(dq), dq]
        W = np.column_stack([
            tau_model[:, j],
            np.sign(dq[:, j]),
            dq[:, j],
        ])
        # Least squares: τ_meas[:,j] ≈ W @ [scale_j, fc_j, fv_j]
        phi, _, _, _ = np.linalg.lstsq(W, tau_meas[:, j], rcond=None)
        scale[j], fc[j], fv[j] = phi
        tau_id[:, j] = W @ phi

    return scale, fc, fv, tau_id

# ══════════════════════════════════════════════════════════════════════════════
# 12.  VALIDATION  &  PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

def _rmse(a, b):
    return np.sqrt(np.mean((a - b) ** 2))

def validate(csv_path, run_identification=True):
    """
    Full pipeline: load data → compute model torques → optionally identify
    friction/scale → plot comparison.

    Returns
    -------
    results : dict with q, dq, ddq, tau_meas, tau_model, tau_id, t, scale, fc, fv
    """
    print(f"\nLoading data from {csv_path} …")
    q, dq, ddq, tau_meas, t = load_data(csv_path)
    N = len(t)

    print(f"  {N} samples  |  duration {t[-1]-t[0]:.1f} s  "
          f"|  mean dt {np.mean(np.diff(t))*1e3:.1f} ms")

    print("Computing inverse dynamics …")
    tau_model = np.zeros((N, N_DOF))
    for k in range(N):
        if k % max(1, N // 10) == 0:
            print(f"  {k:5d}/{N}")
        tau_model[k] = inverse_dynamics(q[k], dq[k], ddq[k])

    scale = fc = fv = tau_id = None
    if run_identification:
        print("\nRunning system identification …")
        scale, fc, fv, tau_id = identify_friction_and_scale(
            tau_model, tau_meas, dq)
        print("\nIdentified parameters:")
        header = f"  {'Joint':<14} {'scale':>8} {'fc [Nm]':>10} {'fv [Nm·s]':>12}"
        print(header)
        for j in range(N_DOF):
            print(f"  {JOINT_NAMES[j]:<14} {scale[j]:>8.4f} {fc[j]:>10.4f} {fv[j]:>12.4f}")

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(N_DOF, 1, figsize=(14, 3 * N_DOF), sharex=True)
    fig.suptitle("ViperX-300s  |  Dynamic Model Validation", fontsize=13, y=1.002)

    for j, ax in enumerate(axes):
        ax.plot(t, tau_meas[:, j],  lw=1.0, alpha=0.7, label="Measured")
        ax.plot(t, tau_model[:, j], lw=1.0, alpha=0.7, label="Model (URDF params)",
                linestyle="--")
        if tau_id is not None:
            ax.plot(t, tau_id[:, j], lw=1.2, alpha=0.9, label="Model + identified",
                    linestyle=":")
        rmse_m = _rmse(tau_meas[:, j], tau_model[:, j])
        label  = f"{JOINT_NAMES[j]}  (RMSE model {rmse_m:.3f} Nm"
        if tau_id is not None:
            rmse_id = _rmse(tau_meas[:, j], tau_id[:, j])
            label  += f",  identified {rmse_id:.3f} Nm"
        label += ")"
        ax.set_title(label, fontsize=9)
        ax.set_ylabel("τ [Nm]")
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.25)

    axes[-1].set_xlabel("Time [s]")
    plt.tight_layout()
    out = csv_path.rsplit(".", 1)[0] + "_validation.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to {out}")
    plt.show()

    return dict(q=q, dq=dq, ddq=ddq, tau_meas=tau_meas,
                tau_model=tau_model, tau_id=tau_id, t=t,
                scale=scale, fc=fc, fv=fv)

# ══════════════════════════════════════════════════════════════════════════════
# 13.  SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def self_test():
    """Evaluate M, g and a gravity-only inverse dynamics at a sample pose."""
    print("=" * 60)
    print("ViperX-300s  Dynamic Model  —  self-test")
    print("=" * 60)

    configs = {
        "zero":          np.zeros(6),
        "shoulder 90°":  np.array([0, np.pi/2, 0, 0, 0, 0]),
        "full extend":   np.array([0, 0, 0, 0, 0, 0]),
    }

    for name, q in configs.items():
        print(f"\n── Config: {name} ──")
        M = mass_matrix(q)
        g = gravity_vector(q)
        K = 0.5 * np.dot(g, g)   # just a sanity number
        print(f"  M(q) diagonal : {np.diag(M).round(4)}")
        print(f"  g(q)          : {g.round(4)}")
        print(f"  g[0] (waist)  : {g[0]:.6f}  (should be ≈ 0, waist axis is vertical)")

    # Gravity-only torques at a tilted shoulder
    q_tilt = np.array([0, 0.5, 0.3, 0, 0, 0])
    tau_grav = inverse_dynamics(q_tilt, np.zeros(6), np.zeros(6))
    print(f"\nGravity-only torques at q=[0, 0.5, 0.3, 0, 0, 0] rad:")
    for j in range(N_DOF):
        print(f"  {JOINT_NAMES[j]:<14}: {tau_grav[j]:+.4f} Nm")
    print("\nSelf-test complete.")

# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ViperX-300s 6-DOF dynamic model and system identification")
    parser.add_argument("csv", nargs="?", default=None,
                        help="Path to arm_data CSV (omit for self-test)")
    parser.add_argument("--no-id", action="store_true",
                        help="Skip system identification, only plot model vs measured")
    args = parser.parse_args()

    if args.csv is None:
        self_test()
    else:
        validate(args.csv, run_identification=not args.no_id)
