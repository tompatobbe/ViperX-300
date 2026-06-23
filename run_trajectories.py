#!/usr/bin/env python3
"""
VX300s excitation trajectory — Momani & Hosseinzadeh (2025) Section 3.2, Eq. 7.

Position (Eq. 7):
    q_j(t) = q0_j + Σ_{k=1}^{N_f} [a_{k,j}·sin(k·ω_l·t) + b_{k,j}·(cos(k·ω_l·t) - 1)] / (k·ω_l)

Velocity (analytic):
    q̇_j(t) = Σ_{k=1}^{N_f} a_{k,j}·cos(k·ω_l·t) − b_{k,j}·sin(k·ω_l·t)

Acceleration (analytic):
    q̈_j(t) = Σ_{k=1}^{N_f} k·ω_l·[−a_{k,j}·sin(k·ω_l·t) − b_{k,j}·cos(k·ω_l·t)]

Paper parameters: f_l = 0.1 Hz, N_f = 5, 900 s @ 200 Hz.

Coefficients are found via scipy SLSQP (analogue of the paper's fmincon active-set)
to minimise the condition number of the stacked velocity/acceleration data matrix,
subject to joint position, velocity, and acceleration limits.

Usage:
    python3 run_trajectories.py                  # optimise, then run
    python3 run_trajectories.py --no-optimize    # analytically scaled random coefficients
    python3 run_trajectories.py --duration 60    # short test run
"""

import argparse
import time

import numpy as np
from scipy.optimize import minimize

# Identification regressor (pure numpy; no ROS) — the excitation is now designed
# to condition the ACTUAL base regressor Φ_b (paper Eq. 11), not just a kinematic
# velocity/acceleration matrix. Imported lazily-safe: only `regressor_fast` and
# `find_base_parameters` are used, both ROS-free.
import sysid_feasible as sf

# ── Paper parameters (Section 3.2) ───────────────────────────────────────────
JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
N_JOINTS = len(JOINT_NAMES)

F_L     = 0.1                       # fundamental frequency [Hz]
OMEGA_L = 2.0 * np.pi * F_L         # ≈ 0.6283 rad/s
N_F     = 5                          # harmonics (paper: N_f = 5 for all joints)

# Maximum harmonic frequency: 5 · ω_l ≈ 3.14 rad/s ≈ 0.5 Hz,
# well below typical VX300s structural resonance.

# ── Joint limits (rad) ────────────────────────────────────────────────────────
# Shoulder/elbow ranges opened to the empirically-swept collision-safe envelope
# (gravity-comp float run data/float_envelope_20260623_113036.csv, 2026-06-23),
# replacing the old conservative box (shoulder ≤0.30, elbow ≤1.00). The box alone
# is NOT safe — shoulder & elbow are coupled (see SH_EL_BAND_*); the box is the
# per-joint backstop and the band constraint forbids the colliding corners.
# SHOULDER FLOOR −1.25: below shoulder ≈ −1.3 the shoulder link collides with the
# anti-tip CLAMP holding the base (observed 2026-06-23). −1.25 leaves 0.05 rad for
# execution overshoot. This supersedes the −1.78 the float sweep reached.
LIMITS_LO = np.array([-2.80, -1.25, -1.36, -1.50, -1.50, -2.80])
LIMITS_HI = np.array([ 2.80,  1.38,  1.58,  1.50,  1.50,  2.80])

# Coupled shoulder(idx1)–elbow(idx2) collision envelope, fitted from the float
# sweep: a diagonal band  elbow ≈ −0.7·shoulder + offset. Enforced on the sampled
# trajectory (slope, intercept), with SH_EL_MARGIN rad shrunk inward for overshoot.
SH_EL_BAND_HI = (-0.71,  0.30)   # elbow ≤ slope·shoulder + intercept
SH_EL_BAND_LO = (-0.67, -0.26)   # elbow ≥ slope·shoulder + intercept
SH_EL_MARGIN  = 0.08

# Operating-point (q0) offsets are free design variables, but kept within ±Q0_MAX
# of each joint's centre so the trajectory starts/sweeps symmetrically about the
# middle. Without this, dynamically-degenerate joints — above all the WAIST, whose
# angle does not affect the dynamics at all (base vertical-axis symmetry) — park at
# arbitrary extremes (e.g. waist q0 = −1.41), giving lopsided, unintuitive motion.
# The gravity-relevant joints' preferred offsets (shoulder ≈+0.2, elbow ≈−0.1) sit
# well inside this band, so conditioning is essentially unaffected.
Q0_MAX = 0.30

HOME_POS   = (LIMITS_LO + LIMITS_HI) / 2.0   # centred in operating range
HALF_RANGE = (LIMITS_HI - LIMITS_LO) / 2.0   # max swing from home

# Velocity and acceleration limits (rad/s, rad/s²). Reduced 2026-06-23 because the
# full-speed extended-arm motion dumped enough REACTION torque into the base to
# move the (clamped) platform — which both risks tipping and violates the
# fixed-base assumption the identification relies on. The WAIST is cut hardest: it
# whips the extended arm in yaw and was the largest base-reaction driver (accel
# amplitude 8.6, highest of all joints). The shoulder/elbow keep meaningful
# acceleration so inertia/Coriolis are still excited. Tune further if the base
# still moves.
VEL_MAX   = np.array([1.8, 2.2, 2.2, 2.5, 2.5, 2.5])   # rad/s
ACCEL_MAX = np.array([3.5, 5.0, 5.0, 6.0, 6.0, 6.0])   # rad/s²

_K_VALS = np.arange(1, N_F + 1, dtype=float)   # [1, 2, 3, 4, 5]


# ── Trajectory functions ──────────────────────────────────────────────────────

def traj_pos(t: np.ndarray, a: np.ndarray, b: np.ndarray,
             q0: np.ndarray = None) -> np.ndarray:
    """
    Joint positions from Eq. 7.
    t: scalar or (T,); a, b: (N_JOINTS, N_F); q0: per-joint offset (defaults HOME).
    Returns (N_JOINTS,) for scalar t, (T, N_JOINTS) for array t.
    """
    scalar = np.ndim(t) == 0
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if q0 is None:
        q0 = HOME_POS
    q = np.tile(q0, (len(t), 1))                        # (T, N_JOINTS)
    for k in range(1, N_F + 1):
        w = k * OMEGA_L
        s = np.sin(w * t)[:, None]                      # (T, 1)
        c = np.cos(w * t)[:, None] - 1.0
        q += (a[None, :, k - 1] * s + b[None, :, k - 1] * c) / w
    return q[0] if scalar else q


def traj_vel(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Analytic joint velocities q̇(t)."""
    scalar = np.ndim(t) == 0
    t = np.atleast_1d(np.asarray(t, dtype=float))
    qd = np.zeros((len(t), N_JOINTS))
    for k in range(1, N_F + 1):
        w = k * OMEGA_L
        qd += a[None, :, k - 1] * np.cos(w * t)[:, None] \
            - b[None, :, k - 1] * np.sin(w * t)[:, None]
    return qd[0] if scalar else qd


def traj_accel(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Analytic joint accelerations q̈(t)."""
    scalar = np.ndim(t) == 0
    t = np.atleast_1d(np.asarray(t, dtype=float))
    qdd = np.zeros((len(t), N_JOINTS))
    for k in range(1, N_F + 1):
        w = k * OMEGA_L
        qdd += w * (
            -a[None, :, k - 1] * np.sin(w * t)[:, None]
            -b[None, :, k - 1] * np.cos(w * t)[:, None]
        )
    return qdd[0] if scalar else qdd


# ── Analytic amplitude bounds (conservative, avoids time-grid sweep) ──────────

def _norms(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-harmonic amplitude: sqrt(a²+b²), shape (N_JOINTS, N_F)."""
    return np.sqrt(a ** 2 + b ** 2)


def pos_amplitude(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Upper bound on |q - HOME| per joint [rad]."""
    return np.sum(_norms(a, b) / (_K_VALS[None, :] * OMEGA_L), axis=1)


def vel_amplitude(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Upper bound on |q̇| per joint [rad/s]."""
    return np.sum(_norms(a, b), axis=1)


def accel_amplitude(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Upper bound on |q̈| per joint [rad/s²]."""
    return np.sum(_norms(a, b) * (_K_VALS[None, :] * OMEGA_L), axis=1)


# ── Optimiser (scipy SLSQP ≈ MATLAB fmincon active-set) ──────────────────────

# Evaluation grid for the design objective: one full fundamental period. Kept
# modest (the SLSQP finite-difference gradient evaluates the objective ~67× per
# step over the 66 design variables, and each eval rebuilds Φ_b with a Python
# loop over this grid) — one period at 5× the top harmonic is enough to capture
# the regressor's conditioning.
_T_EVAL = np.linspace(0, 1.0 / F_L, 80, endpoint=False)
_HALF   = N_JOINTS * N_F        # number of a (== number of b) coefficients


def _unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """x = [a (30), b (30), q0 (6)] → a, b each (N_JOINTS, N_F), q0 (N_JOINTS,)."""
    a  = x[:_HALF].reshape(N_JOINTS, N_F)
    b  = x[_HALF:2 * _HALF].reshape(N_JOINTS, N_F)
    q0 = x[2 * _HALF:2 * _HALF + N_JOINTS]
    return a, b, q0


# ── Batched regressor (vectorised over time) ─────────────────────────────────
# A time-batched re-implementation of sysid_feasible.regressor_fast: identical
# modified-DH Newton–Euler recursion, but with the T samples carried on a leading
# axis so the whole design grid is one set of numpy ops instead of a Python loop.
# ~20× faster, which is what makes the cond(Φ_b) optimiser tractable. Verified
# bit-for-bit against regressor_fast at import (see _assert_batch_matches).
_NP = sf.N_PARAMS
_NJ = sf.N_JOINTS
_PT = sf.N_PARAMS_T


def _batch_dh(alpha, a, d, theta):
    T = theta.shape[0]
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    M = np.zeros((T, 4, 4))
    M[:, 0, 0] = ct;    M[:, 0, 1] = -st;    M[:, 0, 3] = a
    M[:, 1, 0] = st*ca; M[:, 1, 1] = ct*ca;  M[:, 1, 2] = -sa; M[:, 1, 3] = -sa*d
    M[:, 2, 0] = st*sa; M[:, 2, 1] = ct*sa;  M[:, 2, 2] = ca;  M[:, 2, 3] = ca*d
    M[:, 3, 3] = 1.0
    return M


def _batch_regressor(Q, DQ, DDQ):
    """Stacked regressor (T*6, 78) for arrays Q, DQ, DDQ each (T, 6)."""
    T = Q.shape[0]
    EZ = np.array([0.0, 0.0, 1.0])
    # Forward kinematics
    mats = [np.tile(np.eye(4), (T, 1, 1))]
    for i in range(_NJ):
        alpha, a, d, off = sf.DH_PARAMS[i]
        mats.append(np.einsum('tij,tjk->tik', mats[-1], _batch_dh(alpha, a, d, Q[:, i] + off)))
    R   = [m[:, :3, :3] for m in mats]
    pos = [m[:, :3, 3]  for m in mats]

    # NE forward pass
    omega  = [np.zeros((T, 3)) for _ in range(_NJ + 1)]
    domega = [np.zeros((T, 3)) for _ in range(_NJ + 1)]
    ddp    = [np.zeros((T, 3)) for _ in range(_NJ + 1)]
    ddp[0][:] = np.array([0.0, 0.0, 9.81])
    for i in range(1, _NJ + 1):
        Rrel   = np.einsum('tji,tjk->tik', R[i-1], R[i])          # R[i-1].T @ R[i]
        R_to_i = np.transpose(Rrel, (0, 2, 1))
        w, dw, aa = omega[i-1], domega[i-1], ddp[i-1]
        P  = np.einsum('tji,tj->ti', R[i-1], pos[i] - pos[i-1])   # R[i-1].T @ Δp
        dqi  = DQ[:, i-1][:, None]; ddqi = DDQ[:, i-1][:, None]
        Rw = np.einsum('tij,tj->ti', R_to_i, w)
        omega[i]  = Rw + dqi * EZ
        domega[i] = np.einsum('tij,tj->ti', R_to_i, dw) + np.cross(Rw, dqi * EZ) + ddqi * EZ
        lin = np.cross(dw, P) + np.cross(w, np.cross(w, P)) + aa
        ddp[i] = np.einsum('tij,tj->ti', R_to_i, lin)

    # Backward pass building the columns
    W = np.zeros((T, _NJ, _PT))
    f_mat = np.zeros((T, 3, _PT)); n_mat = np.zeros((T, 3, _PT))
    for i in range(_NJ, 0, -1):
        li = i - 1; base = li * _NP
        w, dw, d = omega[i], domega[i], ddp[i]
        fi = np.zeros((T, 3, _NP)); ni = np.zeros((T, 3, _NP))
        fi[:, :, 0] = d
        for k, e in enumerate((np.array([1.,0,0]), np.array([0,1.,0]), np.array([0,0,1.]))):
            fi[:, :, 1+k] = np.cross(dw, e) + np.cross(w, np.cross(w, e))
            ni[:, :, 1+k] = np.cross(e, d)
        w0, w1, w2 = w[:, 0], w[:, 1], w[:, 2]
        d0, d1, d2 = dw[:, 0], dw[:, 1], dw[:, 2]
        Z = np.zeros(T)
        for col, Jdw, Jw in (
            (4, (d0, Z, Z),  (w0, Z, Z)),
            (5, (d1, d0, Z), (w1, w0, Z)),
            (6, (d2, Z, d0), (w2, Z, w0)),
            (7, (Z, d1, Z),  (Z, w1, Z)),
            (8, (Z, d2, d1), (Z, w2, w1)),
            (9, (Z, Z, d2),  (Z, Z, w2)),
        ):
            Jdw = np.stack(Jdw, axis=-1); Jw = np.stack(Jw, axis=-1)
            ni[:, :, col] = Jdw + np.cross(w, Jw)
        fi_full = np.zeros((T, 3, _PT)); fi_full[:, :, base:base+_NP] = fi
        ni_full = np.zeros((T, 3, _PT)); ni_full[:, :, base:base+_NP] = ni
        if i < _NJ:
            Rrel_n = np.einsum('tji,tjk->tik', R[i], R[i+1])      # R[i].T @ R[i+1]
            r_n    = np.einsum('tji,tj->ti', R[i], pos[i+1] - pos[i])
            f_from = np.einsum('tij,tjp->tip', Rrel_n, f_mat)
            cr = np.cross(r_n[:, None, :], np.transpose(f_from, (0, 2, 1)))
            n_from = np.einsum('tij,tjp->tip', Rrel_n, n_mat) + np.transpose(cr, (0, 2, 1))
        else:
            f_from = np.zeros((T, 3, _PT)); n_from = np.zeros((T, 3, _PT))
        f_mat = fi_full + f_from
        n_mat = ni_full + n_from
        W[:, li, :] = n_mat[:, 2, :]
        W[:, li, base+10] += DQ[:, li]
        W[:, li, base+11] += np.sign(DQ[:, li])
        W[:, li, base+12] += 1.0
    return W.reshape(T * _NJ, _PT)


def _assert_batch_matches():
    """Verify the batched regressor equals regressor_fast (correctness gate)."""
    rng = np.random.default_rng(0)
    Q, DQ, DDQ = rng.standard_normal((3, 5, 6))
    Wb = _batch_regressor(Q, DQ, DDQ).reshape(5, _NJ, _PT)
    for i in range(5):
        ref = sf.regressor_fast(Q[i], DQ[i], DDQ[i])
        err = np.max(np.abs(Wb[i] - ref))
        assert err < 1e-9, f'batched regressor mismatch {err:.2e} at sample {i}'


_assert_batch_matches()


def _base_regressor(a, b, q0, base_cols):
    """Stacked base identification regressor Φ_b over _T_EVAL (clipped to limits)."""
    q   = np.clip(traj_pos(_T_EVAL, a, b, q0), LIMITS_LO, LIMITS_HI)
    dq  = traj_vel(_T_EVAL, a, b)
    ddq = traj_accel(_T_EVAL, a, b)
    return _batch_regressor(q, dq, ddq)[:, base_cols]


def optimize_coefficients(seed: int, n_restarts: int = 4,
                          maxiter: int = 150) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find a, b, q0 minimising **cond(Φ_b)** — the condition number of the actual
    base identification regressor (paper Eq. 11), NOT a kinematic [vel;accel]
    matrix. Conditioning Φ_b is what makes the gravity / first-moment columns
    (e.g. shoulder vs elbow m·c_y) linearly separable, preventing the null-space
    lumping diagnosed on hardware (THESIS_NOTES 2026-06-23). q0 offsets are free
    design variables (as in the paper). Subject to joint position, velocity and
    acceleration limits. Multistart over n_restarts seeds for a robust optimum.
    Returns a, b each (N_JOINTS, N_F) and q0 (N_JOINTS,).
    """
    # Fix the identifiable base-column set ONCE from a richly-excited reference
    # trajectory, so the objective's regressor has a constant column basis across
    # iterations (the structural identifiable set is trajectory-independent).
    ref_a, ref_b = scale_to_limits(seed)
    W_ref = _batch_regressor(
        np.clip(traj_pos(_T_EVAL, ref_a, ref_b), LIMITS_LO, LIMITS_HI),
        traj_vel(_T_EVAL, ref_a, ref_b), traj_accel(_T_EVAL, ref_a, ref_b))
    base_cols, _ = sf.find_base_parameters(W_ref)
    print(f'[optimizer] base regressor: {len(base_cols)}/{sf.N_PARAMS_T} identifiable '
          f'columns; minimising cond(Φ_b) over {len(_T_EVAL)} samples …')

    def objective(x: np.ndarray) -> float:
        a, b, q0 = _unpack(x)
        Wb = _base_regressor(a, b, q0, base_cols)
        return float(np.linalg.cond(Wb))

    # Limits as constraints. POSITION uses the ACTUAL sampled trajectory range
    # (min/max over _T_EVAL), not the sum-of-norms upper bound — the bound
    # overstates the swing (harmonics rarely co-peak), which over-restricts and
    # stalls SLSQP. Vectorised over joints per call (returns a 6-vector, so one
    # constraint dict per type covers all joints). VEL/ACCEL keep the analytic
    # amplitude bound: conservative is the safe side for those rate limits.
    # Inward buffer: enforce limits on the coarse design grid with margin so the
    # full-rate trajectory (which peaks slightly higher between grid points) still
    # respects the TRUE limits and clears the hardware safety gates.
    PB, VB = 0.04, 0.10   # rad, rad/s
    def pos_lo_margin(x):
        a, b, q0 = _unpack(x)
        return np.min(traj_pos(_T_EVAL, a, b, q0), axis=0) - (LIMITS_LO + PB)
    def pos_hi_margin(x):
        a, b, q0 = _unpack(x)
        return (LIMITS_HI - PB) - np.max(traj_pos(_T_EVAL, a, b, q0), axis=0)
    # Coupled shoulder–elbow collision band, enforced at every sampled instant.
    def band_hi_margin(x):
        a, b, q0 = _unpack(x)
        q = traj_pos(_T_EVAL, a, b, q0)
        sh, el = q[:, 1], q[:, 2]
        return (SH_EL_BAND_HI[0]*sh + SH_EL_BAND_HI[1] - SH_EL_MARGIN) - el  # ≥0 ⇒ below upper
    def band_lo_margin(x):
        a, b, q0 = _unpack(x)
        q = traj_pos(_T_EVAL, a, b, q0)
        sh, el = q[:, 1], q[:, 2]
        return el - (SH_EL_BAND_LO[0]*sh + SH_EL_BAND_LO[1] + SH_EL_MARGIN)    # ≥0 ⇒ above lower
    # REST-TO-REST boundary conditions: the trajectory must start AND end at zero
    # velocity (and acceleration) so the arm eases out of / back into standstill
    # instead of lurching to full speed the instant execution begins. Because the
    # motion is periodic, these hold at every period boundary too:
    #   q̇(0) = Σ_k a_k  = 0   (zero boundary velocity)
    #   q̈(0) = −Σ_k k·b_k = 0 (zero boundary acceleration)
    # Equality constraints, per joint.
    constraints = [
        {'type': 'eq',   'fun': lambda x: _unpack(x)[0].sum(axis=1)},
        {'type': 'eq',   'fun': lambda x: (_K_VALS[None, :] * _unpack(x)[1]).sum(axis=1)},
        {'type': 'ineq', 'fun': pos_lo_margin},
        {'type': 'ineq', 'fun': pos_hi_margin},
        {'type': 'ineq', 'fun': band_hi_margin},
        {'type': 'ineq', 'fun': band_lo_margin},
        {'type': 'ineq', 'fun': lambda x: (VEL_MAX - VB) - vel_amplitude(_unpack(x)[0], _unpack(x)[1])},
        {'type': 'ineq', 'fun': lambda x: ACCEL_MAX     - accel_amplitude(_unpack(x)[0], _unpack(x)[1])},
        {'type': 'ineq', 'fun': lambda x: Q0_MAX - np.abs(_unpack(x)[2] - HOME_POS)},
    ]

    best = None          # best strictly-feasible result
    best_any = None      # best by objective regardless of feasibility (fallback)
    for r in range(n_restarts):
        # Start each restart from an already-feasible trajectory: scale_to_limits
        # sizes the amplitudes to satisfy ALL (position/velocity/acceleration)
        # limits at once, and q0 = HOME is inside the box and the shoulder–elbow
        # band. So SLSQP begins feasible and only has to *maintain* feasibility
        # while lowering cond(Φ_b) — far more reliable than starting from infeasible
        # random points (which previously diverged and left no feasible optimum).
        init_a, init_b = scale_to_limits(seed + 1000 * r)
        # Project the init onto the rest-to-rest equalities so SLSQP starts on
        # them: Σa=0 (zero-mean per joint), Σk·b=0 (remove the k-weighted mean).
        init_a = init_a - init_a.mean(axis=1, keepdims=True)
        init_b = init_b - ((_K_VALS[None, :] * init_b).sum(axis=1)
                           / np.sum(_K_VALS**2))[:, None] * _K_VALS[None, :]
        q0_0 = HOME_POS.copy()
        x0   = np.concatenate([init_a.ravel(), init_b.ravel(), q0_0])

        result = minimize(objective, x0, method='SLSQP', constraints=constraints,
                          options={'maxiter': maxiter, 'ftol': 1e-4, 'disp': False})
        # Feasibility verdict against the TRUE limits (the optimiser's constraints
        # are buffered inward; a design just inside the buffer is genuinely safe).
        # Matches the hardware gates in main().
        a_, b_, q0_ = _unpack(result.x)
        q_ = traj_pos(_T_EVAL, a_, b_, q0_)
        sh_, el_ = q_[:, 1], q_[:, 2]
        viol = max(
            float((LIMITS_LO - q_.min(0)).max()), float((q_.max(0) - LIMITS_HI).max()),
            float((vel_amplitude(a_, b_) - VEL_MAX).max()),
            float((accel_amplitude(a_, b_) - ACCEL_MAX).max()),
            float(np.max(el_ - (SH_EL_BAND_HI[0]*sh_ + SH_EL_BAND_HI[1]))),
            float(np.max((SH_EL_BAND_LO[0]*sh_ + SH_EL_BAND_LO[1]) - el_)),
            float(np.abs(a_.sum(axis=1)).max()),                       # zero start velocity
            float(np.abs((_K_VALS[None, :] * b_).sum(axis=1)).max()),  # zero start accel
            float((np.abs(q0_ - HOME_POS) - Q0_MAX).max()),            # q0 near centre
        )
        feasible = viol <= 1e-3
        tag = 'ok' if result.success else f'no-conv ({result.message.split(",")[0]})'
        print(f'[optimizer]   restart {r}: cond(Φ_b)={result.fun:8.1f}  '
              f'{tag}  feasible={feasible}'
              f'{"" if feasible else f" (max viol {viol:.3f})"}')
        if feasible and (best is None or result.fun < best.fun):
            best = result
        if best_any is None or result.fun < best_any.fun:
            best_any = result

    if best is not None:
        print(f'[optimizer] best feasible cond(Φ_b) = {best.fun:.1f}')
        return _unpack(best.x)
    # No strictly-feasible optimum found (usually too few iterations). Fall back
    # to the lowest-cond iterate with a warning — positions are clipped to limits
    # before execution; increase --maxiter/--restarts for a clean feasible design.
    print(f'[optimizer] WARNING: no strictly-feasible optimum (best cond(Φ_b)='
          f'{best_any.fun:.1f}); returning it — RAISE --maxiter/--restarts and '
          f'CHECK the printed pos/vel/acc vs limits before running on hardware.')
    return _unpack(best_any.x)


def scale_to_limits(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Fast fallback: analytically scale random coefficients to satisfy all limits."""
    rng   = np.random.default_rng(seed)
    raw_a = rng.standard_normal((N_JOINTS, N_F))
    raw_b = rng.standard_normal((N_JOINTS, N_F))
    nrms  = _norms(raw_a, raw_b) + 1e-12   # (N_JOINTS, N_F)

    pos_scale = HALF_RANGE / (np.sum(nrms / (_K_VALS[None, :] * OMEGA_L), axis=1) + 1e-12)
    vel_scale  = VEL_MAX   / (np.sum(nrms, axis=1) + 1e-12)
    acc_scale  = ACCEL_MAX / (np.sum(nrms * (_K_VALS[None, :] * OMEGA_L), axis=1) + 1e-12)

    scale = (0.80 * np.minimum(pos_scale, np.minimum(vel_scale, acc_scale)))[:, None]
    return raw_a * scale, raw_b * scale


# ── Diagnostics ───────────────────────────────────────────────────────────────

def print_stats(a: np.ndarray, b: np.ndarray, q0: np.ndarray = None) -> None:
    if q0 is None:
        q0 = HOME_POS
    # Use the ACTUAL sampled trajectory range (matches the optimiser's position
    # constraint), not the conservative sum-of-norms bound, so a '✓ within limits'
    # readout reflects what the arm will really do. '!' flags a real overshoot.
    t = np.linspace(0, 1.0 / F_L, 400, endpoint=False)
    q   = traj_pos(t, a, b, q0)
    vel = vel_amplitude(a, b)
    acc = accel_amplitude(a, b)
    print('[run] Trajectory statistics:')
    print(f'      f_l={F_L} Hz  ω_l={OMEGA_L:.4f} rad/s  N_f={N_F}')
    print(f'      Freq range: {OMEGA_L:.3f} – {N_F * OMEGA_L:.3f} rad/s')
    hdr = (f'      {"Joint":<14} {"q0":>7} {"pos range":>16} {"limit":>15} '
           f'{"Vel/lim":>13} {"Acc/lim":>13}')
    print(hdr)
    for j, name in enumerate(JOINT_NAMES):
        lo, hi = q[:, j].min(), q[:, j].max()
        ok = '✓' if lo >= LIMITS_LO[j] - 1e-3 and hi <= LIMITS_HI[j] + 1e-3 else '!'
        print(f'      {name:<14} {q0[j]:>+7.2f} [{lo:>+6.2f},{hi:>+6.2f}] '
              f'[{LIMITS_LO[j]:+.2f},{LIMITS_HI[j]:+.2f}]{ok}'
              f'   {vel[j]:>5.2f}/{VEL_MAX[j]:.1f}'
              f'   {acc[j]:>5.2f}/{ACCEL_MAX[j]:.1f}')
    # Rest-to-rest start: velocity at t=0 must be ≈0 (else the arm lurches from
    # standstill the instant execution begins).
    v0 = traj_vel(0.0, a, b)
    print(f'      start velocity q̇(0): max |{np.abs(v0).max():.3f}| rad/s '
          f'{"✓ rest" if np.abs(v0).max() < 0.05 else "! LURCHES from rest"}')
    # Coupled shoulder–elbow band compliance over the sampled trajectory.
    sh, el = q[:, 1], q[:, 2]
    hi_slack = np.min((SH_EL_BAND_HI[0]*sh + SH_EL_BAND_HI[1]) - el)
    lo_slack = np.min(el - (SH_EL_BAND_LO[0]*sh + SH_EL_BAND_LO[1]))
    band_ok = hi_slack >= -1e-3 and lo_slack >= -1e-3
    print(f'      shoulder–elbow band: min slack {min(hi_slack, lo_slack):+.3f} rad '
          f'{"✓ inside" if band_ok else "! VIOLATES collision band"}')


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='VX300s excitation trajectory (Eq. 7, Section 3.2)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--duration',    type=float, default=900.0,
                        help='Trajectory duration [s]; paper uses 900 s')
    parser.add_argument('--rate',        type=float, default=200.0,
                        help='Command rate [Hz]; paper uses 200 Hz')
    parser.add_argument('--robot-model', default='vx300s')
    parser.add_argument('--seed',        type=int,   default=42,
                        help='RNG seed for coefficient initialisation')
    parser.add_argument('--no-optimize', action='store_true',
                        help='Skip SLSQP; use analytically scaled random coefficients')
    parser.add_argument('--restarts',    type=int,   default=4,
                        help='Multistart count for the cond(Φ_b) optimiser')
    parser.add_argument('--maxiter',     type=int,   default=150,
                        help='SLSQP iterations per restart')
    parser.add_argument('--design-only', action='store_true',
                        help='Optimise + print the trajectory and exit (no hardware/ROS)')
    parser.add_argument('--save', default=None,
                        help='Save the optimised coefficients (a,b,q0) to this .npz')
    parser.add_argument('--load', default=None,
                        help='Load coefficients from a .npz (skips the optimiser; '
                             'use the vetted design for deterministic collection)')
    parser.add_argument('--stride',      type=int,   default=4,
                        help='Send every N-th waypoint; effective command rate = rate/stride')
    args = parser.parse_args()

    # ── Coefficients
    if args.load:
        print(f'[run] Loading design coefficients from {args.load} …')
        d = np.load(args.load)
        a, b, q0 = d['a'], d['b'], d['q0']
    elif args.no_optimize:
        print('[run] Using analytically scaled random coefficients …')
        a, b = scale_to_limits(args.seed)
        q0 = HOME_POS.copy()
    else:
        a, b, q0 = optimize_coefficients(args.seed, n_restarts=args.restarts,
                                         maxiter=args.maxiter)

    if args.save:
        np.savez(args.save, a=a, b=b, q0=q0, seed=args.seed,
                 f_l=F_L, n_f=N_F, limits_lo=LIMITS_LO, limits_hi=LIMITS_HI)
        print(f'[run] Saved design coefficients → {args.save}')

    print_stats(a, b, q0)

    # ── Pre-compute waypoints (position only; vel/accel are analytic)
    dt    = 1.0 / args.rate
    n     = int(args.duration * args.rate)
    t_vec = np.arange(n, dtype=float) * dt

    print(f'[run] Pre-computing {n} waypoints ({args.duration:.0f} s @ {args.rate:.0f} Hz) …')
    q_all = traj_pos(t_vec, a, b, q0)      # (n, N_JOINTS)
    # Box-limit gate BEFORE clipping: a stale/loaded design that exceeds the
    # current limits (e.g. a pre-shoulder-floor design) must NOT be silently
    # clipped into a distorted trajectory — flag it so it can be re-optimised.
    over = np.maximum(LIMITS_LO - q_all.min(axis=0), q_all.max(axis=0) - LIMITS_HI)
    _box_violation = float(over.max())
    np.clip(q_all, LIMITS_LO, LIMITS_HI, out=q_all)
    q_start = q_all[0]                       # trajectory start = q0 (sin0, cos0-1 = 0)

    # Report the achieved conditioning + the first-moment column separability the
    # new objective targets (shoulder vs elbow m·c_y), so a design can be judged
    # before any data collection.
    if not args.no_optimize:
        idx = np.arange(0, n, max(1, n // 400))
        Wc = _batch_regressor(q_all[idx], traj_vel(t_vec[idx], a, b),
                              traj_accel(t_vec[idx], a, b))
        base_cols, _ = sf.find_base_parameters(Wc)
        def _c(link):
            col = Wc[:, link * sf.N_PARAMS + 2]; return col - col.mean()
        sh, el = _c(1), _c(2)
        corr = float(sh @ el / (np.linalg.norm(sh) * np.linalg.norm(el) + 1e-12))
        print(f'[run] achieved cond(Φ_b)≈{np.linalg.cond(Wc[:, base_cols]):.1f}  '
              f'shoulder·elbow m·c_y corr={corr:+.3f}  (lower |corr| ⇒ separable)')

    if args.design_only:
        print('[run] --design-only: not connecting to hardware. Done.')
        return

    # SAFETY GATE: a design whose raw trajectory exceeds the box limits (e.g. a
    # design saved before a limit change, like the shoulder −1.25 clamp floor)
    # would be silently clipped into a distorted, possibly unsafe path. Abort.
    if _box_violation > 1e-3:
        print(f'[run] ABORT: design exceeds joint box limits by {_box_violation:.3f} rad '
              f'(stale design vs current limits, e.g. the shoulder −1.25 clamp floor). '
              f'Re-optimise: run_trajectories.py --design-only --save <npz>.')
        return

    # SAFETY GATE: refuse a design that does not start at rest — q̇(0) far from 0
    # makes the arm lurch to full speed the instant execution begins.
    v0max = float(np.abs(traj_vel(0.0, a, b)).max())
    if v0max > 0.05:
        print(f'[run] ABORT: trajectory starts at {v0max:.2f} rad/s (not at rest) — '
              f'it would lurch on engage. Re-optimise (rest-to-rest constraint), '
              f'or this is a stale design from before that fix.')
        return

    # SAFETY GATE: never drive hardware with a design that breaches the coupled
    # shoulder–elbow collision band (the per-joint clip above cannot enforce a
    # coupled constraint). Abort before connecting.
    sh, el = q_all[:, 1], q_all[:, 2]
    hi_slack = np.min((SH_EL_BAND_HI[0]*sh + SH_EL_BAND_HI[1]) - el)
    lo_slack = np.min(el - (SH_EL_BAND_LO[0]*sh + SH_EL_BAND_LO[1]))
    if min(hi_slack, lo_slack) < -1e-3:
        print(f'[run] ABORT: design violates the shoulder–elbow collision band '
              f'(min slack {min(hi_slack, lo_slack):+.3f} rad). Re-optimise with a '
              f'feasible result (raise --maxiter/--restarts) before collecting.')
        return

    # ── Connect (hardware import is lazy so trajectory DESIGN runs without ROS)
    print(f'[run] Connecting to "{args.robot_model}" …')
    from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
    bot = InterbotixManipulatorXS(
        robot_model=args.robot_model,
        group_name='arm',
        gripper_name='gripper',
    )

    print(f'[run] Moving to trajectory start {np.round(q_start, 3).tolist()} …')
    bot.arm.set_joint_positions(q_start.tolist(), moving_time=4.0, accel_time=0.5, blocking=True)
    time.sleep(2.0)

    print('[run] Closing gripper …')
    bot.gripper.grasp(delay=1.0)

    print('[run] Waiting 1 s — ensure recorder is running …')
    time.sleep(1.0)

    # ── Execute
    # moving_time is fixed to the command interval so the servo's internal
    # velocity limit scales with the actual step size, not near-zero distance.
    cmd_dt      = dt * args.stride
    moving_time = max(2.0 * cmd_dt, 0.08)    # matches run_sysid_pos.py pattern
    accel_time  = max(0.4 * cmd_dt, 0.02)
    print(f'[run] Effective command rate: {1.0/cmd_dt:.1f} Hz  '
          f'moving_time={moving_time:.3f}s  accel_time={accel_time:.3f}s')

    print('[run] Executing excitation trajectory — Ctrl+C to stop early …')
    t_start  = time.monotonic()
    n_sent   = 0
    n_stalls = 0

    try:
        for i in range(0, n, args.stride):
            if time.monotonic() - t_start >= args.duration:
                break

            q = q_all[i]

            # Pace to the scheduled wall-clock time for this waypoint
            t_sched   = t_start + t_vec[i]
            remaining = t_sched - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)
            elif -remaining >= 2.0 * cmd_dt:
                # ≥2 command intervals behind schedule — a comm stall
                # (70–280 ms publisher gaps observed over usbipd/WSL2,
                # 2026-06-12). Bursting the backlog at full speed commands
                # a violent multi-joint catch-up jerk; shift the schedule
                # instead so the trajectory resumes seamlessly from the
                # next waypoint, merely finishing late by the stall time.
                t_start  += -remaining
                n_stalls += 1
                print(f'[run] WARNING: {-remaining*1e3:.0f} ms stall — '
                      f'schedule shifted (total stalls: {n_stalls})')

            bot.arm.set_joint_positions(
                q.tolist(),
                moving_time=moving_time,
                accel_time=accel_time,
                blocking=False,
            )
            n_sent += 1

    except KeyboardInterrupt:
        print('\n[run] Interrupted.')

    elapsed = time.monotonic() - t_start
    print(f'[run] {n_sent} commands in {elapsed:.1f}s '
          f'(≈{n_sent / elapsed:.1f} Hz achieved, {n_stalls} stall(s) absorbed)')

    print('[run] Returning to sleep pose …')
    bot.gripper.release(delay=0.5)
    bot.arm.go_to_sleep_pose(moving_time=3.0, accel_time=0.5)
    print('[run] Done.')


if __name__ == '__main__':
    main()
