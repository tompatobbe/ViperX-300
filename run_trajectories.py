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
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

# ── Paper parameters (Section 3.2) ───────────────────────────────────────────
JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
N_JOINTS = len(JOINT_NAMES)

F_L     = 0.1                       # fundamental frequency [Hz]
OMEGA_L = 2.0 * np.pi * F_L         # ≈ 0.6283 rad/s
N_F     = 5                          # harmonics (paper: N_f = 5 for all joints)

# Maximum harmonic frequency: 5 · ω_l ≈ 3.14 rad/s ≈ 0.5 Hz,
# well below typical VX300s structural resonance.

# ── Joint limits (rad) ────────────────────────────────────────────────────────
LIMITS_LO = np.array([-2.80, -1.50, -0.20, -1.50, -1.50, -2.80])
LIMITS_HI = np.array([ 2.80,  0.30,  1.00,  1.50,  1.50,  2.80])

HOME_POS   = (LIMITS_LO + LIMITS_HI) / 2.0   # centred in operating range
HALF_RANGE = (LIMITS_HI - LIMITS_LO) / 2.0   # max swing from home

# Velocity and acceleration limits (Table 2 equivalent, conservative)
VEL_MAX   = np.array([3.14, 3.14, 3.14, 3.14, 3.14, 3.14])   # rad/s
ACCEL_MAX = np.array([10.0, 10.0, 10.0, 10.0, 10.0, 10.0])   # rad/s²

_K_VALS = np.arange(1, N_F + 1, dtype=float)   # [1, 2, 3, 4, 5]


# ── Trajectory functions ──────────────────────────────────────────────────────

def traj_pos(t: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Joint positions from Eq. 7.
    t: scalar or (T,); a, b: (N_JOINTS, N_F)
    Returns (N_JOINTS,) for scalar t, (T, N_JOINTS) for array t.
    """
    scalar = np.ndim(t) == 0
    t = np.atleast_1d(np.asarray(t, dtype=float))
    q = np.tile(HOME_POS, (len(t), 1))                  # (T, N_JOINTS)
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

def optimize_coefficients(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Find a, b that minimise cond(W) where W = [vel; accel] stacked over
    evaluation time points, subject to position/velocity/acceleration limits.
    Returns a, b each (N_JOINTS, N_F).
    """
    rng = np.random.default_rng(seed)

    # Evaluation grid: 2 full fundamental periods, 300 points
    t_eval = np.linspace(0, 2.0 / F_L, 300)

    def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        half = N_JOINTS * N_F
        return x[:half].reshape(N_JOINTS, N_F), x[half:].reshape(N_JOINTS, N_F)

    def objective(x: np.ndarray) -> float:
        a, b = unpack(x)
        vel = traj_vel(t_eval, a, b)    # (T, N_JOINTS)
        acc = traj_accel(t_eval, a, b)  # (T, N_JOINTS)
        W   = np.vstack([vel, acc])
        _, s, _ = np.linalg.svd(W, full_matrices=False)
        return float(s[0] / (s[-1] + 1e-10))   # minimise condition number

    constraints = []
    for j in range(N_JOINTS):
        constraints.append({'type': 'ineq',
                             'fun': lambda x, j=j: HALF_RANGE[j] - pos_amplitude(*unpack(x))[j]})
        constraints.append({'type': 'ineq',
                             'fun': lambda x, j=j: VEL_MAX[j]   - vel_amplitude(*unpack(x))[j]})
        constraints.append({'type': 'ineq',
                             'fun': lambda x, j=j: ACCEL_MAX[j] - accel_amplitude(*unpack(x))[j]})

    # Initialise: random coefficients scaled to 60 % of velocity limit
    raw_a = rng.standard_normal((N_JOINTS, N_F))
    raw_b = rng.standard_normal((N_JOINTS, N_F))
    nrms  = np.sum(np.sqrt(raw_a ** 2 + raw_b ** 2), axis=1) + 1e-12
    scale = (0.60 * VEL_MAX / nrms)[:, None]
    x0    = np.concatenate([(raw_a * scale).ravel(), (raw_b * scale).ravel()])

    print('[optimizer] Running SLSQP optimisation (≈ fmincon active-set) …')
    result = minimize(objective, x0, method='SLSQP',
                      constraints=constraints,
                      options={'maxiter': 500, 'ftol': 1e-6, 'disp': False})

    status = 'converged' if result.success else f'did not converge ({result.message})'
    print(f'[optimizer] {status} — condition number: {result.fun:.3f}')

    return unpack(result.x)


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

def print_stats(a: np.ndarray, b: np.ndarray) -> None:
    pos = pos_amplitude(a, b)
    vel = vel_amplitude(a, b)
    acc = accel_amplitude(a, b)
    print('[run] Trajectory statistics:')
    print(f'      f_l={F_L} Hz  ω_l={OMEGA_L:.4f} rad/s  N_f={N_F}')
    print(f'      Freq range: {OMEGA_L:.3f} – {N_F * OMEGA_L:.3f} rad/s')
    hdr = f'      {"Joint":<14} {"Pos/lim (rad)":>17} {"Vel/lim (rad/s)":>17} {"Acc/lim (rad/s²)":>18}'
    print(hdr)
    for j, name in enumerate(JOINT_NAMES):
        print(f'      {name:<14} {pos[j]:>7.3f}/{HALF_RANGE[j]:.2f}'
              f'   {vel[j]:>7.3f}/{VEL_MAX[j]:.2f}'
              f'   {acc[j]:>7.3f}/{ACCEL_MAX[j]:.2f}')


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
    parser.add_argument('--stride',      type=int,   default=4,
                        help='Send every N-th waypoint; effective command rate = rate/stride')
    args = parser.parse_args()

    # ── Coefficients
    if args.no_optimize:
        print('[run] Using analytically scaled random coefficients …')
        a, b = scale_to_limits(args.seed)
    else:
        a, b = optimize_coefficients(args.seed)

    print_stats(a, b)

    # ── Pre-compute waypoints (position only; vel/accel are analytic)
    dt    = 1.0 / args.rate
    n     = int(args.duration * args.rate)
    t_vec = np.arange(n, dtype=float) * dt

    print(f'[run] Pre-computing {n} waypoints ({args.duration:.0f} s @ {args.rate:.0f} Hz) …')
    q_all = traj_pos(t_vec, a, b)          # (n, N_JOINTS)
    np.clip(q_all, LIMITS_LO, LIMITS_HI, out=q_all)

    # ── Connect
    print(f'[run] Connecting to "{args.robot_model}" …')
    bot = InterbotixManipulatorXS(
        robot_model=args.robot_model,
        group_name='arm',
        gripper_name='gripper',
    )

    print('[run] Moving to HOME position …')
    bot.arm.set_joint_positions(HOME_POS.tolist(), moving_time=4.0, accel_time=0.5, blocking=True)
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
