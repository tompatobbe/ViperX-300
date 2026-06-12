#!/usr/bin/env python3
"""
VX300s excitation trajectory following the finite Fourier series (FFS) structure
from Momani & Hosseinzadeh, Mechatronics 112 (2025) 103419, Section 3.2.

Trajectory structure (Eq. 7):
    q_j(t) = q0_j + Σ_{k=1}^{N_f} [ a_{j,k}/(k·ω_l) · sin(k·ω_l·t)
                                     + b_{j,k}/(k·ω_l) · (cos(k·ω_l·t) - 1) ]

Properties:
  - Starts exactly at HOME (displacement = 0 at t = 0)
  - Velocity bounded: |q_dot_j| ≤ Σ_k sqrt(a² + b²) = VEL_AMPLITUDES[j]
  - Position amplitude decreases as 1/(k·ω_l) — high-frequency components
    contribute small position swings, eliminating the jerkiness of equal-amplitude designs
  - Paper: ω_l = 2π·0.1 ≈ 0.628 rad/s, ω_max = 5 rad/s → N_f = 7 harmonics

Run AFTER record_joint_states.py is running.

Usage:
python3 run_sysid_pos_paper.py --duration 180 --rate 50
"""
import argparse
import time

import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


# ── Joint ordering ────────────────────────────────────────────────────────────
JOINT_NAMES = [
    'waist', 'shoulder', 'elbow',
    'forearm_roll', 'wrist_angle', 'wrist_rotate',
]

# Safety limits (rad) — tightened to avoid self-collision
LIMITS_LO = np.array([-2.80, -1.5, -0.2, -1.50, -1.50, -2.80])
LIMITS_HI = np.array([ 2.80,  0.30,  1.0,  1.50,  1.50,  2.80])

# Derived from limits so they stay in sync when limits change
HOME_POS       = (LIMITS_LO + LIMITS_HI) / 2.0
HALF_RANGE     = (LIMITS_HI - LIMITS_LO) / 2.0

# ── FFS parameters (paper Section 3.2) ───────────────────────────────────────
# Base angular frequency: ω_l = 2π · f_l, paper sets f_l = 0.1 Hz
OMEGA_L = 2.0 * np.pi * 0.1          # ≈ 0.6283 rad/s

# Number of harmonics: k · ω_l ≤ ω_max = 5 rad/s  →  N_f = 7
N_F = 7

# Velocity amplitudes: 0.80 × half_range × ω_l so the k=1 swing reaches
# ~80 % of each joint's half-range (derived, not hand-tuned)
VEL_AMPLITUDES = 0.95 * HALF_RANGE * OMEGA_L


def build_trajectory(
    duration: float,
    rate: float,
    rng: np.random.Generator,
) -> tuple[list[tuple[float, np.ndarray]], np.ndarray, np.ndarray]:
    """
    Build FFS trajectory following Eq. 7 of the paper.

    Returns (waypoints, a_coeffs, b_coeffs) where:
      waypoints : list of (t, q) pairs
      a_coeffs  : shape (n_joints, N_f) — cosine velocity coefficients
      b_coeffs  : shape (n_joints, N_f) — sine velocity coefficients
    """
    n_joints = len(JOINT_NAMES)

    # Draw random a,b and normalise each joint so its total velocity amplitude
    # equals VEL_AMPLITUDES[j]. This is the random (non-optimised) analogue of
    # the fmincon solution in the paper.
    raw_a = rng.standard_normal((n_joints, N_F))
    raw_b = rng.standard_normal((n_joints, N_F))

    a = np.zeros_like(raw_a)
    b = np.zeros_like(raw_b)
    for j in range(n_joints):
        total = np.sum(np.sqrt(raw_a[j] ** 2 + raw_b[j] ** 2))
        scale = VEL_AMPLITUDES[j] / total
        a[j]  = raw_a[j] * scale
        b[j]  = raw_b[j] * scale

    # Build waypoints
    dt    = 1.0 / rate
    n     = int(duration * rate)
    t_vec = np.arange(n) * dt

    waypoints: list[tuple[float, np.ndarray]] = []
    for t in t_vec:
        q = HOME_POS.copy()
        for k in range(1, N_F + 1):
            w = k * OMEGA_L
            q += (a[:, k - 1] / w) * np.sin(w * t) + (b[:, k - 1] / w) * (np.cos(w * t) - 1.0)
        q = np.clip(q, LIMITS_LO, LIMITS_HI)
        waypoints.append((t, q))

    return waypoints, a, b


def print_trajectory_stats(a: np.ndarray, b: np.ndarray) -> None:
    """Print per-joint amplitude and velocity bounds for verification."""
    print('[mover] Trajectory statistics:')
    print(f'        ω_l = {OMEGA_L:.4f} rad/s,  N_f = {N_F} harmonics')
    print(f'        Frequency range: {OMEGA_L:.3f} – {N_F * OMEGA_L:.3f} rad/s')
    print(f'        {"Joint":<14} {"Vel amp (rad/s)":>16} {"Max pos swing k=1 (rad)":>24}')
    for j, name in enumerate(JOINT_NAMES):
        vel_amp = float(np.sum(np.sqrt(a[j] ** 2 + b[j] ** 2)))
        pos_amp_k1 = vel_amp / OMEGA_L
        print(f'        {name:<14} {vel_amp:>16.3f} {pos_amp_k1:>24.3f}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='VX300s FFS excitation trajectory (paper-compliant)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--duration',      type=float, default=180.0,
                        help='Trajectory duration (s); paper uses 900 s')
    parser.add_argument('--rate',          type=float, default=50.0,
                        help='Command rate (Hz); paper uses 200 Hz')
    parser.add_argument('--robot-model',   default='vx300s',
                        help='Interbotix robot model string')
    parser.add_argument('--seed',          type=int,   default=42,
                        help='RNG seed — same seed reproduces identical trajectory')
    parser.add_argument('--stride',        type=int,   default=25,
                        help='Send every N-th waypoint; larger → bigger jumps')
    parser.add_argument('--move-speed',   type=float, default=1.5,
                        help='Peak joint speed for each blocking move (rad/s)')
    parser.add_argument('--accel-time',   type=float, default=0.30,
                        help='Acceleration phase duration (s) for each move')
    parser.add_argument('--settle-time',  type=float, default=0.0,
                        help='Extra pause after each move completes (s)')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f'[mover] Building {args.duration:.0f}s FFS trajectory @ {args.rate:.0f} Hz …')
    waypoints, a_coeffs, b_coeffs = build_trajectory(args.duration, args.rate, rng)

    print(f'[mover] {len(waypoints)} waypoints')
    print(f'[mover] stride={args.stride}  move_speed={args.move_speed} rad/s  '
          f'accel_time={args.accel_time}s  settle_time={args.settle_time}s')
    print_trajectory_stats(a_coeffs, b_coeffs)

    print(f'[mover] Connecting to "{args.robot_model}" …')
    bot = InterbotixManipulatorXS(
        robot_model=args.robot_model,
        group_name='arm',
        gripper_name='gripper',
    )

    print('[mover] Moving to HOME position …')
    bot.arm.set_joint_positions(HOME_POS.tolist(), moving_time=3.0, accel_time=0.5, blocking=True)
    time.sleep(2.5)

    print('[mover] Waiting 1 s — ensure recorder is running …')
    time.sleep(1.0)

    print('[mover] Executing FFS excitation trajectory …')
    t_start  = time.monotonic()
    n_sent   = 0
    q_prev   = HOME_POS.copy()

    for _, q in waypoints[::args.stride]:
        # Scale moving_time to the largest joint displacement so the robot
        # travels at roughly args.move_speed regardless of step size.
        max_dist    = float(np.max(np.abs(q - q_prev)))
        moving_time = max(max_dist / args.move_speed + 2.0 * args.accel_time, 0.20)

        bot.arm.set_joint_positions(
            q.tolist(),
            moving_time=moving_time,
            accel_time=args.accel_time,
            blocking=True,
        )
        if args.settle_time > 0.0:
            time.sleep(args.settle_time)

        q_prev  = q.copy()
        n_sent += 1

    elapsed = time.monotonic() - t_start
    print(f'[mover] Trajectory complete — {n_sent} moves in {elapsed:.1f}s')

    print('[mover] Returning to sleep pose …')
    bot.arm.go_to_sleep_pose(moving_time=3.0, accel_time=0.5)
    print('[mover] Done.')


if __name__ == '__main__':
    main()
