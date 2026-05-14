#!/usr/bin/env python3
"""
VX300s sum-of-sinusoids excitation trajectory (MOVER).

Sends dynamically rich joint commands for system identification.
Run in Terminal 2 AFTER starting record_joint_states.py in Terminal 1.

Usage:
    python3 run_sysid_trajectory.py --duration 60 --rate 50

No background spinning — all SDK calls stay in the main thread, so
rclpy.spin_until_future_complete() inside the SDK never conflicts.
"""
import argparse
import time

import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


# ── Joint ordering (matches interbotix arm group) ─────────────────────────────
JOINT_NAMES = [
    'waist', 'shoulder', 'elbow',
    'forearm_roll', 'wrist_angle', 'wrist_rotate',
]

# Safe mid-range operating position
HOME_POS = [0.0, -0.5, 0.5, 0.0, 0.3, 0.0]

# Conservative safety limits (rad) — slightly inside hardware limits
LIMITS_LO = np.array([-2.80, -1.70, -1.90, -2.80, -1.60, -2.80])
LIMITS_HI = np.array([ 2.80,  1.80,  1.40,  2.80,  2.00,  2.80])

# Per-joint excitation amplitude (rad) centred around HOME_POS
AMPLITUDES = np.array([0.60, 0.40, 0.45, 0.60, 0.50, 0.50])

# Sinusoid component angular frequencies (rad/s) — logarithmically spaced
# Covers 0.3 – 3.8 rad/s ≈ 0.05 – 0.6 Hz for typical manipulator dynamics
FREQS = np.array([0.30, 0.60, 1.00, 1.60, 2.50, 3.80])

# ── VX300s approximate geometry for end-effector z estimate ──────────────────
_Z_SHOULDER = 0.127   # base plate → shoulder pivot (m)
_L_UPPER    = 0.200   # shoulder → elbow (m)
_L_FORE     = 0.265   # elbow → wrist centre, incl. forearm offset (m)
_L_HAND     = 0.170   # wrist centre → EE tip (m)


def ee_z_estimate(q: np.ndarray) -> float:
    """
    Conservative planar (2-D) estimate of the VX300s end-effector z-height.
    Ignores waist rotation; that only rotates the arm in the x-y plane, not z.
    """
    q2, q3, q5 = float(q[1]), float(q[2]), float(q[4])
    return (
        _Z_SHOULDER
        + _L_UPPER * np.sin(q2)
        + _L_FORE  * np.sin(q2 + q3)
        + _L_HAND  * np.sin(q2 + q3 + q5)
    )


def build_trajectory(
    duration: float,
    rate: float,
    rng: np.random.Generator,
) -> list[tuple[float, np.ndarray]]:
    """
    Return (timestamp, joint_positions) pairs for the full excitation trajectory.

    Each joint uses the same set of FREQS with independent random phases,
    offset around HOME_POS and clipped to safety limits.
    """
    dt     = 1.0 / rate
    n      = int(duration * rate)
    t_vec  = np.arange(n) * dt
    # Independent random phase for every (joint, frequency) pair
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(len(JOINT_NAMES), len(FREQS)))
    home   = np.asarray(HOME_POS, dtype=float)

    waypoints: list[tuple[float, np.ndarray]] = []
    for i, t in enumerate(t_vec):
        delta = np.zeros(len(JOINT_NAMES))
        for j in range(len(JOINT_NAMES)):
            for k, w in enumerate(FREQS):
                delta[j] += (AMPLITUDES[j] / len(FREQS)) * np.sin(w * t + phases[j, k])
        q = np.clip(home + delta, LIMITS_LO, LIMITS_HI)
        waypoints.append((t, q))
    return waypoints


def main() -> None:
    parser = argparse.ArgumentParser(
        description='VX300s sum-of-sinusoids excitation trajectory (mover script)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--duration',      type=float, default=60.0,
                        help='Trajectory duration (s)')
    parser.add_argument('--rate',          type=float, default=50.0,
                        help='Command rate (Hz)')
    parser.add_argument('--robot-model',   default='vx300s',
                        help='Interbotix robot model string')
    parser.add_argument('--min-ee-height', type=float, default=0.05,
                        help='Minimum allowed EE z-height (m); waypoints below this are skipped')
    parser.add_argument('--seed',          type=int,   default=42,
                        help='RNG seed — use the same seed to reproduce a trajectory')
    args = parser.parse_args()

    dt  = 1.0 / args.rate
    rng = np.random.default_rng(args.seed)

    # Trajectory profile times: 2× command period gives smooth overlap between
    # successive setpoints; clamped to 80 ms minimum for servo reliability.
    moving_time = max(2.0 * dt, 0.08)
    accel_time  = max(0.4 * dt, 0.02)

    print(f'[mover] Building {args.duration:.0f}s trajectory @ {args.rate:.0f} Hz …')
    waypoints = build_trajectory(args.duration, args.rate, rng)
    n_skip    = sum(1 for _, q in waypoints if ee_z_estimate(q) < args.min_ee_height)
    print(f'[mover] {len(waypoints)} waypoints, {n_skip} skipped (EE below {args.min_ee_height} m)')
    print(f'[mover] moving_time={moving_time:.3f}s  accel_time={accel_time:.3f}s')

    print(f'[mover] Connecting to "{args.robot_model}" …')
    bot = InterbotixManipulatorXS(
        robot_model=args.robot_model,
        group_name='arm',
        gripper_name='gripper',
    )

    print('[mover] Moving to HOME position …')
    bot.arm.set_joint_positions(HOME_POS, moving_time=2.0, accel_time=0.5, blocking=True)
    time.sleep(2.5)

    print('[mover] Waiting 1 s — ensure recorder is running in Terminal 1 …')
    time.sleep(1.0)

    print('[mover] Executing excitation trajectory …')
    t_start = time.monotonic()
    n_sent  = 0

    for step_t, q in waypoints:
        if ee_z_estimate(q) < args.min_ee_height:
            continue

        # Pace to the scheduled wall-clock deadline for this waypoint.
        # If we are already late (slow service call, OS jitter), just send
        # the command immediately without sleeping.
        remaining = (t_start + step_t) - time.monotonic()
        if remaining > 0.0:
            time.sleep(remaining)

        bot.arm.set_joint_positions(
            q.tolist(),
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=False,
        )
        n_sent += 1

    elapsed = time.monotonic() - t_start
    actual_rate = n_sent / elapsed if elapsed > 0 else 0.0
    print(f'[mover] Trajectory complete — {n_sent} commands in {elapsed:.1f}s '
          f'(≈{actual_rate:.1f} Hz achieved)')

    print('[mover] Returning to sleep pose …')
    bot.arm.go_to_sleep_pose()
    print('[mover] Done.')


if __name__ == '__main__':
    main()
