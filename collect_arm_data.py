#!/usr/bin/env python3
"""
Collect joint state data while executing a sum-of-sinusoids excitation trajectory.

Columns: time (s since epoch), <joint>_pos, <joint>_vel, <joint>_effort for each joint.

Usage example:
  python3 collect_arm_data.py --duration 60 --rate 50 --output data/arm_data.csv
"""
import time
import csv
import argparse
import numpy as np
from typing import List

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


# ── Forward kinematics — Modified (Craig) DH, matching sim.py ─────────────────
# Convention:  T_i = Rx(α_{i-1}) · Tx(a_{i-1}) · Rz(θ_i + θ_off) · Tz(d_i)
# Columns: α_{i-1}(°)  a_{i-1}(m)  d_i(m)  θ_off(°)
# Joint order: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate
_DH = np.array([
    #  α_{i-1}(°)  a_{i-1}(mm)   d_i(mm)   θ_off(°)    Joint
    [    0.0,         0.0,        126.75,     0.0   ],  # 1  Waist
    [  -90.0,         0.0,          0.0,    -78.66  ],  # 2  Shoulder   (3π/2, -0.437π)
    [    0.0,       305.94,         0.0,    -11.34  ],  # 3  Elbow      (-0.063π)
    [  -90.0,         0.0,         300.0,    0.0   ],  # 4  Wrist Rotate (L3+L4=300+50)
    [   90.0,         0.0,         0.0,    0.0   ],  # 5  Wrist Pitch
    [  -90.0,         0.0,          70.0,     0.0   ],  # 6  Wrist Roll
])


def _ee_z(q: np.ndarray) -> float:
    """Return the z-height [m] of the end-effector frame origin (base frame)."""
    T = np.eye(4)
    for i in range(6):
        α, a, d, θ_off = _DH[i]
        θ = np.radians(np.degrees(q[i]) + θ_off)
        α = np.radians(α)
        cθ, sθ, cα, sα = np.cos(θ), np.sin(θ), np.cos(α), np.sin(α)
        T = T @ np.array([
        [ cθ, -sθ*cα,  sθ*sα, a*cθ],
        [ sθ,  cθ*cα, -cθ*sα, a*sθ],
        [  0,     sα,     cα,    d ],
        [  0,      0,      0,    1 ],
        ])
        
    return float(T[2, 3])


# ── Trajectory definition ──────────────────────────────────────────────────────
# Joint order: waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate

# Safe mid-range home position [rad]
HOME_POS = np.array([0.0, -0.5, 0.5, 0.0, 0.3, 0.0])

# Incommensurable frequencies [Hz] for persistent excitation
FREQS = np.array([0.10, 0.27, 0.53])

# Per-joint amplitudes per frequency [rad] — worst-case sum stays within ±60% of joint range
AMPLITUDES = np.array([
    [0.60, 0.30, 0.15],   # waist        (total ±1.05, range ±3.14)
    [0.35, 0.20, 0.10],   # shoulder     (total ±0.65, home -0.5, range [-1.88, 1.53])
    [0.35, 0.20, 0.10],   # elbow        (total ±0.65, home +0.5, range [-2.15, 1.57])
    [0.50, 0.25, 0.12],   # forearm_roll (total ±0.87, range ±3.14)
    [0.35, 0.20, 0.10],   # wrist_angle  (total ±0.65, home +0.3, range [-1.75, 2.15])
    [0.55, 0.28, 0.14],   # wrist_rotate (total ±0.97, range ±3.14)
])

# Phases all zero so trajectory(0) == HOME_POS — no velocity spike on the first command.
# Joints decorrelate over time naturally because the three frequencies grow at different rates.
PHASES = np.zeros((6, 3))


def trajectory(t: float) -> np.ndarray:
    """Target joint positions at time t [s] relative to trajectory start."""
    return HOME_POS + (AMPLITUDES * np.sin(
        2 * np.pi * FREQS[None, :] * t + PHASES
    )).sum(axis=1)


def sample_loop(bot: InterbotixManipulatorXS, duration: float, rate: float,
                outpath: str, min_ee_height: float = 0.10) -> None:
    js = bot.core.robot_get_joint_states()
    names: List[str] = list(js.name)

    header = ["time"]
    for n in names:
        header += [f"{n}_pos", f"{n}_vel", f"{n}_effort"]

    interval = 1.0 / float(rate) if rate > 0 else 0.02
    moving_time = max(interval * 2.0, 0.20)
    end_time = time.time() + float(duration) if duration > 0 else float('inf')

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        t0 = time.time()
        q_last_safe = HOME_POS.copy()
        skipped = 0
        try:
            while True:
                loop_start = time.time()
                if loop_start >= end_time:
                    break

                # Read state at the TOP of the loop — the sleep at the end of the
                # previous iteration gave the ROS joint-state publisher time to push
                # a fresh message, so this read reflects the arm's actual current pose.
                js = bot.core.robot_get_joint_states()
                row = [loop_start]
                for n in names:
                    try:
                        idx = js.name.index(n)
                    except ValueError:
                        idx = None
                    pos = js.position[idx] if idx is not None and idx < len(js.position) else ""
                    vel = js.velocity[idx] if idx is not None and idx < len(js.velocity) else ""
                    eff = js.effort[idx] if idx is not None and idx < len(js.effort) else ""
                    row += [pos, vel, eff]
                writer.writerow(row)

                # Ground-clearance check: fall back to last safe position if too low
                q_cmd = trajectory(loop_start - t0)
                if _ee_z(q_cmd) >= min_ee_height:
                    q_last_safe = q_cmd
                else:
                    q_cmd = q_last_safe
                    skipped += 1

                bot.arm.set_joint_positions(
                    q_cmd.tolist(),
                    moving_time=moving_time,
                    accel_time=moving_time / 4.0,
                    blocking=True,
                )

                elapsed = time.time() - loop_start
                remaining = interval - elapsed
                if remaining > 0:
                    time.sleep(remaining)
        except KeyboardInterrupt:
            pass
        if skipped:
            print(f"  {skipped} setpoints skipped (EE below {min_ee_height:.2f} m)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Collect joint state data with sum-of-sinusoids excitation trajectory")
    parser.add_argument("--robot-model", default="vx300s", help="robot model (default: vx300s)")
    parser.add_argument("--group", default="arm", help="arm group name (default: arm)")
    parser.add_argument("--gripper", default="gripper", help="gripper name (default: gripper)")
    parser.add_argument("--duration", type=float, default=60.0,
                        help="record duration in seconds (default: 60)")
    parser.add_argument("--rate", type=float, default=50.0, help="sample rate Hz (default: 50)")
    parser.add_argument("--output", default="data/arm_data.csv",
                        help="output CSV path (default: data/arm_data.csv)")
    parser.add_argument("--min-ee-height", type=float, default=0.30,
                        help="minimum end-effector z height in metres (default: 0.30)")
    args = parser.parse_args()

    bot = InterbotixManipulatorXS(
        robot_model=args.robot_model,
        group_name=args.group,
        gripper_name=args.gripper,
    )

    home_z = _ee_z(HOME_POS)
    if home_z < args.min_ee_height:
        raise ValueError(
            f"HOME_POS is only z={home_z:.3f} m, which is below --min-ee-height={args.min_ee_height:.3f} m. "
            f"The arm would never move. Lower --min-ee-height to at most {home_z:.2f} m "
            f"(e.g. --min-ee-height 0.15) or raise HOME_POS."
        )

    try:
        print("Moving to home position …")
        bot.arm.set_joint_positions(HOME_POS.tolist(), moving_time=4.0, blocking=True)
        time.sleep(0.5)

        print(f"Recording {args.duration}s at {args.rate}Hz to {args.output} — press Ctrl+C to stop early")
        print(f"  Ground clearance: EE z ≥ {args.min_ee_height:.2f} m")
        sample_loop(bot, args.duration, args.rate, args.output, args.min_ee_height)
    finally:
        print("Returning to sleep pose …")
        bot.arm.go_to_sleep_pose()
        print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
