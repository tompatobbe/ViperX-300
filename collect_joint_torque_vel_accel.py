#!/usr/bin/env python3
"""
Collect torque, velocity, and acceleration from all joints of an Interbotix arm.

Columns: time (s since epoch), <joint>_vel, <joint>_torque, <joint>_accel for each joint.

Usage example:
  python3 collect_joint_torque_vel_accel.py --duration 30 --rate 50 --output joint_data.csv
"""
import time
import csv
import argparse
from typing import List, Optional

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


def get_joint_value(js, joint_name: str, attr: str) -> Optional[float]:
    try:
        idx = js.name.index(joint_name)
    except ValueError:
        return None

    values = getattr(js, attr, None)
    if values is None or idx >= len(values):
        return None

    return values[idx]


def sample_loop(bot: InterbotixManipulatorXS, duration: float, rate: float, outpath: str) -> None:
    js = bot.core.robot_get_joint_states()
    names: List[str] = list(js.name)

    header = ["time"]
    for n in names:
        header += [f"{n}_vel", f"{n}_torque", f"{n}_accel"]

    interval = 1.0 / float(rate) if rate > 0 else 0.02
    end_time = time.time() + float(duration) if duration > 0 else float('inf')

    prev_velocities: List[Optional[float]] = [None] * len(names)
    prev_time: Optional[float] = None

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        try:
            while time.time() < end_time:
                t = time.time()
                js = bot.core.robot_get_joint_states()
                row = [t]
                current_velocities: List[Optional[float]] = []

                for idx, n in enumerate(names):
                    vel = get_joint_value(js, n, "velocity")
                    torque = get_joint_value(js, n, "effort")
                    accel: Optional[float] = None

                    if prev_time is not None and vel is not None and prev_velocities[idx] is not None:
                        dt = t - prev_time
                        if dt > 0:
                            accel = (vel - prev_velocities[idx]) / dt

                    current_velocities.append(vel)
                    row += [vel if vel is not None else "", torque if torque is not None else "", accel if accel is not None else ""]

                writer.writerow(row)
                prev_velocities = current_velocities
                prev_time = t
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect joint torque, velocity, and acceleration from an Interbotix arm")
    parser.add_argument("--robot-model", default="vx300s", help="robot model (default: vx300s)")
    parser.add_argument("--group", default="arm", help="arm group name (default: arm)")
    parser.add_argument("--gripper", default="gripper", help="gripper name (default: gripper)")
    parser.add_argument("--duration", type=float, default=10.0, help="record duration in seconds (default: 10)")
    parser.add_argument("--rate", type=float, default=50.0, help="sample rate Hz (default: 50)")
    parser.add_argument("--output", default="joint_data.csv", help="output CSV path (default: joint_data.csv)")
    args = parser.parse_args()

    bot = InterbotixManipulatorXS(robot_model=args.robot_model, group_name=args.group, gripper_name=args.gripper)
    print(f"Recording {args.duration}s at {args.rate}Hz to {args.output} — press Ctrl+C to stop")
    sample_loop(bot, args.duration, args.rate, args.output)
    print(f"Saved data to {args.output}")


if __name__ == "__main__":
    main()
