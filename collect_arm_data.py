#!/usr/bin/env python3
"""
Simple recorder that samples joint states from an Interbotix arm and writes a CSV.

Columns: time (s since epoch), <joint>_pos, <joint>_vel, <joint>_effort for each joint.

Usage example:
  python3 collect_arm_data.py --duration 30 --rate 50 --output arm_data.csv
"""
import time
import csv
import argparse
from typing import List

from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


def sample_loop(bot: InterbotixManipulatorXS, duration: float, rate: float, outpath: str) -> None:
    js = bot.core.robot_get_joint_states()
    names: List[str] = list(js.name)

    header = ["time"]
    for n in names:
        header += [f"{n}_pos", f"{n}_vel", f"{n}_effort"]

    interval = 1.0 / float(rate) if rate > 0 else 0.02
    end_time = time.time() + float(duration) if duration > 0 else float('inf')

    with open(outpath, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        try:
            while time.time() < end_time:
                t = time.time()
                js = bot.core.robot_get_joint_states()
                row = [t]
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
                time.sleep(interval)
        except KeyboardInterrupt:
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect joint state data from an Interbotix arm")
    parser.add_argument("--robot-model", default="vx300s", help="robot model (default: vx300s)")
    parser.add_argument("--group", default="arm", help="arm group name (default: arm)")
    parser.add_argument("--gripper", default="gripper", help="gripper name (default: gripper)")
    parser.add_argument("--duration", type=float, default=10.0, help="record duration in seconds (default: 10)")
    parser.add_argument("--rate", type=float, default=50.0, help="sample rate Hz (default: 50)")
    parser.add_argument("--output", default="arm_data.csv", help="output CSV path (default: arm_data.csv)")
    args = parser.parse_args()

    bot = InterbotixManipulatorXS(robot_model=args.robot_model, group_name=args.group, gripper_name=args.gripper)
    print(f"Recording {args.duration}s at {args.rate}Hz to {args.output} — press Ctrl+C to stop")
    sample_loop(bot, args.duration, args.rate, args.output)
    print(f"Saved data to {args.output}")


if __name__ == "__main__":
    main()
