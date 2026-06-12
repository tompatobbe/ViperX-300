#!/usr/bin/env python3
"""
Record joint states from a VX300s to a CSV file.

Usage:
    python3 record_arm_data.py [--duration 60] [--rate 50] [--output data/arm_data.csv]
"""
import time, csv, argparse
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=60.0,     help="record duration in seconds")
    ap.add_argument("--rate",     type=float, default=50.0,      help="sample rate in Hz")
    ap.add_argument("--output",   default="data/arm_data.csv",   help="output CSV path")
    args = ap.parse_args()

    bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper")

    # Build header from first message
    js = bot.core.robot_get_joint_states()
    names = list(js.name)
    header = ["time"] + sum([[f"{n}_pos", f"{n}_vel", f"{n}_effort"] for n in names], [])

    interval = 1.0 / args.rate
    deadline = time.time() + args.duration
    rows = 0

    print(f"Recording {args.duration:.0f}s at {args.rate:.0f}Hz → {args.output}  (Ctrl-C to stop early)")

    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        try:
            while time.time() < deadline:
                t0 = time.time()
                js = bot.core.robot_get_joint_states()
                name_to_idx = {n: i for i, n in enumerate(js.name)}
                row = [t0]
                for n in names:
                    i = name_to_idx.get(n)
                    row += [
                        js.position[i] if i is not None else "",
                        js.velocity[i] if i is not None else "",
                        js.effort[i]   if i is not None else "",
                    ]
                writer.writerow(row)
                rows += 1
                rem = interval - (time.time() - t0)
                if rem > 0:
                    time.sleep(rem)
        except KeyboardInterrupt:
            pass

    print(f"Saved {rows} rows → {args.output}")


if __name__ == "__main__":
    main()
