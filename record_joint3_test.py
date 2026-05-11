#!/usr/bin/env python3
"""
Move to home (all zeros), sweep elbow (joint 3) to +0.5 then -0.5 rad,
and record joint states to CSV the whole time (~5 seconds of motion).

Usage:
    python3 record_joint3_test.py [--rate 50] [--output data/joint3_test.csv]
"""
import time
import csv
import threading
import argparse

import rclpy
from rclpy.executors import MultiThreadedExecutor
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

HOME      = [0.0,  0.0,  0.0, 0.0, 0.0, 0.0]
ELBOW_UP  = [0.0,  0.0,  0.5, 0.0, 0.0, 0.0]   # elbow +0.5 rad
ELBOW_DN  = [0.0,  0.0, -0.5, 0.0, 0.0, 0.0]   # elbow -0.5 rad
MOVE_TIME = 1.0                                   # seconds per move


def recorder(bot, names, rate, rows, stop_event):
    """Background thread — polls joint states at `rate` Hz until stop_event is set."""
    interval = 1.0 / rate
    while not stop_event.is_set():
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
        rows.append(row)
        elapsed = time.time() - t0
        rem = interval - elapsed
        if rem > 0:
            time.sleep(rem)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate",   type=float, default=50.0,               help="sample rate in Hz")
    ap.add_argument("--output", default="data/joint3_test.csv",         help="output CSV path")
    args = ap.parse_args()

    bot = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper")

    # Spin the ROS node in a background thread so joint-state callbacks keep firing
    executor = MultiThreadedExecutor()
    executor.add_node(bot.core.robot_node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    js    = bot.core.robot_get_joint_states()
    names = list(js.name)
    header = ["time"] + sum([[f"{n}_pos", f"{n}_vel", f"{n}_effort"] for n in names], [])

    # ── Move to home first (not recorded) ────────────────────────────────────
    print("Moving to home (all zeros) …")
    bot.arm.set_joint_positions(HOME, moving_time=3.0, blocking=True)
    time.sleep(0.3)

    # ── Start recording thread ────────────────────────────────────────────────
    rows       = []
    stop_event = threading.Event()
    rec_thread = threading.Thread(target=recorder,
                                  args=(bot, names, args.rate, rows, stop_event),
                                  daemon=True)
    rec_thread.start()

    # ── Commanded movements (blocking=True so we wait for each to finish) ────
    print("Elbow → +0.5 rad …")
    bot.arm.set_joint_positions(ELBOW_UP, moving_time=MOVE_TIME, blocking=True)

    print("Elbow → -0.5 rad …")
    bot.arm.set_joint_positions(ELBOW_DN, moving_time=MOVE_TIME, blocking=True)

    # ── Stop recording ────────────────────────────────────────────────────────
    stop_event.set()
    rec_thread.join()

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(args.output, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"Saved {len(rows)} rows → {args.output}")

    print("Returning to home …")
    bot.arm.set_joint_positions(HOME, moving_time=2.0, blocking=True)


if __name__ == "__main__":
    main()
