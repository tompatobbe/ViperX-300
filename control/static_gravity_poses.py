#!/usr/bin/env python3
"""Static-pose gravity experiment — MOVER ONLY.

WHY: every model validation so far is open-loop prediction of *recorded* current,
which shares a circularity with what a controller would command. Holding the arm
still at a spread of poses and reading the steady holding current is the clean,
control-loop-free, real-world check of the GRAVITY model: at standstill the
holding current ≈ gravity (+ small stiction), with NO friction/inertia/Coriolis
confound. It also settles the ≈0.6 measured-vs-paper amplitude anomaly
(CHANGELOG 2026-06-13) without trusting any moving-trajectory fit.

DESIGN (mirrors collect_200hz.sh): this script ONLY moves the arm and dwells at
each pose. The PROVEN recorder (record_joint_states_200hz.py) captures the joint
states in parallel — so we reuse the collection path that produced the 200 Hz
dataset and avoid re-implementing (and mis-spinning) joint-state reading.
Segmentation into per-pose stationary windows happens offline in
tools/analyze_static_gravity.py via motion detection on the velocity columns.

SAFE BY CONSTRUCTION: position mode (the servo's own PID holds each pose — we
never command current), slow blocking moves, poses inside the run_trajectories
limits, gripper grasped + NO payload to match the identification condition.

Normally launched by collect_static_gravity.sh. Manual two-terminal use:
    # Terminal 1 (recorder):
    python3 -u record_joint_states_200hz.py --duration 300 \
        --output data/static_gravity_<date>.csv --topic /vx300s/joint_states
    # Terminal 2 (this mover):
    python3 control/static_gravity_poses.py --sidecar data/static_gravity_<date>.poses.json
"""
import argparse
import json
import time

import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# Joint limits (rad) — copied from run_trajectories.py so we never exceed the
# range the trajectories already proved safe.
LIMITS_LO = np.array([-2.80, -1.50, -0.20, -1.50, -1.50, -2.80])
LIMITS_HI = np.array([ 2.80,  0.30,  1.00,  1.50,  1.50,  2.80])

# Curated static poses [waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate].
# Design goals:
#   - sweep the shoulder (dominant gravity joint) across its range
#   - sweep the elbow, and combine shoulder+elbow for spread of total reach
#   - several poses with a BENT wrist (wrist_angle != 0) and varied forearm_roll
#     to probe the forearm_roll gravity anti-correlation found vs the paper model
#     (forearm_roll only sees gravity when the distal CoM is off its roll axis)
#   - waist parked at 0 (vertical axis ⇒ ~0 gravity; a sanity null)
#   - repeat the home pose first & last to check repeatability / drift
POSES = np.array([
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],   # 0  home / null
    [0.0, -1.20, 0.0,  0.0,  0.0,  0.0],   # 1  shoulder back
    [0.0, -0.60, 0.0,  0.0,  0.0,  0.0],   # 2  shoulder mid
    [0.0,  0.20, 0.0,  0.0,  0.0,  0.0],   # 3  shoulder fwd
    [0.0, -0.50, 1.00, 0.0,  0.0,  0.0],   # 4  elbow up
    [0.0, -0.50, 0.50, 0.0,  0.0,  0.0],   # 5  elbow mid
    [0.0, -0.80, 0.80, 0.0,  0.0,  0.0],   # 6  reach out
    [0.0,  0.00, 1.00, 0.0,  0.0,  0.0],   # 7  folded up
    [0.0, -0.60, 0.50, 0.0,  1.20, 0.0],   # 8  wrist bent +
    [0.0, -0.60, 0.50, 0.0, -1.20, 0.0],   # 9  wrist bent -
    [0.0, -0.60, 0.50, 1.20, 1.00, 0.0],   # 10 forearm_roll + , wrist bent
    [0.0, -0.60, 0.50, -1.20, 1.00, 0.0],  # 11 forearm_roll - , wrist bent
    [0.0, -0.60, 0.50, 1.20, -1.00, 0.0],  # 12 forearm_roll + , wrist bent other way
    [0.0, -0.90, 0.70, 0.0,  0.80, 0.0],   # 13 combined reach + bent wrist
    [0.0,  0.0,  0.0,  0.0,  0.0,  0.0],   # 14 home again (repeatability)
])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-model", default="vx300s")
    ap.add_argument("--move-time", type=float, default=4.0, help="s per move (slow)")
    ap.add_argument("--dwell", type=float, default=6.0,
                    help="s to hold each pose (recorder captures this window)")
    ap.add_argument("--sidecar", default=None,
                    help="JSON of commanded poses + dwell timestamps (for analysis)")
    args = ap.parse_args()

    poses = np.clip(POSES, LIMITS_LO, LIMITS_HI)
    print(f"[static] MOVER: {len(poses)} poses | move {args.move_time}s | dwell {args.dwell}s")
    print("[static] (run record_joint_states_200hz.py in parallel — or use collect_static_gravity.sh)")
    print(f"[static] Connecting to '{args.robot_model}' …")
    bot = InterbotixManipulatorXS(robot_model=args.robot_model,
                                  group_name='arm', gripper_name='gripper')

    print("[static] Moving to home & grasping (no payload, matches identification) …")
    bot.arm.set_joint_positions(poses[0].tolist(), moving_time=args.move_time,
                                accel_time=args.move_time * 0.25, blocking=True)
    bot.gripper.grasp(delay=1.0)
    print("[static] Waiting 2 s — ensure the recorder is running …")
    time.sleep(2.0)

    log = []
    try:
        for k, q in enumerate(poses):
            print(f"[static] pose {k:2d}/{len(poses)-1}  cmd={np.round(q,2).tolist()}")
            bot.arm.set_joint_positions(q.tolist(), moving_time=args.move_time,
                                        accel_time=args.move_time * 0.25, blocking=True)
            t0 = time.time()
            time.sleep(args.dwell)
            t1 = time.time()
            log.append({"pose": k, "q": q.tolist(),
                        "dwell_start_wall": t0, "dwell_end_wall": t1})
    except KeyboardInterrupt:
        print("\n[static] Interrupted.")
    finally:
        print("[static] Returning to sleep pose …")
        bot.gripper.release(delay=0.5)
        bot.arm.go_to_sleep_pose(moving_time=3.0, accel_time=0.5)

    if args.sidecar and log:
        with open(args.sidecar, "w") as f:
            json.dump({"joint_names": JOINT_NAMES, "move_time": args.move_time,
                       "dwell": args.dwell, "poses": log}, f, indent=2)
        print(f"[static] Wrote pose sidecar → {args.sidecar}")


if __name__ == "__main__":
    main()
