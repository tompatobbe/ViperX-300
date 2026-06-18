#!/usr/bin/env python3
"""Standstill-stiction hysteresis experiment — MOVER ONLY.

WHY: the static gravity benchmark found the holding current under-reads the
identified/paper gravity by ~1.6× (the "0.63 anomaly"). The leading explanation
is that gear stiction supports part of the gravity load at standstill, so the
motor draws less than the gravity torque to *hold* a pose (THESIS_NOTES
"Standstill stiction"). Static friction settles anywhere in
[-tau_breakaway, +tau_breakaway] depending on the DIRECTION OF THE LAST MOTION
before stopping. This experiment holds each pose TWICE — approached from opposite
directions — so the difference in holding current measures the stiction band, and
the midpoint cancels stiction and should land on the model gravity.

DESIGN (mirrors control/static_gravity_poses.py): this script ONLY moves the arm
and dwells; the proven recorder (record_joint_states_200hz.py) captures joint
states in parallel. For each target pose we run two trials:
  - "ascending"  : approach from a waypoint at q* - delta  (final motion raises the joint)
  - "descending" : approach from a waypoint at q* + delta  (final motion lowers the joint)
on the gravity-bearing joints (default shoulder, elbow, wrist_angle). The sidecar
records, per dwell, the target index, the approach direction and the dwell wall
window; segmentation + analysis happen offline in
tools/analyze_stiction_hysteresis.py.

SAFE BY CONSTRUCTION (identical envelope to the static experiment): position mode
(the servo PID holds each pose — we never command current), slow blocking moves,
poses + waypoints clipped to the run_trajectories limits, gripper grasped + NO
payload to match the identification condition.

Normally launched by collect_stiction_hysteresis.sh. Manual two-terminal use:
    # Terminal 1 (recorder):
    python3 -u record_joint_states_200hz.py --duration 400 \
        --output data/stiction_hyst_<date>.csv --topic /vx300s/joint_states
    # Terminal 2 (this mover):
    python3 control/stiction_hysteresis_poses.py --sidecar data/stiction_hyst_<date>.poses.json
"""
import argparse
import json
import time

import numpy as np
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

JOINT_NAMES = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# Joint limits (rad) — copied from run_trajectories.py / static_gravity_poses.py.
LIMITS_LO = np.array([-2.80, -1.50, -0.20, -1.50, -1.50, -2.80])
LIMITS_HI = np.array([ 2.80,  0.30,  1.00,  1.50,  1.50,  2.80])

# Target poses — gravity-loaded configurations (shoulder/elbow spread, one bent
# wrist, one forearm_roll probe). Each is held from both approach directions.
TARGETS = np.array([
    [0.0, -1.20, 0.00, 0.0,  0.0, 0.0],   # 0  shoulder back (max shoulder gravity)
    [0.0, -0.60, 0.00, 0.0,  0.0, 0.0],   # 1  shoulder mid
    [0.0, -0.50, 1.00, 0.0,  0.0, 0.0],   # 2  elbow up
    [0.0, -0.80, 0.80, 0.0,  0.0, 0.0],   # 3  reach out (shoulder+elbow)
    [0.0, -0.60, 0.50, 0.0,  1.00, 0.0],  # 4  bent wrist
    [0.0, -0.60, 0.50, 1.20, 1.00, 0.0],  # 5  forearm_roll probe
])

# Joints offset to set the approach direction (the gravity-bearing ones).
SWEPT = [JOINT_NAMES.index(j) for j in ('shoulder', 'elbow', 'wrist_angle')]


def waypoint(q_target, approach, delta):
    """q* offset on the swept joints: ascending starts below (q*-delta) so the
    final move raises the joint; descending starts above (q*+delta)."""
    wp = q_target.copy()
    sign = -1.0 if approach == "ascending" else +1.0
    for j in SWEPT:
        wp[j] = q_target[j] + sign * delta
    return np.clip(wp, LIMITS_LO, LIMITS_HI)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-model", default="vx300s")
    ap.add_argument("--move-time", type=float, default=4.0, help="s per move (slow)")
    ap.add_argument("--dwell", type=float, default=6.0, help="s to hold each pose")
    ap.add_argument("--delta", type=float, default=0.20,
                    help="approach offset on the swept joints [rad]")
    ap.add_argument("--sidecar", default=None,
                    help="JSON of trials (target, approach, dwell timestamps)")
    args = ap.parse_args()

    targets = np.clip(TARGETS, LIMITS_LO, LIMITS_HI)
    # Build the trial list: each target held ascending then descending.
    trials = []
    for k, q in enumerate(targets):
        for approach in ("ascending", "descending"):
            trials.append((k, approach, q))

    print(f"[hyst] MOVER: {len(targets)} targets × 2 approaches = {len(trials)} dwells "
          f"| move {args.move_time}s | dwell {args.dwell}s | delta {args.delta} rad")
    print(f"[hyst] swept joints: {[JOINT_NAMES[j] for j in SWEPT]}")
    print(f"[hyst] Connecting to '{args.robot_model}' …")
    bot = InterbotixManipulatorXS(robot_model=args.robot_model,
                                  group_name='arm', gripper_name='gripper')

    print("[hyst] Moving to home & grasping (no payload, matches identification) …")
    home = np.zeros(6)
    bot.arm.set_joint_positions(home.tolist(), moving_time=args.move_time,
                                accel_time=args.move_time * 0.25, blocking=True)
    bot.gripper.grasp(delay=1.0)
    print("[hyst] Waiting 2 s — ensure the recorder is running …")
    time.sleep(2.0)

    log = []
    try:
        for i, (k, approach, q) in enumerate(trials):
            wp = waypoint(q, approach, args.delta)
            print(f"[hyst] trial {i:2d}/{len(trials)-1}  target {k} {approach:<10} "
                  f"q={np.round(q,2).tolist()}")
            # 1) go to the approach waypoint, 2) make the final monotonic move to q*.
            bot.arm.set_joint_positions(wp.tolist(), moving_time=args.move_time,
                                        accel_time=args.move_time * 0.25, blocking=True)
            bot.arm.set_joint_positions(q.tolist(), moving_time=args.move_time,
                                        accel_time=args.move_time * 0.25, blocking=True)
            t0 = time.time()
            time.sleep(args.dwell)
            t1 = time.time()
            log.append({"trial": i, "target": int(k), "approach": approach,
                        "q": q.tolist(), "waypoint": wp.tolist(),
                        "dwell_start_wall": t0, "dwell_end_wall": t1})
    except KeyboardInterrupt:
        print("\n[hyst] Interrupted.")
    finally:
        print("[hyst] Returning to sleep pose …")
        bot.gripper.release(delay=0.5)
        bot.arm.go_to_sleep_pose(moving_time=3.0, accel_time=0.5)

    if args.sidecar and log:
        with open(args.sidecar, "w") as f:
            json.dump({"joint_names": JOINT_NAMES, "move_time": args.move_time,
                       "dwell": args.dwell, "delta": args.delta,
                       "swept": [JOINT_NAMES[j] for j in SWEPT], "trials": log}, f, indent=2)
        print(f"[hyst] Wrote trial sidecar → {args.sidecar}")


if __name__ == "__main__":
    main()
