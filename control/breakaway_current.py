#!/usr/bin/env python3
"""Breakaway-current (stiction) test — single joint, current control.

WHY: the static-gravity experiment (CHANGELOG 2026-06-13) found the steady
holding current is ~0.58x the identified gravity current, with PERFECT gravity
*shape* (shoulder r=0.997). The leading explanation is gearbox stiction: at
standstill the geared XM540 holds part of the load by static friction, so the
position-mode servo only needs ~0.58x gravity to stay put. This test measures
that stiction directly and, as a by-product, a stiction-FREE gravity estimate.

PHYSICS: at a fixed pose a joint stays still for any commanded current in the
band [g - f, g + f], where g = the current that exactly balances gravity and
f = static-friction (stiction) current. Ramp current UP from rest until the
joint breaks away -> I_break+ = g + f. Ramp DOWN -> I_break- = g - f. Then:
    f (stiction)        = (I_break+ - I_break-) / 2
    g (gravity, clean)  = (I_break+ + I_break-) / 2
Comparing g to the identified gravity current (offline) tells us whether the
0.58 anomaly is PURE stiction (g ~ identified, model correct -> PD+G safe) or
also a real gravity SCALE error (g ~ 0.58 x identified).

MODEL-FREE BASELINE: before switching to current mode we read the joint's actual
present current while position-mode holds the pose. That value sits inside the
band, so we start the ramp there (bump-free) -- no gravity estimate needed here.

SAFE BY CONSTRUCTION (mirrors tools/test_waist_current.py):
  - ONLY the tested joint is in current mode; all others stay in position mode
    and hold the pose, so the arm cannot collapse.
  - Ramp starts at the measured holding current (no current jump).
  - Small ramp steps; breakaway is caught at a few-degrees position deviation,
    at which point the joint is IMMEDIATELY put back in position mode (its PID
    actively recaptures and holds) -- no runaway.
  - Hard current cap (--max-current); absolute position abort window.

The tested joint's master motor is commanded; on the dual-motor joints
(shoulder, elbow) the shadow mirrors it, and the present current we read is the
master's -- the same convention as the static-gravity analysis (raw mA).

Prerequisite (own terminal): the arm driver must be running:
    ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s

Run (source ROS + interbotix workspace first):
    python3 control/breakaway_current.py --joint shoulder
    python3 control/breakaway_current.py --joint waist     # gravity-free control
    python3 control/breakaway_current.py --joint elbow
"""
import argparse
import json
import threading
import time

import numpy as np
from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# Default test pose per joint [waist, shoulder, elbow, forearm_roll, wrist_angle,
# wrist_rotate]. Chosen so the tested joint carries a MODERATE gravity load
# (band well inside the current cap) and reuses poses from the static-gravity
# experiment. Override with --pose.
DEFAULT_POSE = {
    'waist':        [0.0, -0.50, 0.50, 0.0, 0.30, 0.0],  # gravity-free axis: g~0, pure stiction
    'shoulder':     [0.0, -0.80, 0.50, 0.0, 0.00, 0.0],  # upper arm tilted back -> modest gravity
    'elbow':        [0.0, -0.50, 0.50, 0.0, 0.00, 0.0],  # elbow carries forearm+wrist
    'forearm_roll': [0.0, -0.60, 0.50, 0.0, 1.00, 0.0],  # distal CoM off the roll axis
    'wrist_angle':  [0.0, -0.60, 0.50, 0.0, 0.40, 0.0],
    'wrist_rotate': [0.0, -0.60, 0.50, 0.0, 0.40, 0.0],  # ~gravity-free: control like waist
}

# Hard per-joint position window (rad): copied from run_trajectories limits used
# in static_gravity_poses.py. The tested joint is also clamped to start +/- a
# narrow ABORT window around the pose.
LIMITS_LO = dict(zip(ARM_JOINTS, [-2.80, -1.50, -0.20, -1.50, -1.50, -2.80]))
LIMITS_HI = dict(zip(ARM_JOINTS, [ 2.80,  0.30,  1.00,  1.50,  1.50,  2.80]))


class JointMonitor(Node):
    """Publishes single-joint commands and tracks the tested joint's state."""

    def __init__(self, robot_model: str, joint: str) -> None:
        super().__init__('breakaway_test')
        self._joint = joint
        self._lock = threading.Lock()
        self._pos = 0.0
        self._vel = 0.0
        self._cur = 0.0  # present current (mA), master motor
        self._have = False

        self._pub = self.create_publisher(
            JointSingleCommand, f'/{robot_model}/commands/joint_single', 10)
        self.create_subscription(
            JointState, f'/{robot_model}/joint_states', self._cb, 10)

    def _cb(self, msg: JointState) -> None:
        for i, name in enumerate(msg.name):
            if name == self._joint:
                with self._lock:
                    self._pos = msg.position[i]
                    self._vel = msg.velocity[i] if i < len(msg.velocity) else 0.0
                    self._cur = msg.effort[i] if i < len(msg.effort) else 0.0
                    self._have = True
                break

    def state(self):
        with self._lock:
            return self._pos, self._vel, self._cur

    def wait_state(self, timeout=5.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            with self._lock:
                if self._have:
                    return
            time.sleep(0.02)
        raise RuntimeError(f"no joint_states for '{self._joint}' within {timeout}s")

    def send_current(self, ma: float) -> None:
        msg = JointSingleCommand()
        msg.name = self._joint
        msg.cmd = float(ma)
        self._pub.publish(msg)

    def send_position(self, rad: float) -> None:
        msg = JointSingleCommand()
        msg.name = self._joint
        msg.cmd = float(rad)
        self._pub.publish(msg)


def avg_present_current(node: JointMonitor, seconds: float) -> float:
    """Average the present current over a window (denoise the band baseline)."""
    vals = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        vals.append(node.state()[2])
        time.sleep(0.02)
    return float(np.mean(vals)) if vals else 0.0


def ramp_to_breakaway(node, bot, joint, target_pos, base_ma, direction,
                      step, dwell, move_thresh, abort_thresh, max_current, log):
    """Ramp current from base in `direction` (+1/-1) until the joint breaks away.

    Returns the commanded current at breakaway (mA), or None if the cap was hit
    without motion. On breakaway (or abort) the joint is returned to position
    mode and re-homed to target_pos.
    """
    p_phase0 = node.state()[0]
    i_ma = base_ma
    breakaway = None
    print(f"  [{'+' if direction > 0 else '-'} ramp] base={base_ma:+.0f} mA, "
          f"step={step:.0f} mA, dwell={dwell:.2f}s, thresh={move_thresh:.3f} rad")
    try:
        while abs(i_ma) <= max_current + 1e-6:
            node.send_current(i_ma)
            t0 = time.monotonic()
            moved = False
            while time.monotonic() - t0 < dwell:
                pos, vel, cur = node.state()
                dev = pos - p_phase0
                log.append([time.time(), joint, i_ma, pos, vel, cur])
                if abs(dev) > abort_thresh:
                    print(f"\n  ABORT: |dev|={abs(dev):.3f} > {abort_thresh:.3f} rad")
                    breakaway = i_ma
                    moved = True
                    break
                if direction > 0 and dev > move_thresh:
                    moved = True
                elif direction < 0 and dev < -move_thresh:
                    moved = True
                if moved:
                    breakaway = i_ma
                    print(f"\n  BREAKAWAY at {i_ma:+.0f} mA "
                          f"(dev={dev:+.3f} rad, vel={vel:+.3f})")
                    break
                time.sleep(0.01)
            if moved:
                break
            print(f"\r    holding @ {i_ma:+.0f} mA  pos={node.state()[0]:+.3f} "
                  f"(dev={node.state()[0]-p_phase0:+.3f})", end='', flush=True)
            i_ma += direction * step
        else:
            print(f"\n  reached cap {max_current:.0f} mA with no breakaway")
    finally:
        # Always hand the joint back to its own position PID and re-home it.
        node.send_current(0.0)
        bot.core.robot_set_operating_modes(cmd_type='single', name=joint, mode='position')
        time.sleep(0.2)
        node.send_position(target_pos)
        time.sleep(2.0)
    return breakaway


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--joint', default='shoulder', choices=ARM_JOINTS)
    ap.add_argument('--pose', type=float, nargs=6, default=None,
                    help='6 joint positions (rad); default depends on --joint')
    ap.add_argument('--step', type=float, default=15.0, help='ramp step (mA)')
    ap.add_argument('--dwell', type=float, default=0.6, help='hold per step (s)')
    ap.add_argument('--move-thresh', type=float, default=0.04,
                    help='position deviation counted as breakaway (rad)')
    ap.add_argument('--abort-thresh', type=float, default=0.20,
                    help='absolute position deviation that aborts a ramp (rad)')
    ap.add_argument('--max-current', type=float, default=600.0,
                    help='hard current cap (mA); ramp stops here')
    ap.add_argument('--move-time', type=float, default=4.0, help='homing move (s)')
    ap.add_argument('--robot-model', default='vx300s')
    ap.add_argument('--output', default=None, help='CSV trace path')
    args = ap.parse_args()

    joint = args.joint
    pose = list(args.pose) if args.pose is not None else list(DEFAULT_POSE[joint])
    pose = np.clip(pose, [LIMITS_LO[j] for j in ARM_JOINTS],
                   [LIMITS_HI[j] for j in ARM_JOINTS]).tolist()
    target = pose[ARM_JOINTS.index(joint)]
    stamp = time.strftime('%Y%m%d_%H%M%S')
    out_csv = args.output or f'data/breakaway_{joint}_{stamp}.csv'
    out_json = out_csv.rsplit('.', 1)[0] + '.json'

    print(f"[breakaway] joint={joint}  pose={np.round(pose, 2).tolist()}")
    print(f"[breakaway] step={args.step} mA  dwell={args.dwell}s  "
          f"cap={args.max_current} mA  thresh={args.move_thresh} rad")
    print(f"[breakaway] Connecting to '{args.robot_model}' …")
    bot = InterbotixManipulatorXS(robot_model=args.robot_model,
                                  group_name='arm', gripper_name='gripper')

    # Everything in position mode; move to pose; grasp (no payload, matches id).
    bot.core.robot_set_operating_modes(cmd_type='group', name='arm', mode='position')
    time.sleep(0.3)
    print("[breakaway] Moving to test pose & grasping …")
    bot.arm.set_joint_positions(pose, moving_time=args.move_time,
                                accel_time=args.move_time * 0.25, blocking=True)
    bot.gripper.grasp(delay=1.0)
    time.sleep(1.5)

    node = JointMonitor(args.robot_model, joint)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    threading.Thread(target=executor.spin, daemon=True).start()
    node.wait_state()

    log = []
    i_break_plus = i_break_minus = None
    try:
        # ---- + direction: ramp up from the position-mode holding current ------
        base = avg_present_current(node, 1.0)
        print(f"\n[breakaway] position-mode holding current ~ {base:+.0f} mA")
        print("[breakaway] Switching joint to current mode (+ ramp) …")
        node.send_current(base)
        bot.core.robot_set_operating_modes(cmd_type='single', name=joint, mode='current')
        time.sleep(0.3)
        node.send_current(base)
        time.sleep(0.5)
        i_break_plus = ramp_to_breakaway(
            node, bot, joint, target, base, +1, args.step, args.dwell,
            args.move_thresh, args.abort_thresh, args.max_current, log)

        # ---- - direction: ramp down from the (re-read) holding current --------
        base = avg_present_current(node, 1.0)
        print(f"\n[breakaway] holding current re-read ~ {base:+.0f} mA")
        print("[breakaway] Switching joint to current mode (- ramp) …")
        node.send_current(base)
        bot.core.robot_set_operating_modes(cmd_type='single', name=joint, mode='current')
        time.sleep(0.3)
        node.send_current(base)
        time.sleep(0.5)
        i_break_minus = ramp_to_breakaway(
            node, bot, joint, target, base, -1, args.step, args.dwell,
            args.move_thresh, args.abort_thresh, args.max_current, log)

    except KeyboardInterrupt:
        print("\n[breakaway] Interrupted.")
    finally:
        node.send_current(0.0)
        bot.core.robot_set_operating_modes(cmd_type='single', name=joint, mode='position')
        time.sleep(0.3)

    # ---- results ------------------------------------------------------------
    g = f = None
    if i_break_plus is not None and i_break_minus is not None:
        g = 0.5 * (i_break_plus + i_break_minus)
        f = 0.5 * (i_break_plus - i_break_minus)
    print("\n========== RESULTS ==========")
    print(f"  joint            : {joint}")
    print(f"  I_break+ (g+f)   : {i_break_plus}")
    print(f"  I_break- (g-f)   : {i_break_minus}")
    if g is not None:
        print(f"  g  (gravity, mA) : {g:+.0f}   <- compare to identified gravity current")
        print(f"  f  (stiction, mA): {f:+.0f}   ({100*abs(f)/abs(g):.0f}% of g)" if g else '')
    print("=============================")

    if log:
        import csv
        with open(out_csv, 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['wall_time', 'joint', 'cmd_ma', 'pos', 'vel', 'present_ma'])
            w.writerows(log)
        with open(out_json, 'w') as fh:
            json.dump({'joint': joint, 'pose': pose, 'step': args.step,
                       'dwell': args.dwell, 'move_thresh': args.move_thresh,
                       'max_current': args.max_current,
                       'I_break_plus': i_break_plus, 'I_break_minus': i_break_minus,
                       'g_gravity_ma': g, 'f_stiction_ma': f}, fh, indent=2)
        print(f"[breakaway] trace → {out_csv}\n[breakaway] result → {out_json}")

    print("[breakaway] Returning to sleep pose …")
    bot.gripper.release(delay=0.5)
    bot.arm.go_to_sleep_pose(moving_time=3.0, accel_time=0.5)
    executor.shutdown()
    print("[breakaway] Done.")


if __name__ == '__main__':
    main()
