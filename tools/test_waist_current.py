#!/usr/bin/env python3
"""
Test current control on the waist joint only.

All other joints remain in position mode and hold HOME, so the arm
cannot collapse. A slow sinusoidal current is applied to the waist
to verify that current control actually moves the joint.

Soft position limits scale the current to zero before the hard limit
is reached, so the joint never gets stuck at the end stop.

Usage:
    python3 test_waist_current.py
    python3 test_waist_current.py --amplitude 150 --frequency 0.3 --duration 20
"""
import argparse
import math
import threading
import time

from rclpy.node import Node
from rclpy.executors import SingleThreadedExecutor
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointSingleCommand
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


HOME_POS = [0.0, -0.5, 0.5, 0.0, 0.3, 0.0]

# Waist hard limits (rad) — capped at ±π/2 (±90°, half a rotation from centre)
WAIST_LO = -math.pi / 2   # ≈ -1.571 rad
WAIST_HI =  math.pi / 2   # ≈  1.571 rad

# Soft limit buffer — current ramps to zero over this range before the hard limit
_SOFT_BUFFER = 0.20  # rad
WAIST_SOFT_LO = WAIST_LO + _SOFT_BUFFER
WAIST_SOFT_HI = WAIST_HI - _SOFT_BUFFER


def soft_scale(current: float, waist_pos: float) -> float:
    """
    Only reduce current that pushes the joint further toward a limit.
    Current pulling the joint back toward centre is always allowed through.
    """
    if waist_pos <= WAIST_LO and current < 0.0:
        return 0.0
    if waist_pos >= WAIST_HI and current > 0.0:
        return 0.0
    if waist_pos < WAIST_SOFT_LO and current < 0.0:
        # Pushing further negative — ramp down
        return current * (waist_pos - WAIST_LO) / _SOFT_BUFFER
    if waist_pos > WAIST_SOFT_HI and current > 0.0:
        # Pushing further positive — ramp down
        return current * (WAIST_HI - waist_pos) / _SOFT_BUFFER
    return current


class WaistCurrentNode(Node):
    def __init__(self, robot_model: str) -> None:
        super().__init__('waist_current_test')
        self._lock     = threading.Lock()
        self._waist_pos = 0.0

        self._pub = self.create_publisher(
            JointSingleCommand,
            f'/{robot_model}/commands/joint_single',
            10,
        )
        self.create_subscription(
            JointState,
            f'/{robot_model}/joint_states',
            self._cb,
            10,
        )

    def _cb(self, msg: JointState) -> None:
        for name, pos in zip(msg.name, msg.position):
            if name == 'waist':
                with self._lock:
                    self._waist_pos = pos
                break

    def waist_pos(self) -> float:
        with self._lock:
            return self._waist_pos

    def send(self, current_ma: float) -> None:
        msg = JointSingleCommand()
        msg.name = 'waist'
        msg.cmd  = float(current_ma)
        self._pub.publish(msg)

    def zero(self) -> None:
        self.send(0.0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Test sinusoidal current on waist joint (all others hold position)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--amplitude',   type=float, default=300.0,
                        help='Peak current amplitude (mA)')
    parser.add_argument('--frequency',   type=float, default=0.2,
                        help='Sinusoid frequency (Hz)')
    parser.add_argument('--duration',    type=float, default=20.0,
                        help='Test duration (s)')
    parser.add_argument('--rate',        type=float, default=50.0,
                        help='Command rate (Hz)')
    parser.add_argument('--robot-model', default='vx300s',
                        help='Interbotix robot model string')
    args = parser.parse_args()

    omega = 2.0 * math.pi * args.frequency

    # ── Connect and move to HOME in position mode ─────────────────────────────
    print(f'[waist_test] Connecting to "{args.robot_model}" …')
    bot = InterbotixManipulatorXS(
        robot_model=args.robot_model,
        group_name='arm',
        gripper_name='gripper',
    )

    print('[waist_test] Ensuring waist is in position-control mode …')
    bot.core.robot_set_operating_modes(cmd_type='single', name='waist', mode='position')
    time.sleep(0.3)

    print('[waist_test] Moving to HOME …')
    bot.arm.set_joint_positions(HOME_POS, moving_time=2.0, accel_time=0.5, blocking=True)
    time.sleep(2.0)

    # ── Start position monitor in background thread ───────────────────────────
    node     = WaistCurrentNode(args.robot_model)
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    # ── Switch waist only to current mode ─────────────────────────────────────
    print('[waist_test] Switching waist to current-control mode …')
    bot.core.robot_set_operating_modes(cmd_type='single', name='waist', mode='current')

    print(f'[waist_test] Running {args.duration:.0f}s @ {args.amplitude:.0f} mA amplitude, '
          f'{args.frequency:.2f} Hz …')
    print(f'[waist_test] Soft limits: [{WAIST_SOFT_LO:.2f}, {WAIST_SOFT_HI:.2f}] rad')
    print('[waist_test] Press Ctrl+C to stop early.')

    dt      = 1.0 / args.rate
    t_start = time.monotonic()

    try:
        while True:
            t = time.monotonic() - t_start
            if t >= args.duration:
                break

            raw_current   = args.amplitude * math.sin(omega * t)
            pos           = node.waist_pos()
            safe_current  = soft_scale(raw_current, pos)
            node.send(safe_current)
            if int(t * 2) % 2 == 0:  # print once per second
                print(f'\r  t={t:5.1f}s  waist={pos:+.3f} rad  cmd={safe_current:+7.1f} mA', end='', flush=True)
            time.sleep(dt)

    finally:
        for _ in range(10):
            node.zero()
            time.sleep(0.02)
        print('[waist_test] Switching waist back to position-control mode …')
        bot.core.robot_set_operating_modes(cmd_type='single', name='waist', mode='position')
        time.sleep(0.3)
        executor.shutdown()

    print('[waist_test] Returning to sleep pose …')
    bot.arm.go_to_sleep_pose()
    print('[waist_test] Done.')


if __name__ == '__main__':
    main()
