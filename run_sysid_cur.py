#!/usr/bin/env python3
"""
VX300s sum-of-sinusoids current excitation for system identification.

Switches the arm to current-control mode, applies sinusoidal current commands,
and monitors joint positions in real time to scale back current near limits.

Run AFTER record_joint_states.py is running.

Usage:
    python3 run_sysid_cur.py --duration 60 --rate 50
"""
import argparse
import threading
import time

import numpy as np
import rclpy
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


# ── Joint ordering (matches interbotix arm group) ─────────────────────────────
JOINT_NAMES = [
    'waist', 'shoulder', 'elbow',
    'forearm_roll', 'wrist_angle', 'wrist_rotate',
]

HOME_POS = np.array([0.0, -0.5, 0.5, 0.0, 0.3, 0.0])

# Hard safety limits (rad) — same as position script
LIMITS_LO = np.array([-2.80, -1.70, -1.90, -2.80, -1.60, -2.80])
LIMITS_HI = np.array([ 2.80,  1.80,  1.40,  2.80,  2.00,  2.80])

# Soft limits — current ramps to zero linearly over this buffer before the hard limit
_SOFT_BUFFER = 0.20   # rad
SOFT_LO = LIMITS_LO + _SOFT_BUFFER
SOFT_HI = LIMITS_HI - _SOFT_BUFFER

# Per-joint current amplitudes (mA) for excitation — conservative for first run
CURRENT_AMPLITUDES = np.array([150.0, 250.0, 200.0, 120.0, 120.0,  80.0])

# Absolute current cap per joint (mA) — never exceeded regardless of amplitude
MAX_CURRENT_MA = np.array([300.0, 400.0, 350.0, 250.0, 250.0, 150.0])

# Sinusoid component angular frequencies (rad/s)
FREQS = np.array([0.30, 0.60, 1.00, 1.60, 2.50, 3.80])

# ── VX300s geometry for EE z estimate ────────────────────────────────────────
_Z_SHOULDER = 0.127
_L_UPPER    = 0.200
_L_FORE     = 0.265
_L_HAND     = 0.170


def ee_z_estimate(q: np.ndarray) -> float:
    q2, q3, q5 = float(q[1]), float(q[2]), float(q[4])
    return (
        _Z_SHOULDER
        + _L_UPPER * np.sin(q2)
        + _L_FORE  * np.sin(q2 + q3)
        + _L_HAND  * np.sin(q2 + q3 + q5)
    )


class SafetyMonitor(Node):
    """Subscribes to joint states and exposes thread-safe position reads."""

    def __init__(self, robot_model: str) -> None:
        super().__init__('sysid_cur_monitor')
        self._lock = threading.Lock()
        self._positions = HOME_POS.copy()
        self._received = False
        self.create_subscription(
            JointState,
            f'/{robot_model}/joint_states',
            self._cb,
            10,
        )
        self._pub = self.create_publisher(
            JointGroupCommand,
            f'/{robot_model}/commands/joint_group',
            10,
        )

    def _cb(self, msg: JointState) -> None:
        name_to_pos = dict(zip(msg.name, msg.position))
        pos = np.array([name_to_pos.get(n, 0.0) for n in JOINT_NAMES])
        with self._lock:
            self._positions = pos
            self._received = True

    def get_positions(self) -> np.ndarray:
        with self._lock:
            return self._positions.copy()

    def has_data(self) -> bool:
        with self._lock:
            return self._received

    def send_current(self, currents: np.ndarray) -> None:
        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = currents.tolist()
        self._pub.publish(msg)

    def zero_current(self) -> None:
        self.send_current(np.zeros(len(JOINT_NAMES)))


def build_trajectory(
    duration: float,
    rate: float,
    rng: np.random.Generator,
) -> list[tuple[float, np.ndarray]]:
    """Return (timestamp, currents_mA) pairs for the full excitation."""
    dt     = 1.0 / rate
    n      = int(duration * rate)
    t_vec  = np.arange(n) * dt
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(len(JOINT_NAMES), len(FREQS)))

    waypoints: list[tuple[float, np.ndarray]] = []
    for i, t in enumerate(t_vec):
        cur = np.zeros(len(JOINT_NAMES))
        for j in range(len(JOINT_NAMES)):
            for k, w in enumerate(FREQS):
                cur[j] += (CURRENT_AMPLITUDES[j] / len(FREQS)) * np.sin(w * t + phases[j, k])
        cur = np.clip(cur, -MAX_CURRENT_MA, MAX_CURRENT_MA)
        waypoints.append((t, cur))
    return waypoints


def safety_scale(cur: np.ndarray, pos: np.ndarray, min_ee_height: float) -> np.ndarray:
    """
    Per-joint linear ramp-down near soft limits; full zero if hard limit or EE too low.
    Returns scaled current array.
    """
    # Hard limit or EE too low → emergency zero
    if np.any(pos <= LIMITS_LO) or np.any(pos >= LIMITS_HI):
        return np.zeros(len(JOINT_NAMES))
    if ee_z_estimate(pos) < min_ee_height:
        return np.zeros(len(JOINT_NAMES))

    # Soft limit: per-joint linear scale in [0, 1]
    scale = np.ones(len(JOINT_NAMES))
    for j in range(len(JOINT_NAMES)):
        if pos[j] < SOFT_LO[j]:
            scale[j] = max(0.0, (pos[j] - LIMITS_LO[j]) / _SOFT_BUFFER)
        elif pos[j] > SOFT_HI[j]:
            scale[j] = max(0.0, (LIMITS_HI[j] - pos[j]) / _SOFT_BUFFER)

    return cur * scale


def main() -> None:
    parser = argparse.ArgumentParser(
        description='VX300s sum-of-sinusoids current excitation for sysid',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--duration',      type=float, default=60.0,
                        help='Excitation duration (s)')
    parser.add_argument('--rate',          type=float, default=50.0,
                        help='Command rate (Hz)')
    parser.add_argument('--robot-model',   default='vx300s',
                        help='Interbotix robot model string')
    parser.add_argument('--min-ee-height', type=float, default=0.10,
                        help='Min EE z-height (m) — higher than position script for safety')
    parser.add_argument('--seed',          type=int,   default=42,
                        help='RNG seed — use same seed to reproduce a trajectory')
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f'[cur_mover] Building {args.duration:.0f}s current trajectory @ {args.rate:.0f} Hz …')
    waypoints = build_trajectory(args.duration, args.rate, rng)

    # ── Phase 1: home in position mode ───────────────────────────────────────
    print(f'[cur_mover] Connecting to "{args.robot_model}" …')
    bot = InterbotixManipulatorXS(
        robot_model=args.robot_model,
        group_name='arm',
        gripper_name='gripper',
    )

    print('[cur_mover] Moving to HOME position …')
    bot.arm.set_joint_positions(HOME_POS.tolist(), moving_time=2.0, accel_time=0.5, blocking=True)
    time.sleep(2.5)

    # ── Phase 2: start safety monitor in background thread ───────────────────
    monitor = SafetyMonitor(args.robot_model)
    executor = SingleThreadedExecutor()
    executor.add_node(monitor)
    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    print('[cur_mover] Waiting for joint states …')
    deadline = time.monotonic() + 5.0
    while not monitor.has_data() and time.monotonic() < deadline:
        time.sleep(0.05)
    if not monitor.has_data():
        print('[cur_mover] ERROR: no joint states received — aborting.')
        executor.shutdown(wait=False)
        return

    # ── Phase 3: switch to current mode and excite ───────────────────────────
    print('[cur_mover] Switching arm to current-control mode …')
    bot.core.robot_set_operating_modes(cmd_type='group', name='arm', mode='current')
    time.sleep(0.5)

    print('[cur_mover] Executing current excitation …')
    t_start  = time.monotonic()
    n_sent   = 0
    n_zeroed = 0

    try:
        for step_t, cur in waypoints:
            remaining = (t_start + step_t) - time.monotonic()
            if remaining > 0.0:
                time.sleep(remaining)

            pos       = monitor.get_positions()
            safe_cur  = safety_scale(cur, pos, args.min_ee_height)

            if np.any(safe_cur != cur):
                n_zeroed += 1

            monitor.send_current(safe_cur)
            n_sent += 1

    finally:
        monitor.zero_current()
        time.sleep(0.2)

    elapsed     = time.monotonic() - t_start
    actual_rate = n_sent / elapsed if elapsed > 0 else 0.0
    print(
        f'[cur_mover] Trajectory complete — {n_sent} commands in {elapsed:.1f}s '
        f'(≈{actual_rate:.1f} Hz), {n_zeroed} steps safety-scaled'
    )

    # ── Phase 4: return to position mode and sleep ───────────────────────────
    print('[cur_mover] Switching back to position-control mode …')
    bot.core.robot_set_operating_modes(cmd_type='group', name='arm', mode='position')
    time.sleep(0.5)

    print('[cur_mover] Returning to sleep pose …')
    bot.arm.go_to_sleep_pose()

    executor.shutdown(wait=False)
    print('[cur_mover] Done.')


if __name__ == '__main__':
    main()
