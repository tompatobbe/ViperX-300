#!/usr/bin/env python3
"""
VX300s joint-state CSV recorder (RECORDER).

Pure rclpy subscriber — zero interbotix SDK dependency, zero executor conflicts.
Run in Terminal 1 BEFORE starting run_sysid_trajectory.py in Terminal 2.

Usage:
    python3 record_joint_states.py --duration 60 --output data/arm_data.csv

CSV columns:
    time, waist_pos, waist_vel, waist_effort,
          shoulder_pos, ..., wrist_rotate_effort,
          left_finger_pos, ..., right_finger_effort

Stops automatically after --duration seconds, or immediately on Ctrl-C.
Prints the saved row count on exit.
"""
import argparse
import csv
import datetime
import signal
import sys
import time
from pathlib import Path

import rclpy
import rclpy.executors
from rclpy.node import Node
from sensor_msgs.msg import JointState


# All joints published on /vx300s/joint_states, in the desired CSV column order.
# Gripper fingers are appended after the 6 arm joints.
JOINT_ORDER = [
    'waist',
    'shoulder',
    'elbow',
    'forearm_roll',
    'wrist_angle',
    'wrist_rotate',
    'left_finger',
    'right_finger',
]


class JointStateRecorder(Node):
    """Subscribes to /vx300s/joint_states and writes rows to a CSV file."""

    def __init__(self, output_path: str, duration: float, rate: float) -> None:
        super().__init__('joint_state_recorder')

        self._duration  = duration
        self._min_dt    = 1.0 / rate          # minimum seconds between written rows
        self._t_start: float | None = None
        self._t_last:  float | None = None    # wall time of the last written row
        self._row_count = 0
        self.done       = False     # main loop polls this flag

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # line-buffered so data reaches disk continuously (safer on Ctrl-C)
        self._csv_file = open(output_path, 'w', newline='', buffering=1)
        self._writer   = self._init_csv()

        self.create_subscription(
            JointState,
            '/vx300s/joint_states',
            self._callback,
            qos_profile=10,
        )
        self.get_logger().info(
            f'Recorder ready  →  {output_path}  '
            f'(duration={duration:.0f} s, rate={rate:.0f} Hz)'
        )

    # ── CSV initialisation ─────────────────────────────────────────────────────

    def _init_csv(self) -> csv.writer:
        writer = csv.writer(self._csv_file)
        header = ['time']
        for j in JOINT_ORDER:
            header += [f'{j}_pos', f'{j}_vel', f'{j}_effort']
        writer.writerow(header)
        return writer

    # ── ROS2 subscriber callback ───────────────────────────────────────────────

    def _callback(self, msg: JointState) -> None:
        if self.done:
            return

        wall = time.time()

        # Throttle: drop messages that arrive sooner than the target period
        if self._t_last is not None and (wall - self._t_last) < self._min_dt:
            return

        if self._t_start is None:
            self._t_start = wall
            self.get_logger().info('First joint-state message received — recording started.')

        elapsed = wall - self._t_start

        if elapsed >= self._duration:
            self.done = True
            self.get_logger().info(
                f'Duration {self._duration:.0f}s reached — recording stopped.'
            )
            return

        # Build name → index map for this message (order can vary)
        name_idx: dict[str, int] = {n: i for i, n in enumerate(msg.name)}

        def _get(seq, idx: int | None) -> str:
            """Return seq[idx] as a string, or '' if idx is None or out of range."""
            if idx is None or idx >= len(seq):
                return ''
            return repr(float(seq[idx]))

        row: list[str] = [f'{elapsed:.6f}']
        for joint in JOINT_ORDER:
            i = name_idx.get(joint)       # None if joint not in this message
            row.append(_get(msg.position, i))
            row.append(_get(msg.velocity, i))
            row.append(_get(msg.effort,   i))

        self._writer.writerow(row)
        self._row_count += 1
        self._t_last = wall

    # ── Clean shutdown ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the CSV; print summary."""
        self._csv_file.flush()
        self._csv_file.close()
        print(f'\n[recorder] Saved {self._row_count} rows.')
        if self._t_start is not None:
            actual_dur = time.time() - self._t_start
            if actual_dur > 0 and self._row_count > 0:
                print(f'[recorder] Effective rate: {self._row_count / actual_dur:.1f} Hz')


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='VX300s joint-state CSV recorder — pure rclpy, no SDK',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output', default='data/arm_data.csv',
        help='Output CSV file path',
    )
    parser.add_argument(
        '--duration', type=float, default=60.0,
        help='Recording duration in seconds',
    )
    parser.add_argument(
        '--rate', type=float, default=50.0,
        help='Target recording rate in Hz — messages arriving faster are dropped',
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output = output.with_stem(f'{output.stem}_{stamp}')
        print(f'[recorder] {args.output} already exists — writing to {output}')

    rclpy.init()
    recorder = JointStateRecorder(str(output), args.duration, args.rate)

    # Handle Ctrl-C gracefully: set the done flag and let the loop exit cleanly
    def _on_sigint(sig, frame):  # noqa: ANN001
        recorder.get_logger().info('SIGINT received — stopping.')
        recorder.done = True

    signal.signal(signal.SIGINT, _on_sigint)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(recorder)

    try:
        # spin_once loop so we can check recorder.done without a background thread
        while rclpy.ok() and not recorder.done:
            executor.spin_once(timeout_sec=0.05)
    except Exception as exc:
        recorder.get_logger().error(f'Unexpected error: {exc}')
    finally:
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
