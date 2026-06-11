#!/usr/bin/env python3
"""
VX300s joint-state CSV recorder for the 200 Hz collection (Tier-2 RECORDER).

New-file variant of record_joint_states.py for the 200 Hz re-collection
(docs/COLLECTION_200HZ.md); the original recorder is kept untouched so past
runs stay reproducible. Normally launched by collect_200hz.sh, but can be run
by hand in Terminal 1 before run_trajectories.py in Terminal 2:

    python3 -u record_joint_states_200hz.py --duration 900 --output data/traj_run_200hz.csv

Differences vs record_joint_states.py, each closing a documented fault mode:

1. **No throttle — every received message becomes a CSV row.** The old recorder
   dropped messages arriving sooner than 1/rate after the last *written* row,
   which at publish-rate ≈ recorder-rate can halve the effective rate
   (COLLECTION_200HZ.md Step-4 caveat). Recording everything removes the caveat
   and the --rate flag entirely.
2. **`time` column from the message header stamp** (the driver's sample clock),
   not the subscriber's wall clock — removes executor/scheduling jitter from the
   time base used for differentiation. Falls back to wall clock (with a loud
   warning) if the driver does not stamp. Arrival wall time is kept in an extra
   trailing `recv_time` column for jitter diagnostics; all downstream loaders
   (sysid_feasible.load_and_filter) select columns by name, so the extra column
   is ignored.
3. **Sensor-data QoS** (BEST_EFFORT, KEEP_LAST depth 50) — a BEST_EFFORT
   subscription is compatible with both RELIABLE and BEST_EFFORT publishers,
   and depth 50 gives ≈250 ms of queue cushion at 200 Hz against transient
   subscriber stalls.
4. **Watchdogs** — aborts (exit 1) if no first message within
   --first-msg-timeout; warns on mid-run publisher stalls (> 5 s without a
   message); hard-stops at duration + 30 s wall time so it can never hang the
   orchestrating script.

CSV columns:
    time, waist_pos, waist_vel, waist_effort, ..., right_finger_effort, recv_time

Stops after --duration seconds of data, or immediately on Ctrl-C / SIGTERM.
Prints a summary (rows, rate, dropout-sentinel count) on exit.
"""
import argparse
import csv
import datetime
import math
import signal
import sys
import time
from pathlib import Path

import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
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
ARM_JOINTS = JOINT_ORDER[:6]

# Communication-dropout sentinel: sync-read failure reports −π on ALL arm
# joints at once (same definition as sysid_feasible.py --drop-glitches).
DROPOUT_TOL = 1e-3


class JointStateRecorder200(Node):
    """Subscribes to the joint-state topic and writes EVERY message to CSV."""

    def __init__(self, output_path: str, duration: float, topic: str) -> None:
        super().__init__('joint_state_recorder_200hz')

        self._duration = duration
        self._row_count = 0
        self._dropout_count = 0      # rows where all 6 arm joints ≈ −π
        self._nonmono_count = 0      # rows where the time column went backwards
        self._use_stamp: bool | None = None   # decided on the first message
        self._t0: float | None = None         # time-base origin (stamp or wall)
        self.t_first_wall: float | None = None  # wall clock of first message
        self.t_last_wall: float | None = None   # wall clock of latest message
        self._t_last_col: float | None = None   # last written `time` value
        self.done = False            # main loop polls this flag

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        # line-buffered so data reaches disk continuously (safer on Ctrl-C)
        self._csv_file = open(output_path, 'w', newline='', buffering=1)
        self._writer = self._init_csv()

        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.create_subscription(JointState, topic, self._callback, qos)
        self.get_logger().info(
            f'Recorder ready  →  {output_path}  '
            f'(duration={duration:.0f} s, no throttle — every message recorded)'
        )

    # ── CSV initialisation ─────────────────────────────────────────────────────

    def _init_csv(self) -> csv.writer:
        writer = csv.writer(self._csv_file)
        header = ['time']
        for j in JOINT_ORDER:
            header += [f'{j}_pos', f'{j}_vel', f'{j}_effort']
        header.append('recv_time')   # arrival wall time (diagnostics only)
        writer.writerow(header)
        return writer

    # ── ROS2 subscriber callback ───────────────────────────────────────────────

    def _callback(self, msg: JointState) -> None:
        if self.done:
            return

        wall = time.time()
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.t_first_wall is None:
            self.t_first_wall = wall
            # Decide the time base ONCE: header stamp if the driver stamps,
            # else wall clock (mixing bases mid-run would corrupt dt).
            self._use_stamp = stamp > 1.0
            self._t0 = stamp if self._use_stamp else wall
            base = ('header stamp' if self._use_stamp
                    else 'WALL CLOCK — driver does not stamp messages!')
            self.get_logger().info(
                f'First joint-state message received — recording started '
                f'(time base: {base}).'
            )
            if not self._use_stamp:
                self.get_logger().warning(
                    'Header stamps are zero; timing quality limited to '
                    'subscriber arrival times.'
                )
            print('RECORDING_STARTED', flush=True)   # sentinel for collect_200hz.sh

        self.t_last_wall = wall
        t_col = (stamp if self._use_stamp else wall) - self._t0

        if t_col >= self._duration:
            self.done = True
            self.get_logger().info(
                f'Duration {self._duration:.0f} s reached — recording stopped.'
            )
            return

        # Build name → index map for this message (order can vary)
        name_idx: dict[str, int] = {n: i for i, n in enumerate(msg.name)}

        def _get_f(seq, idx: int | None) -> float | None:
            """Return seq[idx] as float, or None if idx is None or out of range."""
            if idx is None or idx >= len(seq):
                return None
            return float(seq[idx])

        row: list[str] = [f'{t_col:.6f}']
        arm_pos: list[float | None] = []
        for joint in JOINT_ORDER:
            i = name_idx.get(joint)       # None if joint not in this message
            pos = _get_f(msg.position, i)
            vel = _get_f(msg.velocity, i)
            eff = _get_f(msg.effort, i)
            if joint in ARM_JOINTS:
                arm_pos.append(pos)
            row.append('' if pos is None else repr(pos))
            row.append('' if vel is None else repr(vel))
            row.append('' if eff is None else repr(eff))
        row.append(f'{wall - self.t_first_wall:.6f}')

        # Raw data is written as-is; these counters only flag problems live.
        if (all(p is not None for p in arm_pos)
                and all(abs(p + math.pi) < DROPOUT_TOL for p in arm_pos)):
            self._dropout_count += 1
        if self._t_last_col is not None and t_col < self._t_last_col:
            self._nonmono_count += 1
            if self._nonmono_count <= 3:
                self.get_logger().warning(
                    f'time column went backwards at row {self._row_count} '
                    f'({self._t_last_col:.6f} → {t_col:.6f})'
                )

        self._writer.writerow(row)
        self._row_count += 1
        self._t_last_col = t_col

    # ── Clean shutdown ─────────────────────────────────────────────────────────

    def close(self) -> None:
        """Flush and close the CSV; print summary."""
        self._csv_file.flush()
        self._csv_file.close()
        print(f'\n[recorder] Saved {self._row_count} rows.')
        if self._row_count > 1 and self._t_last_col:
            print(f'[recorder] Span {self._t_last_col:.1f} s  →  effective rate '
                  f'{(self._row_count - 1) / self._t_last_col:.1f} Hz')
        print(f'[recorder] Dropout-sentinel rows (all arm joints ≈ −π): '
              f'{self._dropout_count}')
        if self._nonmono_count:
            print(f'[recorder] WARNING: {self._nonmono_count} non-monotonic '
                  f'time steps — inspect with check_collection.py')


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description='VX300s joint-state CSV recorder, Tier 2 — records every '
                    'message, header-stamp time base',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--output', default='data/traj_run_200hz.csv',
        help='Output CSV file path',
    )
    parser.add_argument(
        '--duration', type=float, default=900.0,
        help='Recording duration in seconds (from the first message)',
    )
    parser.add_argument(
        '--topic', default='/vx300s/joint_states',
        help='JointState topic to record',
    )
    parser.add_argument(
        '--first-msg-timeout', type=float, default=20.0,
        help='Abort if no message arrives within this many seconds',
    )
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists():
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        output = output.with_stem(f'{output.stem}_{stamp}')
        print(f'[recorder] {args.output} already exists — writing to {output}')

    rclpy.init()
    recorder = JointStateRecorder200(str(output), args.duration, args.topic)

    # Handle Ctrl-C / SIGTERM gracefully: set the done flag, loop exits cleanly
    def _on_signal(sig, frame):  # noqa: ANN001
        recorder.get_logger().info(f'Signal {sig} received — stopping.')
        recorder.done = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(recorder)

    exit_code = 0
    t_loop_start = time.time()
    stall_warned = False
    try:
        # spin_once loop so we can check flags without a background thread
        while rclpy.ok() and not recorder.done:
            executor.spin_once(timeout_sec=0.05)
            now = time.time()

            if recorder.t_first_wall is None:
                if now - t_loop_start > args.first_msg_timeout:
                    recorder.get_logger().error(
                        f'No message within {args.first_msg_timeout:.0f} s — '
                        f'is the driver publishing on {args.topic}?'
                    )
                    exit_code = 1
                    break
                continue

            # Publisher-stall warning (once per stall episode)
            if recorder.t_last_wall is not None:
                gap = now - recorder.t_last_wall
                if gap > 5.0 and not stall_warned:
                    recorder.get_logger().warning(
                        f'No message for {gap:.1f} s — publisher stalled?'
                    )
                    stall_warned = True
                elif gap < 1.0:
                    stall_warned = False

            # Hard stop: never outlive the requested duration by more than 30 s
            # of wall time (covers a dead publisher near the end of the run).
            if now - recorder.t_first_wall > args.duration + 30.0:
                recorder.get_logger().warning(
                    'Wall-time limit (duration + 30 s) reached — stopping.'
                )
                recorder.done = True
    except Exception as exc:
        recorder.get_logger().error(f'Unexpected error: {exc}')
        exit_code = 1
    finally:
        recorder.close()
        recorder.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(exit_code)


if __name__ == '__main__':
    main()
