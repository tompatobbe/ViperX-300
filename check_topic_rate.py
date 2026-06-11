#!/usr/bin/env python3
"""
Preflight GATE for the 200 Hz collection — docs/COLLECTION_200HZ.md, Step 3.

Measures the actual publish rate of the joint-state topic for --duration
seconds and exits 0 only if it meets --min-rate. A scriptable replacement for
eyeballing `ros2 topic hz`, used by collect_200hz.sh to refuse a long run on a
slow bus (the ~47 Hz FTDI latency_timer signature).

Also sanity-checks the header stamps (present? monotonic?), because
record_joint_states_200hz.py uses them as the time base.

Usage:
    python3 check_topic_rate.py --min-rate 150
    python3 check_topic_rate.py --duration 10 --topic /vx300s/joint_states

Exit codes: 0 = rate OK · 1 = rate below --min-rate · 2 = no messages at all.
"""
import argparse
import sys
import time

import numpy as np
import rclpy
import rclpy.executors
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import JointState

FIRST_MSG_WAIT = 10.0   # seconds to wait for the first message before giving up


class RateProbe(Node):
    """Collects arrival times and header stamps for a fixed window."""

    def __init__(self, topic: str) -> None:
        super().__init__('topic_rate_probe')
        self.arrivals: list[float] = []
        self.stamps: list[float] = []
        qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=50,
        )
        self.create_subscription(JointState, topic, self._callback, qos)

    def _callback(self, msg: JointState) -> None:
        self.arrivals.append(time.time())
        self.stamps.append(msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Measure a JointState topic rate and gate on a minimum',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--topic', default='/vx300s/joint_states')
    parser.add_argument('--duration', type=float, default=5.0,
                        help='Measurement window in seconds (after first message)')
    parser.add_argument('--min-rate', type=float, default=150.0,
                        help='Exit 1 if the measured rate is below this [Hz]')
    args = parser.parse_args()

    rclpy.init()
    probe = RateProbe(args.topic)
    executor = rclpy.executors.SingleThreadedExecutor()
    executor.add_node(probe)

    print(f'[gate] Listening on {args.topic} for {args.duration:.0f} s …')
    t_start = time.time()
    try:
        while rclpy.ok():
            executor.spin_once(timeout_sec=0.05)
            now = time.time()
            if not probe.arrivals:
                if now - t_start > FIRST_MSG_WAIT:
                    break
            elif now - probe.arrivals[0] >= args.duration:
                break
    except KeyboardInterrupt:
        pass
    finally:
        probe.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    n = len(probe.arrivals)
    if n == 0:
        print(f'[gate] FAIL — no messages within {FIRST_MSG_WAIT:.0f} s. '
              f'Is the driver running?')
        print('[gate]   ros2 launch interbotix_xsarm_control '
              'xsarm_control.launch.py robot_model:=vx300s')
        sys.exit(2)
    if n < 2:
        print('[gate] FAIL — only one message received; cannot measure a rate.')
        sys.exit(1)

    arr = np.asarray(probe.arrivals)
    stp = np.asarray(probe.stamps)
    span = arr[-1] - arr[0]
    rate = (n - 1) / span
    dt = np.diff(arr)

    print(f'[gate] {n} messages in {span:.2f} s  →  {rate:.1f} Hz '
          f'(arrival dt: median {np.median(dt)*1e3:.2f} ms, '
          f'p95 {np.percentile(dt, 95)*1e3:.2f} ms, max {dt.max()*1e3:.1f} ms)')

    # Header-stamp quality (informational — the recorder falls back to wall
    # clock automatically if stamps are absent)
    if np.all(stp < 1.0):
        print('[gate] NOTE: header stamps are zero — recorder will use the '
              'wall-clock fallback.')
    else:
        sdt = np.diff(stp)
        backwards = int((sdt < 0).sum())
        print(f'[gate] header-stamp dt: median {np.median(sdt)*1e3:.2f} ms, '
              f'max {sdt.max()*1e3:.1f} ms, non-monotonic steps: {backwards}')

    if rate < args.min_rate:
        print(f'[gate] FAIL — {rate:.1f} Hz < required {args.min_rate:.0f} Hz.')
        print('[gate] Likely cause: FTDI latency_timer still 16 ms '
              '(see docs/COLLECTION_200HZ.md Step 1).')
        sys.exit(1)

    print(f'[gate] PASS — {rate:.1f} Hz ≥ {args.min_rate:.0f} Hz.')
    sys.exit(0)


if __name__ == '__main__':
    main()
