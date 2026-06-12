#!/usr/bin/env python3
"""Live supply-voltage watcher (diagnostic, 2026-06-12).

Polls Present_Input_Voltage on the first (waist) and last (wrist_rotate)
motors of the power daisy-chain via /vx300s/get_motor_registers while a
trajectory runs, to catch the supply dips suspected of rebooting the
downstream motors (see CHANGELOG 2026-06-12). Each register read is one
extra transaction on the Dynamixel bus (~8/s total — negligible next to
the 200 Hz sync reads).

A slow or failed service call is evidence too: it means the bus itself was
disturbed at that moment, so failures are logged rather than retried.

Usage:  python3 volt_watch.py [duration_seconds]   (default 600)
"""
import sys
import time

import rclpy
from interbotix_xs_msgs.srv import RegisterValues
from rclpy.node import Node


def main():
    duration = float(sys.argv[1]) if len(sys.argv) > 1 else 600.0
    rclpy.init()
    node = Node('volt_watch')
    cli = node.create_client(RegisterValues, '/vx300s/get_motor_registers')
    if not cli.wait_for_service(timeout_sec=5.0):
        print('[volt_watch] service unavailable — is the driver running?', flush=True)
        return
    print(f'[volt_watch] polling waist + wrist_rotate for {duration:.0f} s …', flush=True)

    motors = ['waist', 'wrist_rotate']
    t0 = time.monotonic()
    i = 0
    while time.monotonic() - t0 < duration:
        m = motors[i % 2]
        i += 1
        req = RegisterValues.Request(
            cmd_type='single', name=m, reg='Present_Input_Voltage')
        t1 = time.monotonic()
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(node, fut, timeout_sec=1.0)
        lat = time.monotonic() - t1
        ts = time.monotonic() - t0
        if fut.done() and fut.result() is not None and fut.result().values:
            v = fut.result().values[0] / 10.0
            dip = '  <<< DIP' if v < 11.5 else ''
            slow = f'  (slow: {lat * 1e3:.0f} ms)' if lat > 0.15 else ''
            print(f'{ts:8.2f}s  {m:13s} {v:4.1f} V{slow}{dip}', flush=True)
        else:
            print(f'{ts:8.2f}s  {m:13s} CALL FAILED/TIMEOUT '
                  f'after {lat * 1e3:.0f} ms  <<<', flush=True)
        time.sleep(max(0.0, 0.12 - (time.monotonic() - t1)))

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
