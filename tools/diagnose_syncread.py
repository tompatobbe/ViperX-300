#!/usr/bin/env python3
"""
Reproduces the xs_sdk syncRead pattern to confirm WSL2 USB/IP jitter is the cause.

The xs_sdk reads Present_Current (126, 2B) + Present_Velocity (128, 4B) +
Present_Position (132, 4B) from all 9 servo IDs in a single GroupSyncRead.
This script does exactly that and measures packet timing to expose WSL2 jitter.

Usage:
    python3 diagnose_syncread.py [--port /dev/ttyUSB0] [--reps 500]
"""

import sys
import time
import argparse
import statistics

try:
    from dynamixel_sdk import (
        PortHandler, PacketHandler, GroupSyncRead, COMM_SUCCESS
    )
except ImportError:
    print("[FATAL] dynamixel_sdk not found.")
    sys.exit(1)

PROTOCOL  = 2.0
BAUD      = 1_000_000

SERVO_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SERVO_NAMES = {
    1:"waist", 2:"shoulder", 3:"shld_shad", 4:"elbow", 5:"elbw_shad",
    6:"frm_roll", 7:"wrist_ang", 8:"wrist_rot", 9:"gripper",
}

# xs_sdk GroupSyncRead parameters (from interbotix xs_sdk source)
ADDR_PRESENT_CURRENT  = 126  # 2 bytes
ADDR_PRESENT_VELOCITY = 128  # 4 bytes
ADDR_PRESENT_POSITION = 132  # 4 bytes
# xs_sdk reads a contiguous block starting at 126, length 10
SYNC_START_ADDR = 126
SYNC_DATA_LEN   = 10   # covers current(2) + velocity(4) + position(4)


def resolve_usb_sys_path(dev):
    """Resolve the sysfs device path via the usb-serial symlink (no recursive glob)."""
    import os
    symlink = f"/sys/bus/usb-serial/devices/{dev}"
    if os.path.exists(symlink):
        try:
            return os.path.realpath(symlink)
        except OSError:
            pass
    return None


def print_usb_diagnosis(port):
    import os
    dev = port.split("/")[-1]
    print(f"\n{'='*60}")
    print("USB / WSL2 ENVIRONMENT CHECK")
    print(f"{'='*60}")

    sys_path = resolve_usb_sys_path(dev)

    # latency timer — look relative to the resolved sysfs path
    lt_path = None
    if sys_path and os.path.exists(os.path.join(sys_path, "latency_timer")):
        lt_path = os.path.join(sys_path, "latency_timer")
    else:
        candidate = f"/sys/bus/usb-serial/devices/{dev}/latency_timer"
        if os.path.exists(candidate):
            lt_path = candidate

    if lt_path:
        lt = open(lt_path).read().strip()
        status = "[OK]" if lt == "1" else "[WARN]"
        print(f"  {status} FTDI latency timer: {lt} ms  (want 1)")
        if lt != "1":
            print(f"  Fix: echo 1 | sudo tee {lt_path}")
    else:
        print("  [WARN] Could not find latency_timer sysfs entry")

    if sys_path and "vhci_hcd" in sys_path:
        print(f"\n  [!!] WSL2 USB/IP detected (vhci_hcd in device path):")
        print(f"       {sys_path}")
        print(f"\n  This is the likely root cause of syncRead failures.")
        print(f"  USB/IP adds 1–20ms of non-deterministic jitter to each USB")
        print(f"  transaction. At 1 Mbps, the 9-servo syncRead packet window is")
        print(f"  ~1 ms — smaller than the jitter, so packets arrive corrupt.")
        print(f"\n  Workarounds (in order of effectiveness):")
        print(f"    1. [Best]    Boot native Linux (dual-boot or live USB)")
        print(f"    2. [Good]    Lower baud rate to 57600 or 115200 (reconfigure servos)")
        print(f"    3. [Partial] Pin WSL2 to a fixed CPU and set process priority:")
        print(f"                   sudo chrt -f 99 ros2 launch ...")
        print(f"    4. [Partial] Patch xs_sdk to fall back to individual reads on syncRead error")
    elif sys_path:
        print(f"  USB path: {sys_path}")
        print(f"  [OK] No WSL2 USB/IP detected — jitter is from another source")


def run_syncread_test(port, reps):
    port_handler   = PortHandler(port)
    packet_handler = PacketHandler(PROTOCOL)

    if not port_handler.openPort():
        print(f"[FATAL] Cannot open {port}")
        sys.exit(1)
    port_handler.setBaudRate(BAUD)

    groupSyncRead = GroupSyncRead(port_handler, packet_handler, SYNC_START_ADDR, SYNC_DATA_LEN)
    for sid in SERVO_IDS:
        groupSyncRead.addParam(sid)

    print(f"\n{'='*60}")
    print(f"SYNCREAD TEST — {reps} reps, mimicking xs_sdk pattern")
    print(f"{'='*60}")
    print(f"  Reading IDs {SERVO_IDS} from addr {SYNC_START_ADDR} len {SYNC_DATA_LEN}")
    print(f"  (Present_Current + Present_Velocity + Present_Position)\n")

    ok_count   = 0
    err_count  = 0
    latencies  = []
    err_pattern = {}    # sid -> error count when getdata fails
    first_errors = []

    for i in range(reps):
        t0 = time.perf_counter()
        result = groupSyncRead.txRxPacket()
        t1 = time.perf_counter()
        rtt_ms = (t1 - t0) * 1000
        latencies.append(rtt_ms)

        if result != COMM_SUCCESS:
            err_count += 1
            reason = packet_handler.getTxRxResult(result)
            if len(first_errors) < 5:
                first_errors.append((i, rtt_ms, reason))
            # probe which IDs are unreadable
            for sid in SERVO_IDS:
                if not groupSyncRead.isAvailable(sid, ADDR_PRESENT_POSITION, 4):
                    err_pattern[sid] = err_pattern.get(sid, 0) + 1
        else:
            ok_count += 1

        # Print a dot every 50 reps
        if (i + 1) % 50 == 0:
            print(f"  {i+1:4d}/{reps}  errors so far: {err_count}  "
                  f"avg RTT: {statistics.mean(latencies[-50:]):.1f} ms  "
                  f"max RTT: {max(latencies[-50:]):.1f} ms")

        # Small delay to match xs_sdk 50Hz polling (20ms period)
        elapsed = time.perf_counter() - t0
        sleep_s = max(0, 0.020 - elapsed)
        time.sleep(sleep_s)

    port_handler.closePort()

    print(f"\n{'='*60}")
    print("SYNCREAD RESULTS")
    print(f"{'='*60}")
    print(f"  Reps:    {reps}")
    print(f"  OK:      {ok_count}  ({100*ok_count/reps:.1f}%)")
    print(f"  Errors:  {err_count}  ({100*err_count/reps:.1f}%)")

    if latencies:
        print(f"\n  RTT stats (ms):")
        print(f"    mean:   {statistics.mean(latencies):.2f}")
        print(f"    median: {statistics.median(latencies):.2f}")
        print(f"    stdev:  {statistics.stdev(latencies):.2f}" if len(latencies) > 1 else "    stdev:  n/a")
        print(f"    min:    {min(latencies):.2f}")
        print(f"    max:    {max(latencies):.2f}")
        # count spikes > 5ms (jitter threshold for 1Mbps syncRead)
        spikes = [l for l in latencies if l > 5]
        print(f"    spikes >5ms: {len(spikes)} ({100*len(spikes)/len(latencies):.1f}%)")

    if first_errors:
        print(f"\n  First {len(first_errors)} error(s):")
        for rep, rtt, reason in first_errors:
            print(f"    rep {rep:4d}: {reason}  (RTT {rtt:.1f} ms)")

    if err_pattern:
        print(f"\n  IDs that failed to provide data after comm errors:")
        for sid, cnt in sorted(err_pattern.items(), key=lambda x: -x[1]):
            print(f"    ID {sid} ({SERVO_NAMES[sid]}): {cnt} times")

    # Diagnosis
    print(f"\n{'='*60}")
    print("DIAGNOSIS")
    print(f"{'='*60}")
    if err_count == 0:
        print("  No syncRead errors in this run.")
        print("  The failure may be load-triggered (arm moving under torque).")
        print("  Try running with --reps 2000 while physically moving the arm.")
    else:
        spike_pct = 100 * len([l for l in latencies if l > 5]) / len(latencies)

        # Check if all IDs failed equally (USB packet corruption) vs one ID failing more
        if err_pattern:
            counts = list(err_pattern.values())
            all_equal = len(set(counts)) == 1 and set(err_pattern.keys()) == set(SERVO_IDS)

        if all_equal:
            print(f"  All 9 IDs failed equally — this is whole-packet USB corruption,")
            print(f"  NOT a specific servo hardware problem.")
            print(f"  When a USB jitter spike delays the read window, the SDK receives")
            print(f"  a malformed packet and cannot extract data for any ID.")
        elif err_pattern:
            worst = max(err_pattern, key=err_pattern.get)
            print(f"  ID {worst} ({SERVO_NAMES[worst]}) failed most often.")
            print(f"  This servo may have a hardware or cable issue in addition to USB jitter.")

        max_rtt = max(latencies)
        print(f"\n  Triggering spike: {max_rtt:.1f} ms RTT (normal is ~{statistics.median(latencies):.1f} ms)")
        print(f"  At 1 Mbps, the 9 servo responses arrive within ~1 ms of each other.")
        print(f"  A {max_rtt:.0f} ms USB/IP jitter spike delayed the read and caused packet corruption.")
        print(f"\n  Error rate: {100*err_count/reps:.1f}% — rare but the xs_sdk has no retry logic,")
        print(f"  so each occurrence prints 3 error lines in the ROS log.")
        print(f"\n  Fixes:")
        print(f"    1. [Best]    Boot native Linux — eliminates USB/IP jitter entirely")
        print(f"    2. [Easy]    Set real-time scheduling:  sudo chrt -f 99 ros2 launch ...")
        print(f"    3. [Medium]  Patch xs_sdk to retry syncRead on failure (see below)")
        print(f"\n  xs_sdk retry patch location:")
        print(f"    ~/interbotix_ws/src/interbotix_ros_core/interbotix_ros_xseries/")
        print(f"    interbotix_xs_driver/src/xs_driver.cpp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",  default="/dev/ttyUSB0")
    parser.add_argument("--reps",  type=int, default=500,
                        help="Number of syncRead cycles (default 500 ≈ 10 s at 50Hz)")
    args = parser.parse_args()

    print_usb_diagnosis(args.port)
    run_syncread_test(args.port, args.reps)


if __name__ == "__main__":
    main()
