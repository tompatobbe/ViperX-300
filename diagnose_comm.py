#!/usr/bin/env python3
"""
Diagnostic script for vx300s Dynamixel communication errors.

Checks port access, scans for servos at expected and fallback baud rates,
and reports which IDs respond vs. which are missing/misconfigured.

Expected servos (from vx300s.yaml):
  ID 1: waist
  ID 2: shoulder
  ID 3: shoulder_shadow
  ID 4: elbow
  ID 5: elbow_shadow
  ID 6: forearm_roll
  ID 7: wrist_angle
  ID 8: wrist_rotate
  ID 9: gripper
"""

import sys
import os
import stat
import grp
import time

try:
    from dynamixel_sdk import PortHandler, PacketHandler
except ImportError:
    print("[FATAL] dynamixel_sdk not found. Install with: pip3 install dynamixel-sdk")
    sys.exit(1)

PORT = "/dev/ttyUSB0"
PROTOCOL = 2.0

# Dynamixel baud rate register value → actual baud
BAUD_MAP = {
    0: 9600,
    1: 57600,
    2: 115200,
    3: 1_000_000,   # expected by vx300s.yaml
    4: 2_000_000,
    5: 3_000_000,
    6: 4_000_000,
}

EXPECTED_IDS = {
    1: "waist",
    2: "shoulder",
    3: "shoulder_shadow",
    4: "elbow",
    5: "elbow_shadow",
    6: "forearm_roll",
    7: "wrist_angle",
    8: "wrist_rotate",
    9: "gripper",
}

SCAN_IDS = list(range(1, 20))  # scan a wider range to catch rogue/misconfigured servos

# XM430 / XL430 control table addresses
ADDR_MODEL_NUMBER   = 0
ADDR_BAUD_RATE      = 8
ADDR_RETURN_DELAY   = 9
ADDR_OPERATING_MODE = 11
ADDR_TORQUE_ENABLE  = 64
ADDR_PRESENT_TEMP   = 146
ADDR_PRESENT_VOLT   = 144
ADDR_HW_ERROR       = 70


def check_port_permissions():
    print(f"\n{'='*60}")
    print(f"PORT CHECK: {PORT}")
    print(f"{'='*60}")

    if not os.path.exists(PORT):
        print(f"  [FAIL] {PORT} does not exist — is the USB cable plugged in?")
        return False

    s = os.stat(PORT)
    mode = s.st_mode
    gid = s.st_gid
    uid = s.st_uid

    try:
        group_name = grp.getgrgid(gid).gr_name
    except KeyError:
        group_name = str(gid)

    print(f"  Owner: uid={uid}, gid={gid} ({group_name})")
    print(f"  Mode:  {stat.filemode(mode)}")

    current_uid = os.getuid()
    current_groups = os.getgroups()

    readable = (
        (uid == current_uid and mode & stat.S_IRUSR)
        or (gid in current_groups and mode & stat.S_IRGRP)
        or (mode & stat.S_IROTH)
    )
    writable = (
        (uid == current_uid and mode & stat.S_IWUSR)
        or (gid in current_groups and mode & stat.S_IWGRP)
        or (mode & stat.S_IWOTH)
    )

    if readable and writable:
        print("  [OK] Current user has read+write access.")
    else:
        print(f"  [FAIL] No read/write access. Add yourself to the '{group_name}' group:")
        print(f"         sudo usermod -aG {group_name} $USER && newgrp {group_name}")
        return False
    return True


def ping_servo(port_handler, packet_handler, servo_id, timeout=0.05):
    """Return (model_number, error) or (None, None) if no response."""
    model, comm_result, dxl_error = packet_handler.ping(port_handler, servo_id)
    if comm_result == 0:  # COMM_SUCCESS
        return model, dxl_error
    return None, None


def read_byte(port_handler, packet_handler, servo_id, address):
    val, comm_result, _ = packet_handler.read1ByteTxRx(port_handler, servo_id, address)
    if comm_result == 0:
        return val
    return None


def read_word(port_handler, packet_handler, servo_id, address):
    val, comm_result, _ = packet_handler.read2ByteTxRx(port_handler, servo_id, address)
    if comm_result == 0:
        return val
    return None


def scan_at_baud(baud_rate):
    port_handler = PortHandler(PORT)
    packet_handler = PacketHandler(PROTOCOL)

    if not port_handler.openPort():
        print(f"  [FAIL] Could not open {PORT}")
        return {}

    if not port_handler.setBaudRate(baud_rate):
        print(f"  [FAIL] Could not set baud rate {baud_rate}")
        port_handler.closePort()
        return {}

    found = {}
    for sid in SCAN_IDS:
        model, err = ping_servo(port_handler, packet_handler, sid)
        if model is not None:
            found[sid] = {"model": model, "hw_error": err}

    port_handler.closePort()
    return found


def read_servo_details(baud_rate, servo_ids):
    port_handler = PortHandler(PORT)
    packet_handler = PacketHandler(PROTOCOL)

    if not port_handler.openPort() or not port_handler.setBaudRate(baud_rate):
        return {}

    details = {}
    for sid in servo_ids:
        temp  = read_byte(port_handler, packet_handler, sid, ADDR_PRESENT_TEMP)
        volt  = read_word(port_handler, packet_handler, sid, ADDR_PRESENT_VOLT)
        hwErr = read_byte(port_handler, packet_handler, sid, ADDR_HW_ERROR)
        baud  = read_byte(port_handler, packet_handler, sid, ADDR_BAUD_RATE)
        rdt   = read_byte(port_handler, packet_handler, sid, ADDR_RETURN_DELAY)
        torq  = read_byte(port_handler, packet_handler, sid, ADDR_TORQUE_ENABLE)
        details[sid] = {
            "temp_C":         temp,
            "voltage_mV":     volt * 100 if volt is not None else None,
            "hw_error_flags": hwErr,
            "baud_reg":       baud,
            "return_delay":   rdt,
            "torque_enabled": torq,
        }

    port_handler.closePort()
    return details


def decode_hw_error(flags):
    if flags is None:
        return "unreadable"
    if flags == 0:
        return "none"
    errors = []
    if flags & (1 << 0): errors.append("InputVoltage")
    if flags & (1 << 2): errors.append("Overheating")
    if flags & (1 << 3): errors.append("MotorEncoder")
    if flags & (1 << 4): errors.append("ElectricalShock")
    if flags & (1 << 5): errors.append("Overload")
    return ", ".join(errors) if errors else f"unknown(0x{flags:02X})"


def main():
    print("\nvx300s Dynamixel Communication Diagnostic")
    print("==========================================")

    if not check_port_permissions():
        sys.exit(1)

    expected_baud = 1_000_000  # Baud_Rate register 3
    print(f"\n{'='*60}")
    print(f"SCANNING at expected baud {expected_baud} bps (register value 3)")
    print(f"{'='*60}")

    found_expected = scan_at_baud(expected_baud)
    if found_expected:
        print(f"  Found {len(found_expected)} servo(s): IDs {sorted(found_expected.keys())}")
    else:
        print("  [WARN] No servos found at expected baud rate!")

    # Check for missing / extra servos
    expected_set = set(EXPECTED_IDS.keys())
    found_set    = set(found_expected.keys())
    missing      = expected_set - found_set
    unexpected   = found_set - expected_set

    if missing:
        print(f"\n  [FAIL] Missing servo IDs: {sorted(missing)}")
        for sid in sorted(missing):
            print(f"         ID {sid} = {EXPECTED_IDS[sid]}")
    if unexpected:
        print(f"\n  [WARN] Unexpected servo IDs found: {sorted(unexpected)}")

    # Per-servo detail for those that responded at expected baud
    if found_expected:
        details = read_servo_details(expected_baud, sorted(found_expected.keys()))
        print(f"\n{'='*60}")
        print("SERVO DETAILS (at 1 Mbps)")
        print(f"{'='*60}")
        print(f"  {'ID':<4} {'Name':<16} {'Model':<8} {'Temp':>6} {'Volt':>8} {'HW Error':<25} {'Baud reg':>8} {'Ret delay':>9} {'Torque':>6}")
        print(f"  {'-'*100}")
        for sid in sorted(found_expected.keys()):
            name   = EXPECTED_IDS.get(sid, "unknown")
            model  = found_expected[sid]["model"]
            d      = details.get(sid, {})
            temp   = f"{d.get('temp_C', '?')}°C"
            volt   = f"{d.get('voltage_mV', '?')} mV" if d.get('voltage_mV') is not None else "?"
            hwerr  = decode_hw_error(d.get('hw_error_flags'))
            baud_r = str(d.get('baud_reg', '?'))
            rdt    = str(d.get('return_delay', '?'))
            torq   = "ON" if d.get('torque_enabled') else "off"

            baud_ok = "  " if d.get('baud_reg') == 3 else "!!"
            rdt_ok  = "  " if d.get('return_delay') == 0 else "!!"

            print(f"  {sid:<4} {name:<16} {model:<8} {temp:>6} {volt:>8} {hwerr:<25} {baud_r:>6}{baud_ok} {rdt:>7}{rdt_ok} {torq:>6}")

    # Scan other baud rates for missing servos
    if missing:
        print(f"\n{'='*60}")
        print("SCANNING OTHER BAUD RATES for missing servos")
        print(f"{'='*60}")
        for reg_val, baud in BAUD_MAP.items():
            if baud == expected_baud:
                continue
            found_other = scan_at_baud(baud)
            recovered = set(found_other.keys()) & missing
            if recovered:
                print(f"\n  [FOUND at {baud} bps / reg={reg_val}]: IDs {sorted(recovered)}")
                for sid in sorted(recovered):
                    print(f"    ID {sid} = {EXPECTED_IDS.get(sid, 'unknown')} — baud register is WRONG (is {reg_val}, should be 3)")
                print(f"    Fix: set Baud_Rate register to 3 (1 Mbps) on these servos via Dynamixel Wizard")
        else:
            if not any(set(scan_at_baud(b).keys()) & missing for b in BAUD_MAP.values() if b != expected_baud):
                print("  No missing servos found at any standard baud rate.")
                print("  Possible causes for missing IDs:")
                print("    - Servo is unpowered (check 12V bus and cable daisy chain)")
                print("    - Servo hardware failure")
                print("    - ID mismatch (use Dynamixel Wizard to scan with one servo at a time)")

    # Sync-read specific analysis
    print(f"\n{'='*60}")
    print("SYNC-READ FAILURE ANALYSIS")
    print(f"{'='*60}")
    if missing:
        print(f"  Root cause: syncRead requires ALL listed servo IDs to respond.")
        print(f"  IDs {sorted(missing)} are not replying → 'Incorrect status packet' is expected.")
        print(f"\n  Steps to fix:")
        print(f"  1. Check power: confirm 12V is supplied to all servos in the chain.")
        print(f"  2. Check daisy-chain cable continuity between the non-responding servo")
        print(f"     and its neighbor.")
        print(f"  3. Connect only the suspect servo in isolation and scan with Dynamixel Wizard.")
        print(f"  4. If the baud rate scan above found it at a wrong baud rate, reflash it.")
    elif found_set == expected_set:
        print("  All 9 expected servos responded at 1 Mbps.")
        print("  The sync-read failure may be intermittent. Possible causes:")
        print("  - Loose cable connection (wiggle each cable connector while pinging)")
        print("  - Voltage drop under load: check that PSU delivers ≥ 12V at rated current")
        print("  - Return_Delay_Time != 0 on any servo (see '!!' flags above)")
        print("  - Another process is also opening the port (check: lsof /dev/ttyUSB0)")
    else:
        print("  Could not determine root cause — check details above.")

    # Check if another process holds the port
    print(f"\n{'='*60}")
    print("PORT USAGE CHECK")
    print(f"{'='*60}")
    os.system(f"lsof {PORT} 2>/dev/null || echo '  (lsof not available)'")


if __name__ == "__main__":
    main()
