#!/usr/bin/env python3
"""
Live servo monitor for vx300s intermittent syncRead failures.

Polls each servo individually (bypassing syncRead) to identify which ID is
causing "Incorrect status packet" errors and why. Logs temperature, voltage,
hardware error flags, and comm errors over time.

Run WHILE the arm is operating (or reproducing the failure condition).

Usage:
    python3 monitor_servos.py [--port /dev/ttyUSB0] [--interval 0.2]
"""

import sys
import time
import argparse
import signal
from collections import defaultdict
from datetime import datetime

try:
    from dynamixel_sdk import PortHandler, PacketHandler, COMM_SUCCESS
except ImportError:
    print("[FATAL] dynamixel_sdk not found.")
    sys.exit(1)

SERVO_NAMES = {
    1: "waist",
    2: "shoulder",
    3: "shld_shad",
    4: "elbow",
    5: "elbw_shad",
    6: "frm_roll",
    7: "wrist_ang",
    8: "wrist_rot",
    9: "gripper",
}

# XM430/XM540 control table
ADDR_HW_ERROR       = 70   # 1 byte
ADDR_PRESENT_VOLT   = 144  # 2 bytes  (unit: 0.1 V)
ADDR_PRESENT_TEMP   = 146  # 1 byte   (°C)
ADDR_PRESENT_CURR   = 126  # 2 bytes  (unit: 2.69 mA, signed)
ADDR_PRESENT_VEL    = 128  # 4 bytes
ADDR_PRESENT_POS    = 132  # 4 bytes

HW_ERROR_LABELS = {
    0: "InputVoltage",
    2: "Overheating",
    3: "MotorEncoder",
    4: "ElecShock",
    5: "Overload",
}

PROTOCOL = 2.0
WARNING_TEMP_C   = 65     # XM series rated max ~80°C, warn early
WARNING_VOLT_MV  = 11000  # warn below 11V (nominal 12V)
CRITICAL_VOLT_MV = 10000


def decode_hw_error(flags):
    if flags == 0:
        return ""
    return "|".join(label for bit, label in HW_ERROR_LABELS.items() if flags & (1 << bit))


class ServoMonitor:
    def __init__(self, port, baud=1_000_000):
        self.port_handler   = PortHandler(port)
        self.packet_handler = PacketHandler(PROTOCOL)
        self.stats = defaultdict(lambda: {"ok": 0, "err": 0, "corrupt": 0})
        self.events = []
        self.start_time = time.time()

        if not self.port_handler.openPort():
            raise RuntimeError(f"Cannot open {port}")
        if not self.port_handler.setBaudRate(baud):
            raise RuntimeError(f"Cannot set baud rate {baud}")

    def close(self):
        self.port_handler.closePort()

    def _read1(self, sid, addr):
        val, result, _ = self.packet_handler.read1ByteTxRx(self.port_handler, sid, addr)
        return val, result

    def _read2(self, sid, addr):
        val, result, _ = self.packet_handler.read2ByteTxRx(self.port_handler, sid, addr)
        return val, result

    def _read4(self, sid, addr):
        val, result, _ = self.packet_handler.read4ByteTxRx(self.port_handler, sid, addr)
        return val, result

    def poll_servo(self, sid):
        """Read all interesting registers for one servo. Returns dict of values + comm status."""
        results = {}

        hw_err, r = self._read1(sid, ADDR_HW_ERROR)
        results["hw_error"]  = hw_err if r == COMM_SUCCESS else None
        results["hw_err_ok"] = r == COMM_SUCCESS

        temp, r = self._read1(sid, ADDR_PRESENT_TEMP)
        results["temp_C"]    = temp if r == COMM_SUCCESS else None

        volt, r = self._read2(sid, ADDR_PRESENT_VOLT)
        results["volt_mV"]   = volt * 100 if r == COMM_SUCCESS else None

        curr, r = self._read2(sid, ADDR_PRESENT_CURR)
        if r == COMM_SUCCESS:
            # signed 16-bit, unit 2.69 mA
            if curr > 32767:
                curr -= 65536
            results["curr_mA"] = curr * 2.69
        else:
            results["curr_mA"] = None

        pos, r = self._read4(sid, ADDR_PRESENT_POS)
        results["pos"]       = pos if r == COMM_SUCCESS else None

        # Track comm quality
        if results["hw_err_ok"]:
            self.stats[sid]["ok"] += 1
        else:
            self.stats[sid]["err"] += 1

        return results

    def log_event(self, sid, tag, detail):
        elapsed = time.time() - self.start_time
        ts = f"+{elapsed:7.1f}s"
        name = SERVO_NAMES.get(sid, f"id{sid}")
        msg = f"[{ts}] ID {sid:2d} ({name:<9}) {tag}: {detail}"
        self.events.append(msg)
        return msg

    def print_header(self):
        names = "  ".join(f"{SERVO_NAMES[i]:>9}" for i in sorted(SERVO_NAMES))
        ids   = "  ".join(f"ID{i:>7}" for i in sorted(SERVO_NAMES))
        print(f"\n{'─'*120}")
        print(f"  {'Elapsed':>8}  {ids}")
        print(f"  {'':>8}  {names}")
        print(f"  {'':>8}  {'  '.join(['Tmp°C Volt  HWErr  CommErr'] * 1)}")
        print(f"{'─'*120}")

    def run(self, interval, max_duration=None):
        running = True

        def on_signal(sig, frame):
            nonlocal running
            running = False

        signal.signal(signal.SIGINT, on_signal)
        signal.signal(signal.SIGTERM, on_signal)

        print(f"\nvx300s Live Servo Monitor — polling every {interval:.2f}s")
        print(f"Press Ctrl+C to stop and see summary.\n")

        prev = {}
        poll_count = 0
        last_header = 0

        while running:
            if max_duration and (time.time() - self.start_time) > max_duration:
                break

            if poll_count - last_header >= 30:
                self.print_header()
                last_header = poll_count

            elapsed = time.time() - self.start_time
            row_parts = [f"  {elapsed:>7.1f}s  "]

            anomaly_lines = []

            for sid in sorted(SERVO_NAMES):
                data = self.poll_servo(sid)
                name = SERVO_NAMES[sid]

                temp   = data["temp_C"]
                volt   = data["volt_mV"]
                hw     = data["hw_error"]
                curr   = data["curr_mA"]
                comm_ok = data["hw_err_ok"]

                # Detect changes/anomalies
                prev_hw = prev.get(sid, {}).get("hw_error", 0)
                if hw is not None and hw != 0 and hw != prev_hw:
                    msg = self.log_event(sid, "HW_ERROR", decode_hw_error(hw))
                    anomaly_lines.append(f"  !! {msg}")

                if temp is not None and temp >= WARNING_TEMP_C:
                    label = "CRITICAL" if temp >= 75 else "WARN"
                    if prev.get(sid, {}).get("temp_C", 0) < WARNING_TEMP_C:
                        msg = self.log_event(sid, f"TEMP_{label}", f"{temp}°C")
                        anomaly_lines.append(f"  !! {msg}")

                if volt is not None and volt < WARNING_VOLT_MV:
                    label = "CRITICAL" if volt < CRITICAL_VOLT_MV else "WARN"
                    if prev.get(sid, {}).get("volt_mV", 99999) >= WARNING_VOLT_MV:
                        msg = self.log_event(sid, f"VOLT_{label}", f"{volt/1000:.2f}V")
                        anomaly_lines.append(f"  !! {msg}")

                if not comm_ok:
                    msg = self.log_event(sid, "COMM_FAIL", "read failed")
                    anomaly_lines.append(f"  !! {msg}")

                # Build row cell
                t_str   = f"{temp:3d}°C" if temp is not None else " ???  "
                v_str   = f"{volt/1000:.2f}V" if volt is not None else "  ??.??V"
                hw_str  = decode_hw_error(hw) if hw is not None else "?"
                if hw_str == "":
                    hw_str = "ok"
                err_str = f"{self.stats[sid]['err']:>3}err"

                cell = f"{t_str} {v_str} {hw_str:<6} {err_str}"
                row_parts.append(f"{cell:<28}")

                prev[sid] = data

            print("".join(row_parts))

            for line in anomaly_lines:
                print(line)

            poll_count += 1
            time.sleep(interval)

        self.print_summary()

    def print_summary(self):
        print(f"\n{'='*80}")
        print("SUMMARY")
        print(f"{'='*80}")
        print(f"\nComm stats (polls with successful vs. failed reads):")
        print(f"  {'ID':<4} {'Name':<12} {'OK':>6} {'FAIL':>6} {'Fail%':>7}")
        print(f"  {'─'*40}")
        for sid in sorted(SERVO_NAMES):
            s    = self.stats[sid]
            ok   = s["ok"]
            err  = s["err"]
            total = ok + err
            pct  = 100 * err / total if total else 0
            flag = " <-- most failures" if err == max(self.stats[s2]["err"] for s2 in self.stats) and err > 0 else ""
            print(f"  {sid:<4} {SERVO_NAMES[sid]:<12} {ok:>6} {err:>6} {pct:>6.1f}%{flag}")

        if self.events:
            print(f"\nEvent log ({len(self.events)} events):")
            for e in self.events:
                print(f"  {e}")
        else:
            print("\nNo anomalies detected during monitoring.")

        print(f"\nDiagnosis hints:")
        worst_sid = max(self.stats, key=lambda s: self.stats[s]["err"]) if self.stats else None
        if worst_sid and self.stats[worst_sid]["err"] > 0:
            total = self.stats[worst_sid]["ok"] + self.stats[worst_sid]["err"]
            pct = 100 * self.stats[worst_sid]["err"] / total
            print(f"  Most failures: ID {worst_sid} ({SERVO_NAMES[worst_sid]}) — {pct:.1f}% error rate")
            if pct > 20:
                print("  → High error rate: likely a hardware fault (cable, connector, or servo board)")
            elif pct > 2:
                print("  → Intermittent errors: check cable seating and power supply voltage under load")
            else:
                print("  → Low error rate: may be electrical noise; check USB isolator or shielded cable")

        hw_events = [e for e in self.events if "HW_ERROR" in e]
        if hw_events:
            print(f"  Hardware error flags were set — the servo has a fault condition:")
            for e in hw_events:
                print(f"    {e}")
            print("  → Clear with: write 0 to EEPROM Register 70, or power-cycle the servo")

        temp_events = [e for e in self.events if "TEMP" in e]
        if temp_events:
            print("  Temperature warnings detected — servo may be entering thermal protection")

        volt_events = [e for e in self.events if "VOLT" in e]
        if volt_events:
            print("  Voltage warnings detected — PSU may be sagging under load")
            print("  → Measure 12V rail with a multimeter while the arm moves")


def main():
    parser = argparse.ArgumentParser(description="vx300s live servo monitor")
    parser.add_argument("--port",     default="/dev/ttyUSB0")
    parser.add_argument("--interval", type=float, default=0.2,
                        help="Poll interval in seconds (default 0.2)")
    parser.add_argument("--duration", type=float, default=None,
                        help="Stop after N seconds (default: run until Ctrl+C)")
    args = parser.parse_args()

    try:
        monitor = ServoMonitor(args.port)
    except RuntimeError as e:
        print(f"[FATAL] {e}")
        sys.exit(1)

    try:
        monitor.run(interval=args.interval, max_duration=args.duration)
    finally:
        monitor.close()


if __name__ == "__main__":
    main()
