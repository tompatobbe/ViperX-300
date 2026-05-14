#!/usr/bin/env bash
# Run recorder and excitation trajectory together.
# Usage: ./collect_sysid_pos.sh [--duration 90] [--rate 50] [--output data/sysid_run1.csv]

DURATION=60
RATE=50
OUTPUT="data/arm_data.csv"

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration) DURATION="$2"; shift 2 ;;
        --rate)     RATE="$2";     shift 2 ;;
        --output)   OUTPUT="$2";   shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

python3 record_joint_states.py --duration "$DURATION" --rate "$RATE" --output "$OUTPUT" &
RECORDER_PID=$!

sleep 1  # give the recorder time to initialise before sending commands

python3 run_sysid_trajectory.py --duration "$DURATION" --rate "$RATE"

kill "$RECORDER_PID" 2>/dev/null
wait "$RECORDER_PID" 2>/dev/null
echo "Done. Data saved to $OUTPUT"
