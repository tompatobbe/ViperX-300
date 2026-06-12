#!/usr/bin/env bash
# Run recorder and FFS excitation trajectory together.
#
# Minimal run (all defaults):
#   bash collect_sysid_pos.sh
#
# Full example:
#   bash collect_sysid_pos.sh --duration 180 --rate 50 --output data/sysid_run.csv \
#       --robot-model vx300s --seed 42 --stride 25 \
#       --move-speed 1.5 --accel-time 0.30 --settle-time 0.0

DURATION=180
RATE=50
OUTPUT="data/sysid_run1.csv"
ROBOT_MODEL="vx300s"
SEED=42
STRIDE=25
MOVE_SPEED=1.5
ACCEL_TIME=0.30
SETTLE_TIME=0.0

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)    DURATION="$2";    shift 2 ;;
        --rate)        RATE="$2";        shift 2 ;;
        --output)      OUTPUT="$2";      shift 2 ;;
        --robot-model) ROBOT_MODEL="$2"; shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        --stride)      STRIDE="$2";      shift 2 ;;
        --move-speed)  MOVE_SPEED="$2";  shift 2 ;;
        --accel-time)  ACCEL_TIME="$2";  shift 2 ;;
        --settle-time) SETTLE_TIME="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

python3 record_joint_states.py --duration "$DURATION" --rate "$RATE" --output "$OUTPUT" &
RECORDER_PID=$!

sleep 1  # give the recorder time to initialise before sending commands

python3 run_sysid_pos_paper.py \
    --duration    "$DURATION"    \
    --rate        "$RATE"        \
    --robot-model "$ROBOT_MODEL" \
    --seed        "$SEED"        \
    --stride      "$STRIDE"      \
    --move-speed  "$MOVE_SPEED"  \
    --accel-time  "$ACCEL_TIME"  \
    --settle-time "$SETTLE_TIME"

kill "$RECORDER_PID" 2>/dev/null
wait "$RECORDER_PID" 2>/dev/null
echo "Done. Data saved to $OUTPUT"
