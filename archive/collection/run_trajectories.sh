#!/usr/bin/env bash
# Run the paper excitation trajectory (Eq. 7) and recorder together.
#
# Minimal run (all defaults — 900 s, 200 Hz, scipy-optimised coefficients):
#   bash run_trajectories.sh
#
# Short test run with random coefficients:
#   bash run_trajectories.sh --duration 60 --no-optimize
#
# Full example:
#   bash run_trajectories.sh --duration 900 --rate 200 --output data/traj_run.csv \
#       --robot-model vx300s --seed 42 --stride 4

DURATION=900
RATE=200
OUTPUT="data/traj_run.csv"
ROBOT_MODEL="vx300s"
SEED=42
STRIDE=4
NO_OPTIMIZE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --duration)    DURATION="$2";    shift 2 ;;
        --rate)        RATE="$2";        shift 2 ;;
        --output)      OUTPUT="$2";      shift 2 ;;
        --robot-model) ROBOT_MODEL="$2"; shift 2 ;;
        --seed)        SEED="$2";        shift 2 ;;
        --stride)      STRIDE="$2";      shift 2 ;;
        --no-optimize) NO_OPTIMIZE="--no-optimize"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

mkdir -p "$(dirname "$OUTPUT")"

python3 record_joint_states.py \
    --duration "$DURATION" \
    --rate     "$RATE"     \
    --output   "$OUTPUT"   &
RECORDER_PID=$!

sleep 1  # give the recorder time to initialise before sending commands

python3 run_trajectories.py \
    --duration    "$DURATION"    \
    --rate        "$RATE"        \
    --robot-model "$ROBOT_MODEL" \
    --seed        "$SEED"        \
    --stride      "$STRIDE"      \
    $NO_OPTIMIZE

kill "$RECORDER_PID" 2>/dev/null
wait "$RECORDER_PID" 2>/dev/null
echo "Done. Data saved to $OUTPUT"
