#!/usr/bin/env bash
# Standstill-stiction hysteresis collection — recorder + bidirectional mover.
# Same proven pattern as collect_static_gravity.sh: the 200 Hz recorder
# (record_joint_states_200hz.py) captures joint states while
# control/stiction_hysteresis_poses.py holds each target pose from BOTH approach
# directions (ascending / descending). Analyse offline (no ROS for --phi/paper):
#   python3 tools/analyze_stiction_hysteresis.py --csv <out> \
#       --phi outputs/npy/traj_run_200hz_20260612_131613__sysid_feasible-v1-5__cfg-a92e984c.npy
#
# Prerequisite (own terminal): the arm driver must be running:
#   ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s
#
#   bash collect_stiction_hysteresis.sh                 # full run
#   bash collect_stiction_hysteresis.sh --dwell 8 --delta 0.25
#
# Options:
#   --dwell S         seconds held at each pose (default 6)
#   --move-time S     seconds per move (default 4)
#   --delta R         approach offset on the swept joints [rad] (default 0.20)
#   --output PATH     CSV path (default data/stiction_hyst_<stamp>.csv)
#   --robot-model M   robot model (default vx300s)

set -euo pipefail
cd "$(dirname "$0")"

DWELL=6
MOVE_TIME=4
DELTA=0.20
OUTPUT=""
ROBOT_MODEL="vx300s"

while [[ $# -gt 0 ]]; do
    case $1 in
        --dwell)       DWELL="$2";       shift 2 ;;
        --move-time)   MOVE_TIME="$2";   shift 2 ;;
        --delta)       DELTA="$2";       shift 2 ;;
        --output)      OUTPUT="$2";      shift 2 ;;
        --robot-model) ROBOT_MODEL="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

STAMP=$(date +%Y%m%d_%H%M%S)
[[ -z "$OUTPUT" ]] && OUTPUT="data/stiction_hyst_${STAMP}.csv"
[[ -e "$OUTPUT" ]] && { echo "Output $OUTPUT already exists — pass --output"; exit 1; }

TOPIC="/${ROBOT_MODEL}/joint_states"
BASE=$(basename "$OUTPUT" .csv)
SIDECAR="data/${BASE}.poses.json"
mkdir -p data data/logs
REC_LOG="data/logs/${BASE}.recorder.log"
MOVE_LOG="data/logs/${BASE}.mover.log"

step() { echo; echo "════ [$1] $2 ════"; }
die()  { echo; echo "ABORT: $1" >&2; exit 1; }

# ── [1/4] Environment ─────────────────────────────────────────────────────────
step 1/4 "Environment"
if ! python3 -c 'import rclpy' >/dev/null 2>&1; then
    [[ -f /opt/ros/humble/setup.bash ]] && { set +u; source /opt/ros/humble/setup.bash; set -u; }
fi
python3 -c 'import rclpy' >/dev/null 2>&1 || die "rclpy not importable — source ROS 2 and re-run"
if ! python3 -c 'import interbotix_xs_modules.xs_robot.arm' >/dev/null 2>&1; then
    [[ -f "$HOME/interbotix_ws/install/setup.bash" ]] && { set +u; source "$HOME/interbotix_ws/install/setup.bash"; set -u; }
fi
python3 -c 'import interbotix_xs_modules.xs_robot.arm' >/dev/null 2>&1 \
    || die "interbotix_xs_modules not importable — source the interbotix workspace and re-run"
echo "Python environment OK."

# ── [2/4] Recorder (background) ──────────────────────────────────────────────
# 6 targets × 2 approaches × (2 moves + dwell) + homing/grasp slack.
step 2/4 "Recorder → $OUTPUT"
REC_DURATION=$(python3 -c "print(int(6*2*(2*$MOVE_TIME+$DWELL)+150))")
python3 -u record_joint_states_200hz.py \
    --duration "$REC_DURATION" --output "$OUTPUT" --topic "$TOPIC" \
    >"$REC_LOG" 2>&1 &
REC_PID=$!
cleanup() {
    if kill -0 "$REC_PID" 2>/dev/null; then
        kill -INT "$REC_PID" 2>/dev/null || true
        wait "$REC_PID" 2>/dev/null || true
    fi
}
trap cleanup EXIT
for _ in $(seq 1 60); do
    grep -q RECORDING_STARTED "$REC_LOG" 2>/dev/null && break
    kill -0 "$REC_PID" 2>/dev/null || { cat "$REC_LOG"; die "recorder exited before receiving data"; }
    sleep 0.5
done
grep -q RECORDING_STARTED "$REC_LOG" || die "recorder saw no message within 30 s — see $REC_LOG"
echo "Recorder running (PID $REC_PID, log $REC_LOG)."

# ── [3/4] Mover (foreground) ─────────────────────────────────────────────────
step 3/4 "Hysteresis sequence (dwell ${DWELL}s, move ${MOVE_TIME}s, delta ${DELTA} rad)"
python3 -u control/stiction_hysteresis_poses.py \
    --robot-model "$ROBOT_MODEL" --move-time "$MOVE_TIME" --dwell "$DWELL" \
    --delta "$DELTA" --sidecar "$SIDECAR" 2>&1 | tee "$MOVE_LOG" \
    || die "mover failed — see $MOVE_LOG (recorder stopped; partial CSV at $OUTPUT)"

echo "Poses done — stopping recorder …"
kill -INT "$REC_PID" 2>/dev/null || true
wait "$REC_PID" 2>/dev/null || true
echo "── recorder summary ──"; tail -n 6 "$REC_LOG"

# ── [4/4] Next ───────────────────────────────────────────────────────────────
step 4/4 "Done"
echo "Collection: $OUTPUT   (trial sidecar: $SIDECAR)"
echo "Analyse (no ROS):"
echo "    python3 tools/analyze_stiction_hysteresis.py --csv $OUTPUT \\"
echo "        --phi outputs/npy/traj_run_200hz_20260612_131613__sysid_feasible-v1-5__cfg-a92e984c.npy"
