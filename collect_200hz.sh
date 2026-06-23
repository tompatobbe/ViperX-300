#!/usr/bin/env bash
# One-command, gated 200 Hz identification-data collection
# (procedure: docs/COLLECTION_200HZ.md). New-files-only: uses
# record_joint_states_200hz.py / check_topic_rate.py / check_collection.py and
# the existing, unmodified run_trajectories.py (seed 42 = the trajectory that
# produced the delivered model). Past collection scripts are untouched.
#
# Prerequisite (own terminal): the arm driver must be running:
#   ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s
#
# Recommended lab flow:
#   bash collect_200hz.sh --smoke     # 60 s end-to-end rehearsal, all gates
#   bash collect_200hz.sh             # full 900 s run
#
# Options:
#   --smoke           60 s rehearsal (output data/smoke_200hz_<stamp>.csv)
#   --duration S      trajectory duration, integer seconds (default 900; smoke 60)
#   --design PATH     replay a vetted, optimised design .npz (run_trajectories
#                     --load); deterministic, skips the optimiser. Recommended.
#   --seed N          excitation coefficient seed (default 42; ignored with --design)
#   --stride N        send every N-th waypoint (default 30 → ~6.7 Hz commands;
#                     matches the 0.5 Hz trajectory bandwidth, robust to comms stalls)
#   --min-rate HZ     topic-rate gate threshold (default 150)
#   --output PATH     CSV path (default data/traj_run_200hz_<stamp>.csv)
#   --robot-model M   robot model (default vx300s)
#   --skip-latency    skip the FTDI latency_timer step (rate gate still decides)

set -euo pipefail
cd "$(dirname "$0")"

DURATION=""
SEED=42
# Command rate = 200/STRIDE Hz. stride 30 ⇒ ~6.7 Hz, which matches the trajectory's
# real bandwidth (0.5 Hz, 13× oversampled) and stays robust to WSL2/usbipd comms
# stalls (a stall shorter than one ~150 ms command interval no longer forces a
# schedule shift). stride 4 (50 Hz) floods the link and re-triggers the stalls.
STRIDE=30
MIN_RATE=150
OUTPUT=""
ROBOT_MODEL="vx300s"
SMOKE=0
SKIP_LATENCY=0
DESIGN=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --smoke)        SMOKE=1;           shift   ;;
        --duration)     DURATION="$2";     shift 2 ;;
        --design)       DESIGN="$2";       shift 2 ;;
        --seed)         SEED="$2";         shift 2 ;;
        --stride)       STRIDE="$2";       shift 2 ;;
        --min-rate)     MIN_RATE="$2";     shift 2 ;;
        --output)       OUTPUT="$2";       shift 2 ;;
        --robot-model)  ROBOT_MODEL="$2";  shift 2 ;;
        --skip-latency) SKIP_LATENCY=1;    shift   ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [[ -n "$DESIGN" && ! -f "$DESIGN" ]]; then
    echo "--design file not found: $DESIGN"; exit 1
fi

if [[ -z "$DURATION" ]]; then
    [[ "$SMOKE" = 1 ]] && DURATION=60 || DURATION=900
fi
case "$DURATION" in *[!0-9]*|'') echo "--duration must be integer seconds"; exit 1 ;; esac

STAMP=$(date +%Y%m%d_%H%M%S)
if [[ -z "$OUTPUT" ]]; then
    [[ "$SMOKE" = 1 ]] && OUTPUT="data/smoke_200hz_${STAMP}.csv" \
                       || OUTPUT="data/traj_run_200hz_${STAMP}.csv"
fi
[[ -e "$OUTPUT" ]] && { echo "Output $OUTPUT already exists — pass --output"; exit 1; }

TOPIC="/${ROBOT_MODEL}/joint_states"
BASE=$(basename "$OUTPUT" .csv)
mkdir -p data data/logs
REC_LOG="data/logs/${BASE}.recorder.log"
TRAJ_LOG="data/logs/${BASE}.trajectory.log"

step() { echo; echo "════ [$1] $2 ════"; }
die()  { echo; echo "ABORT: $1" >&2; exit 1; }

# ── [1/6] Environment ─────────────────────────────────────────────────────────
step 1/6 "Environment"
if ! python3 -c 'import rclpy' >/dev/null 2>&1; then
    if [[ -f /opt/ros/humble/setup.bash ]]; then
        echo "rclpy not importable — sourcing /opt/ros/humble/setup.bash"
        set +u; source /opt/ros/humble/setup.bash; set -u
    fi
fi
python3 -c 'import rclpy' >/dev/null 2>&1 \
    || die "rclpy not importable — source your ROS 2 environment and re-run"
if ! python3 -c 'import interbotix_xs_modules.xs_robot.arm' >/dev/null 2>&1; then
    if [[ -f "$HOME/interbotix_ws/install/setup.bash" ]]; then
        echo "interbotix_xs_modules not importable — sourcing ~/interbotix_ws/install/setup.bash"
        set +u; source "$HOME/interbotix_ws/install/setup.bash"; set -u
    fi
fi
python3 -c 'import interbotix_xs_modules.xs_robot.arm' >/dev/null 2>&1 \
    || die "interbotix_xs_modules not importable — source the interbotix workspace and re-run"
echo "Python environment OK (rclpy + interbotix_xs_modules)."

# ── [2/6] FTDI latency timer (the 47-Hz root cause; runbook Step 1) ──────────
step 2/6 "FTDI latency timer"
if [[ "$SKIP_LATENCY" = 1 ]]; then
    echo "Skipped (--skip-latency); the rate gate below is authoritative."
else
    TTY=""
    if [[ -e /dev/ttyDXL ]]; then
        TTY=$(basename "$(readlink -f /dev/ttyDXL)")
    else
        CANDS=$(ls /sys/bus/usb-serial/devices/ 2>/dev/null || true)
        if [[ $(echo "$CANDS" | grep -c .) -eq 1 ]]; then
            TTY="$CANDS"
            echo "/dev/ttyDXL absent — using the only usb-serial device: $TTY"
        elif [[ -n "$CANDS" ]]; then
            echo "WARN: /dev/ttyDXL absent and several usb-serial devices exist:"
            echo "$CANDS"
        fi
    fi
    LATFILE="/sys/bus/usb-serial/devices/${TTY}/latency_timer"
    if [[ -z "$TTY" || ! -f "$LATFILE" ]]; then
        echo "WARN: latency_timer sysfs entry not found — continuing;"
        echo "      the rate gate below will catch a slow bus."
    else
        LAT=$(cat "$LATFILE")
        echo "$TTY latency_timer = ${LAT} ms"
        if [[ "$LAT" != "1" ]]; then
            echo "Setting to 1 ms (needs sudo) …"
            echo 1 | sudo tee "$LATFILE" >/dev/null
            LAT=$(cat "$LATFILE")
            if [[ "$LAT" = "1" ]]; then
                echo "→ latency_timer now 1 ms (runtime only; udev rule makes it persistent — runbook Step 1.4)"
            else
                echo "WARN: still ${LAT} ms — the rate gate below will decide."
            fi
        fi
    fi
fi

# ── [3/6] Topic-rate GATE (runbook Step 3 — do not skip) ─────────────────────
step 3/6 "Topic-rate gate on $TOPIC"
python3 check_topic_rate.py --topic "$TOPIC" --duration 5 --min-rate "$MIN_RATE" \
    || die "topic-rate gate failed — fix per docs/COLLECTION_200HZ.md Steps 1–3, then re-run"

# ── [4/6] Recorder (background; duration has slack for the optimiser+home) ───
step 4/6 "Recorder → $OUTPUT"
REC_DURATION=$((DURATION + 360))
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
    kill -0 "$REC_PID" 2>/dev/null \
        || { cat "$REC_LOG"; die "recorder exited before receiving data"; }
    sleep 0.5
done
grep -q RECORDING_STARTED "$REC_LOG" \
    || die "recorder saw no message within 30 s — see $REC_LOG"
echo "Recorder running (PID $REC_PID, log $REC_LOG); it will be stopped when the trajectory ends."

# ── [5/6] Excitation trajectory (existing, proven script) ────────────────────
# With --design: replay the vetted, optimised coefficients (--load), so the
# collected trajectory is exactly the one approved offline (no live re-optimise,
# deterministic). Without it: the legacy seed-based path.
if [[ -n "$DESIGN" ]]; then
    step 5/6 "Excitation trajectory (${DURATION}s, design $DESIGN, stride $STRIDE)"
    TRAJ_ARGS=(--load "$DESIGN")
else
    step 5/6 "Excitation trajectory (${DURATION}s, seed $SEED, stride $STRIDE)"
    TRAJ_ARGS=(--seed "$SEED")
fi
python3 -u run_trajectories.py \
    --duration "$DURATION" --rate 200 --robot-model "$ROBOT_MODEL" \
    "${TRAJ_ARGS[@]}" --stride "$STRIDE" 2>&1 | tee "$TRAJ_LOG" \
    || die "trajectory script failed — see $TRAJ_LOG (recorder stopped; partial CSV at $OUTPUT)"

echo "Trajectory done — stopping recorder …"
kill -INT "$REC_PID" 2>/dev/null || true
wait "$REC_PID" 2>/dev/null || true
echo "── recorder summary ──"
tail -n 6 "$REC_LOG"

# ── [6/6] Post-run verification (runbook Step 5) ─────────────────────────────
step 6/6 "Post-run verification"
python3 check_collection.py "$OUTPUT" --min-rate "$MIN_RATE" --expect-duration "$DURATION" \
    || die "collection FAILED verification — do not identify on $OUTPUT"

echo
echo "════ Collection OK: $OUTPUT ════"
if [[ "$SMOKE" = 1 ]]; then
    echo "Rehearsal passed — now run the full collection:"
    echo "    bash collect_200hz.sh"
else
    echo "Next (identification needs no ROS):"
    echo "    python3 sysid_feasible.py $OUTPUT --no-plot --stride 1 \\"
    echo "        --method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL"
    echo "    python3 phi_to_urdf.py outputs/npy/<the npy it wrote>.npy"
    echo "    source /opt/ros/humble/setup.bash"
    echo "    python3 compare_urdf_performance.py --friction --csv $OUTPUT \\"
    echo "        --urdf-b outputs/urdf/<the urdf>.urdf"
fi
