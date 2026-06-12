#!/usr/bin/env bash
# One-shot identify → export → validate for the 2026-06-12 200 Hz run.
# Exists because multi-line paste keeps splitting commands (see CHANGELOG
# 2026-06-12). Recipe = the delivered-model settings (HANDOVER), plus
# --stride 4 (subsamples AFTER 200 Hz filtering/differentiation; keeps W
# small enough for WSL2 RAM) and --drop-glitches (33 sentinel rows).
set -eo pipefail
cd "$(dirname "$0")"

CSV=data/traj_run_200hz_20260612_131613.csv
MAY_CSV=data/traj_run_20260518_143818.csv

echo "════ [1/4] Identification (SDP, entropic 0.05, w2 100, stride 4) ════"
python3 sysid_feasible.py "$CSV" --no-plot --stride 4 --method cvxpy \
    --entropic 0.05 --w2 100 --solver CLARABEL --drop-glitches \
    | tee /tmp/identify_200hz.out

NPY=$(grep -oP 'Saved\s+→\s+\K\S+\.npy' /tmp/identify_200hz.out | tail -1)
[ -n "$NPY" ] || { echo "ABORT: no npy path found in identification output"; exit 1; }
echo; echo "════ [2/4] Export URDF from $NPY ════"
python3 phi_to_urdf.py "$NPY" | tee /tmp/phi_to_urdf.out

URDF=$(grep -oP '(Saved|Wrote)\s+→?\s*\K\S+\.urdf' /tmp/phi_to_urdf.out | tail -1)
[ -n "$URDF" ] || URDF=$(ls -t outputs/urdf/*.urdf | head -1)
echo; echo "════ [3/4] Validation HELD-IN (vs the 200 Hz run itself) ════"
source /opt/ros/humble/setup.bash
python3 compare_urdf_performance.py --friction --csv "$CSV" --urdf-b "$URDF"

echo; echo "════ [4/4] Validation HELD-OUT (vs the independent May run) ════"
python3 compare_urdf_performance.py --friction --csv "$MAY_CSV" --urdf-b "$URDF"

echo; echo "Done. Model: $URDF"
