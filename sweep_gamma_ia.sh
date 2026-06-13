#!/usr/bin/env bash
# γ retune for the motor-inertia recipe (2026-06-13, see THESIS_NOTES
# "Reflected motor inertia" → Outcome). The Ia model lost the matrix narrowly
# on May-tuned hyperparameters (γ=0.05); question: does its own γ close the
# gap? Validation uses the --fit-ia protocol (NOT comparable to friction-only
# numbers from sweep_gamma.sh).
#
# References (friction+Ia-fitted mean RMSE, --drop-glitches):
#   Ia model γ=0.05 (cfg-f512651d): held-in 0.343, held-out 0.579
#   May model (cfg-640cb8ef):       0.332 on 200 Hz data, 0.520 on May data
#   → beat BOTH 0.332 and 0.520 to dethrone the May model.
set -eo pipefail
cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash   # references unset vars — incompatible with -u
set -u

CSV=data/traj_run_200hz_20260612_131613.csv
MAY=data/traj_run_20260518_143818.csv
RESULTS=data/logs/gamma_ia_sweep_$(date +%Y%m%d_%H%M%S).txt

for G in 0.02 0.1 0.2 0.5; do
    echo "════════════════ γ=$G (+Ia) — identify ════════════════"
    python3 sysid_feasible.py "$CSV" --no-plot --stride 4 --method cvxpy \
        --entropic "$G" --w2 100 --solver CLARABEL --drop-glitches \
        --motor-inertia | tee /tmp/id_gamma_ia.out
    # Matches both "Saved → …" and the bare path a cache hit prints.
    NPY=$(grep -oP '\S*outputs/\S+\.npy' /tmp/id_gamma_ia.out | tail -1)

    echo "════ γ=$G — export URDF ════"
    python3 phi_to_urdf.py "$NPY" | tee /tmp/urdf_gamma_ia.out
    URDF=$(grep -oP '\S*outputs/\S+\.urdf' /tmp/urdf_gamma_ia.out | tail -1)
    # shoulder Ia: the physical-plausibility indicator (expect ~0.6-0.8)
    SH_IA=$(grep -oP 'shoulder=\K[0-9.]+' "$URDF" | head -1)

    echo "════ γ=$G — validate held-in (200 Hz, --fit-ia) ════"
    IN=$(python3 compare_urdf_performance.py --friction --fit-ia --drop-glitches \
         --csv "$CSV" --urdf-b "$URDF" | awk '/^  B  /{v=$(NF-1)} END{print v}')
    echo "════ γ=$G — validate held-out (May, --fit-ia) ════"
    OUT=$(python3 compare_urdf_performance.py --friction --fit-ia --drop-glitches \
         --csv "$MAY" --urdf-b "$URDF" | awk '/^  B  /{v=$(NF-1)} END{print v}')

    echo "γ=$G  held-in=$IN  held-out=$OUT  shoulder_Ia=$SH_IA" | tee -a "$RESULTS"
done

echo
echo "════════════════ SWEEP SUMMARY (+Ia, --fit-ia protocol) ════════════════"
echo "(beat held-in 0.332 AND held-out 0.520 to dethrone the May model)"
cat "$RESULTS"
