#!/usr/bin/env bash
# γ sweep on the 200 Hz run (2026-06-12, see THESIS_NOTES "Cross-run validation
# and the 200 Hz re-identification puzzle"). Question: is the upper-arm inertia
# inflation regularisable away (some γ beats the May model held-out), or
# structural (inflation persists → implement the Ia·q̈ motor-inertia term)?
#
# References (friction-fitted mean RMSE, --drop-glitches):
#   γ=0.05 (cfg-a92e984c): held-in 0.460, held-out 2.355
#   May model (cfg-640cb8ef): 0.438 on 200 Hz data, 0.645 on May data
set -eo pipefail
cd "$(dirname "$0")"
source /opt/ros/humble/setup.bash   # references unset vars — incompatible with -u
set -u

CSV=data/traj_run_200hz_20260612_131613.csv
MAY=data/traj_run_20260518_143818.csv
RESULTS=data/logs/gamma_sweep_$(date +%Y%m%d_%H%M%S).txt

for G in 0.1 0.2 0.5 1.0 2.0; do
    echo "════════════════ γ=$G — identify ════════════════"
    python3 sysid_feasible.py "$CSV" --no-plot --stride 4 --method cvxpy \
        --entropic "$G" --w2 100 --solver CLARABEL --drop-glitches \
        | tee /tmp/id_gamma.out
    NPY=$(grep -oP 'Saved\s+→\s+\K\S+\.npy' /tmp/id_gamma.out | tail -1)

    echo "════ γ=$G — export URDF ════"
    python3 phi_to_urdf.py "$NPY" | tee /tmp/urdf_gamma.out
    URDF=$(grep -oP 'Wrote\s+→\s+\K\S+\.urdf' /tmp/urdf_gamma.out | tail -1)
    # upper-arm inertia: the inflation indicator (May-model scale ≈ 0.04)
    UA=$(awk '/upper_arm_link/ && NF>=4 {line=$0} END{print line}' /tmp/urdf_gamma.out)

    echo "════ γ=$G — validate held-in (200 Hz) ════"
    IN=$(python3 compare_urdf_performance.py --friction --drop-glitches \
         --csv "$CSV" --urdf-b "$URDF" | awk '/^  B  /{v=$(NF-1)} END{print v}')
    echo "════ γ=$G — validate held-out (May) ════"
    OUT=$(python3 compare_urdf_performance.py --friction --drop-glitches \
         --csv "$MAY" --urdf-b "$URDF" | awk '/^  B  /{v=$(NF-1)} END{print v}')

    echo "γ=$G  held-in=$IN  held-out=$OUT  upper_arm: $UA" | tee -a "$RESULTS"
done

echo
echo "════════════════ SWEEP SUMMARY ════════════════"
echo "(beat held-in 0.438 AND held-out 0.645 to dethrone the May model)"
cat "$RESULTS"
