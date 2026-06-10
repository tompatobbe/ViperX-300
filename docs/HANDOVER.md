# HANDOVER — start here

**Last updated:** 2026-06-10 (end of session). **Phase:** identification (control
deferred until a valid URDF exists).

This is the "where was I / what's next" anchor. For depth: `CHANGELOG.md` (dated
log), `THESIS_NOTES.md` (discussion), `PAPER_SUMMARY.md` (the paper).

---

## TL;DR — do this first tomorrow

A convex SDP solver (`--method cvxpy`) was implemented but **not yet run**. Step 1
is to install cvxpy and run it.

```bash
# 1. install the SDP solver (identification does NOT need ROS/pinocchio)
python3 -m pip install cvxpy
python3 -c "import cvxpy as cp; print(cp.__version__, cp.installed_solvers())"   # want SCS and/or CLARABEL

# 2. identify with the convex SDP
python3 sysid_feasible.py data/traj_run_20260518_143818.csv --no-plot --stride 5 --method cvxpy
```

**Success = BOTH at once** (neither NLP solver managed this):
- `[6] Final REL` mean ≈ **0.59** (matches the unconstrained baseline = good fit), and
- `[8] … Overall: [ALL OK]` (feasible) with non-degenerate masses.

Then export + validate (validation **needs ROS** for pinocchio):
```bash
# export phi -> URDF (note the 'Saved ->' path it prints)
python3 phi_to_urdf.py outputs/npy/<the-cvxpy-npy-it-just-wrote>.npy

# validate vs baseline (source ROS so the REAL pinocchio loads)
source /opt/ros/humble/setup.bash
python3 compare_urdf_performance.py --friction \
  --urdf-a urdf/vx300s.urdf \
  --urdf-b outputs/urdf/<the-cvxpy-urdf-it-just-wrote>.urdf
```
Goal: SDP model's RMSE should close on / beat the manufacturer baseline
(friction-fitted MAE baseline ≈ 0.753 Nm).

---

## Decision tree on the SDP result

- **REL ≈ 0.59 AND feasible** → success. Re-run at `--stride 1` for the final
  model (the SDP solve is fast regardless of stride; only the regressor build
  scales). Then this becomes the validated URDF → unblocks the control phase.
- **Feasible but REL still ~0.8** → the convex problem is being solved but the fit
  is data-limited; pivot to **data/excitation** (see open question below).
- **cvxpy has no SDP solver** → `python3 -m pip install clarabel` (or scs).
- **Masses feasible but ugly/degenerate** → add the model-free entropic
  (log-det pseudo-inertia) regulariser to `identify_sdp` — already noted as
  "Phase 2" in `THESIS_NOTES.md`. Do NOT use a CAD prior (would make the thesis
  circular — see below).

---

## State of play

**Done this session**
- Fixed effort units: `*_effort` is Dynamixel current in **mA**;
  `τ = (mA/1000)·2.409·motors`, motors=[1,2,2,1,1,1] (shoulder & elbow dual).
  This was the big bug (old runs were ~400× mis-scaled).
- Built `compare_urdf_performance.py` (RNEA torque-prediction validation).
- Re-identified with correct units → **trust-constr v1-4** (`cfg-f8c3b062`):
  feasible, physical masses, friction-fitted MAE 0.708 Nm (≈ matches baseline
  0.753). **This is the current best usable model / fallback.**
- Diagnosed: NLP solvers fail (trust-constr stalls; SLSQP fits but infeasible +
  mass collapse). Root cause = base-vs-standard **non-identifiability**, NOT poor
  excitation (base regressor cond ≈ 197 is fine).
- Implemented the fix: **convex SDP** (`--method cvxpy`, `identify_sdp`) using the
  pseudo-inertia LMI. ← needs running (TL;DR above).

**Key constraint (thesis integrity):** no CAD/reference-model prior in
identification — the thesis must show a model built *from data alone* for a
model-less robot. `vx300s.urdf` is used only as a validation baseline and for
kinematics, never as an inertial prior.

**Env gotchas**
- Validation/sim need **ROS sourced** (`source /opt/ros/humble/setup.bash`) so
  pinocchio is the real 3.9.0 with URDF support; without it a junk
  `~/.local/pinocchio 0.4.3` shadows it. (Consider `pip uninstall pinocchio`.)
- Identification (`sysid_feasible.py`) does NOT use pinocchio — runs without ROS.

**Open question (next big lever after the SDP):** unconstrained REL is 0.59 vs the
paper's 0.43, and conditioning is good — so the gap is data/model fidelity. Likely
needs the paper's Eq. 11 condition-number-optimised **excitation trajectory**
(not yet implemented) and/or a richer friction model.

**Reminder:** new results land as new artifact files (version + config hash);
nothing overwrites. The scripts are run by you (not the agent) so you can watch
progress.
