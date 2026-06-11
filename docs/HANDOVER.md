# HANDOVER — start here

**Last updated:** 2026-06-11. **Phase:** identification **COMPLETE**
→ **control** (a validated, control-ready URDF now exists).

> **2026-06-11 note — preprocessing investigated, delivered model unchanged.** Added
> two opt-in flags to `sysid_feasible.py` (`--drop-glitches`, `--use-measured-vel`,
> both default-off) and tested them. Result is a **negative result**: dropout removal
> is negligible (22/40 316 rows), and using the encoder velocity register only *looks*
> better (unconstrained REL 0.48) because that register is **lagged ~43 ms and
> attenuated ~0.4–0.9×** at 47 Hz — its feasible REL is actually worse (0.63) and it
> inflates the F0 offsets. Differentiated position stays the default; the delivered
> model is untouched. **The real lever confirmed: a 200 Hz re-collection** (paper's
> rate) — cleans q̈ for inertia ID *and* shrinks the velocity-register lag. Full detail
> in the 2026-06-11 CHANGELOG entry and THESIS_NOTES "Encoder velocity vs
> differentiated position". All four attribution runs preserved on disk by cfg hash.

This is the "where was I / what's next" anchor. For depth: `CHANGELOG.md` (dated
log), `THESIS_NOTES.md` (discussion), `PAPER_SUMMARY.md` (the paper).

---

## TL;DR — identification is done; the model is delivered

**The validated, control-ready dynamic model:**
- **URDF:** `outputs/urdf/traj_run_20260518_143818__sysid_feasible-v1-4__cfg-640cb8ef__phi_to_urdf-v1-0__cfg-3ef0a00c.urdf`
- **phi:**  `outputs/npy/traj_run_20260518_143818__sysid_feasible-v1-4__cfg-640cb8ef.npy`
- Recipe: `--method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL --stride 1`
- Feasible (`[ALL OK]`), REL 0.5997, and **beats `vx300s.urdf`** on torque
  prediction: friction-fitted mean RMSE **0.822** vs 2.682 Nm (−69.3%), MAE
  **0.386** vs 0.753. (Full numbers in the 2026-06-10 changelog "FINAL" entry.)

**Next: start the control phase** (paper §4–5: robust law + ERG — see
`docs/PAPER_SUMMARY.md` §10). `control/trq.py` already does torque→current.
First two control-phase to-dos that also strengthen the identification chapter:
1. **Held-out validation** — collect/identify on one run, validate on another
   (current result is held-in). The single biggest credibility upgrade.
2. (Optional) implement the paper's **Eq. 11 excitation trajectory** to push REL
   from 0.59 toward the paper's 0.43 and actually identify the per-link inertias
   (this run only identifies gravity `mc` + friction; inertias ≈ generic blob).

*Why the identification settings, for the write-up:* `w2` is the paper's Eq. 16a
coupling weight — strong (100) makes the per-link realisation reproduce the
data-identified base parameters (so the gravity-critical first moments `mc` aren't
pulled off by the prior); `--entropic γ` (our documented deviation) resolves the
residual mass-scale non-uniqueness toward a generic blob `P0=diag(--ref-inertia·I₃,
--ref-mass)` — a scale only, never the CAD inertials. Pure `−log det` is unbounded
(it exploded masses to 8000 kg); the bounded log-det Bregman divergence is what
works. Full derivation/justification in `THESIS_NOTES.md`.

**To reproduce the model from scratch** (identification needs no ROS):
```bash
python3 sysid_feasible.py data/traj_run_20260518_143818.csv --no-plot --stride 1 \
  --method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL
python3 phi_to_urdf.py outputs/npy/<the npy it wrote>.npy
source /opt/ros/humble/setup.bash                       # validation needs ROS
python3 compare_urdf_performance.py --friction --urdf-b outputs/urdf/<the urdf>.urdf
```

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
- Implemented + **ran + validated** the fix: **convex SDP** (`--method cvxpy`,
  `identify_sdp`, pseudo-inertia LMI). REL 0.5921 = baseline, feasible, and it
  **beats `vx300s.urdf`** on torque prediction (friction MAE 0.41 vs 0.75 Nm).
- Standard masses collapse (degenerate realisation) → added the model-free
  realisation selector: first tried pure `−log det` (unbounded → masses exploded
  to 8000 kg), then the **bounded log-det Bregman divergence** `--entropic γ` to a
  generic blob `P0` (`--ref-mass`/`--ref-inertia`), plus `--solver` and a
  validation-only `phi_to_urdf --mass-floor`. ← next is to run `--entropic` with a
  tuned γ for a control-ready URDF (TL;DR above).

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
