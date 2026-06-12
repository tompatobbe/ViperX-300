# HANDOVER — start here

**Last updated:** 2026-06-12 (evening, after second 200 Hz run). **Phase:**
identification **COMPLETE** → **control**, with one open identification
experiment (γ sweep / motor inertia) running in parallel.

> **2026-06-12 evening — replicate 200 Hz run collected; γ sweep is the live
> experiment, half-finished. Pick up here.**
>
> **What happened after the cross-validation entry below:**
> 1. **Second 200 Hz collection** `data/traj_run_200hz_20260612_161025.csv`
>    (185 000 rows, 926.9 s, 199.6 Hz, 25 sentinel rows, PASS; 2 stalls
>    absorbed, worst dt gap 173 ms). **Same seed-42 trajectory as the 13:16
>    run → it is a *replicate*, not an independent held-out set.** The May run
>    (`traj_run_20260518_143818.csv`) remains the only independent held-out.
> 2. **Repeatability confirmed** (friction-fitted mean RMSE, `--drop-glitches`):
>    factory 0.719→0.707, 200 Hz model `cfg-a92e984c` 0.460 (held-in on 13:16)
>    →0.468 on the replicate. ~1.5 % drift ⇒ collection is reproducible;
>    differences >0.01 Nm between models are real. The a92e984c **waist defect
>    reproduced exactly** (RMSE 0.452, R² −7.1 vs factory 0.133/+0.30) —
>    structural, consistent with the inertia-inflation hypothesis.
> 3. **γ sweep (`sweep_gamma.sh`, on the 13:16 CSV) ran only its first point**
>    before being interrupted for the collection. Result in
>    `data/logs/gamma_sweep_20260612_160544.txt`:
>    **γ=0.1 → held-in 0.437 (ties May's 0.438!), held-out 0.807** (vs 2.355 at
>    γ=0.05), upper-arm inertia deflated to ≈0.0046. γ is a real lever but 0.1
>    doesn't yet beat May held-out (0.645). **γ ∈ {0.2, 0.5, 1.0, 2.0} pending.**
> 4. **Identification on the replicate CSV already ran** (γ=0.05 recipe;
>    artifacts `outputs/{npy,urdf}/traj_run_200hz_20260612_161025__…cfg-a92e984c…`).
>    Its upper-arm inertia is **inflated again** (ixx 0.051) — the artifact
>    reproduces on independent data. Its validation numbers were **not**
>    recorded; May-model-on-replicate also not yet run.
> 5. Tooling: `identify_200hz.sh` now takes the CSV as optional `$1` and its
>    validation steps gained `--drop-glitches` (required for comparable
>    numbers). The `[warn] --fs=50.0 but CSV is 200.0 Hz` is benign —
>    `load_and_filter` overrides with the detected rate (`sysid_feasible.py:278`).
>
> **Next actions (at home, in order):**
> ```bash
> # 1. Finish the γ sweep (≈25 min/point; edit CSV= inside if you want the replicate)
> bash sweep_gamma.sh
> # 2. Complete the comparison matrix on the replicate CSV:
> source /opt/ros/humble/setup.bash
> python3 compare_urdf_performance.py --friction --drop-glitches \
>     --csv data/traj_run_200hz_20260612_161025.csv \
>     --urdf-b outputs/urdf/traj_run_20260518_143818__sysid_feasible-v1-4__cfg-640cb8ef__phi_to_urdf-v1-0__cfg-3ef0a00c.urdf
> python3 compare_urdf_performance.py --friction --drop-glitches \
>     --csv data/traj_run_200hz_20260612_161025.csv \
>     --urdf-b outputs/urdf/traj_run_200hz_20260612_161025__sysid_feasible-v1-4__cfg-a92e984c__phi_to_urdf-v1-0__cfg-3ef0a00c.urdf
> ```
> **Decision rule:** a 200 Hz model dethrones the delivered May model only if it
> beats **both** 0.645 held-out on May data **and** ≈0.44 on 200 Hz data. If no
> γ achieves that, the inflation is structural → implement the **`Ia·q̈`
> per-joint motor-inertia term** in the regressor (THESIS_NOTES "Cross-run
> validation and the 200 Hz re-identification puzzle").
>
> **Home-PC prerequisites:** identification needs only numpy/scipy/cvxpy
> (+CLARABEL); **validation needs real pinocchio 3.9 via ROS 2 Humble**
> (`source /opt/ros/humble/setup.bash` — see Env gotchas below). Data CSVs are
> tracked in git (the 76 MB replicate is under GitHub's 100 MB limit), so a
> plain `git pull` brings everything; CHANGELOG entries for today's evening
> results are **not yet written** — write them once the matrix completes.

> **2026-06-12 — the delivered model is now CROSS-VALIDATED; control phase is
> unblocked.** The 200 Hz dataset was collected
> (`data/traj_run_200hz_20260612_131613.csv`, 185 238 rows, 199.7 Hz, PASS)
> after a day of failure forensics — five arm collapses root-caused to a loose
> 12 V brick→power-hub connection sagging under load (CHANGELOG 2026-06-12;
> `volt_watch.py` is the reusable diagnostic; driver `vx300s.yaml` update_rate
> 50→200 outside this repo; `run_trajectories.py` gained anti-burst pacing).
> The full cross-validation matrix (CHANGELOG table) then showed: **the May
> model scores 0.438 RMSE / 0.313 MAE held-out** on the unseen 200 Hz data —
> better than its held-in numbers, 39 % ahead of factory — while re-identifying
> on the 200 Hz data under the unchanged recipe is *strictly dominated*
> (suspected unmodeled reflected actuator inertia; see THESIS_NOTES). **Next:
> either start the control phase on the validated `cfg-640cb8ef` model, or run
> the motor-inertia identification experiment (`Ia·q̈` term + γ sweep) on the
> 200 Hz data.** Power headroom is thin (9.7 V min under load); secure the
> brick→PHB connector before long campaigns. Use `--drop-glitches` in all
> validations from now on.

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
>
> **Next action: the 200 Hz re-collection — tooling is ready (2026-06-11 evening).**
> One-command gated flow, new files only (old scripts untouched):
> ```bash
> bash collect_200hz.sh --smoke     # 60 s rehearsal through ALL gates
> bash collect_200hz.sh             # the real 900 s run (seed-42 trajectory)
> ```
> The script gates everything before the arm moves (env, FTDI latency_timer → 1 ms,
> topic-rate ≥ 150 Hz) and verdicts the CSV after (`check_collection.py`, validated
> against the May run's known 46.7 Hz / 22 dropouts). New Tier-2 recorder
> (`record_joint_states_200hz.py`): every message, header-stamp time base — the
> runbook's throttle caveat is gone. Runbook: **`docs/COLLECTION_200HZ.md`** (top
> section); excitation-conditioning (Tier 3) remains deferred.

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
