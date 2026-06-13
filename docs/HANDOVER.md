# HANDOVER — start here

**Last updated:** 2026-06-13 (KINEMATICS ROOT-CAUSE FIXED). **Phase:**
identification **REOPENED** — a DH-convention bug that corrupted every model so
far has been found and fixed; everything must be re-identified before control.

---

## ⇒ LAB RE-RUN PLAN (do this — start here)

**What changed (one line).** `sysid_feasible.py` was running the modified-DH
`DH_PARAMS` table through a *standard*-DH transform + Newton–Euler recursion, so
the shoulder was modelled as a yaw axis instead of a pitch axis and gravity
could never be identified. Fixed to modified (Craig) DH — matches `sim/sim.py`
and the real `urdf/vx300s.urdf`. **Verified** (regressor == Pinocchio RNEA to
3e-7 Nm; FK axis structure matches the real robot). Details: CHANGELOG
2026-06-13 "ROOT CAUSE FOUND & FIXED".

**Consequence.** Every identified `phi`/URDF to date is invalid (May
`cfg-640cb8ef`, Ia `cfg-ce8e7059`, all γ-sweep models). `PIPELINE_VERSION` was
bumped (sysid 1.4→1.5, phi_to_urdf 1.0→1.1) so all artifacts recompute. The
whole model-selection history is moot. **Re-identify from scratch.**

**Datasets (unchanged, still valid — only the recordings, not the models):**
- 200 Hz primary: `data/traj_run_200hz_20260612_131613.csv`
- 200 Hz replicate (same seed-42 traj): `data/traj_run_200hz_20260612_161025.csv`
- May 47 Hz (independent held-out): `data/traj_run_20260518_143818.csv`

**Step 0 — sanity-check the fix is in place (fast, do once):**
```bash
source /opt/ros/humble/setup.bash
python3 tools/test_fk_equivalence.py          # expect: PASS (axis structure matches real URDF)
python3 tools/test_phi_urdf_consistency.py    # expect: PASS (regressor == URDF RNEA, ~3e-7 Nm)
```

**Step 1 — re-identify (plain model, the old "May recipe", now on corrected
kinematics). Identification does NOT need ROS:**
```bash
# May data
python3 sysid_feasible.py data/traj_run_20260518_143818.csv \
    --method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL --drop-glitches --no-plot
# 200 Hz data
python3 sysid_feasible.py data/traj_run_200hz_20260612_131613.csv \
    --method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL --drop-glitches --stride 4 --no-plot
```
Each prints `Saved → outputs/npy/…__sysid_feasible-v1-5__cfg-XXXXXXXX.npy`. Note
the two new paths (call them `$PHI_MAY`, `$PHI_200`).

**Step 2 — export URDFs:**
```bash
python3 phi_to_urdf.py $PHI_MAY      # → outputs/urdf/…phi_to_urdf-v1-1….urdf  (call it $URDF_MAY)
python3 phi_to_urdf.py $PHI_200      # → $URDF_200
```

**Step 3 — THE KEY CHECK: does gravity now land on the shoulder/elbow?**
In the `phi_to_urdf` summary the shoulder/elbow CoM and mass should now be
non-trivial (not pinned at the 0.5 kg blob with CoM≈0). Then validate:
```bash
source /opt/ros/humble/setup.bash
python3 compare_urdf_performance.py --csv data/traj_run_200hz_20260612_131613.csv \
    --urdf-b $URDF_200 --friction --drop-glitches
```
**Read the new RIGID-BODY-ONLY (PRIMARY) table and the "Margin over no-model
baseline" line** (added 2026-06-13). Success = model B beats the no-model
baseline by a clear margin on the shoulder/elbow rigid-body torque — that is the
first time we will have a model that actually contains gravity. (Friction-fitted
numbers are now a secondary diagnostic only.)

**Step 4 — held-out cross-check:** validate the 200 Hz model on the May data and
vice-versa (swap `--csv`/`--urdf-b`). Compare per-joint rigid-body RMSE, not the
friction-fitted mean.

**Optional — Ia (reflected motor inertia) variant**, once the plain model is
confirmed sane. Re-run with `--motor-inertia` and validate with `--fit-ia`:
```bash
python3 sysid_feasible.py data/traj_run_200hz_20260612_131613.csv \
    --method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL --drop-glitches \
    --stride 4 --motor-inertia --no-plot
# validate: add --fit-ia to compare_urdf_performance.py (NOT comparable to friction-only numbers)
```
Note: the Ia values from before the fix are also invalid — the shoulder ≈0.8
result must be re-confirmed on corrected kinematics.

**Hardware (optional but high-value) — static gravity experiment.** Hold ~10
spread static poses, record holding currents at standstill. This is the physical
ground-truth check that gravity now sits on the shoulder/elbow, and it settles
the open ≈0.63 measured-vs-CAD gravity scale anomaly (the ×2 dual-motor
`EFFORT_SCALE` / `k_t` assumption — see CHANGELOG "Cross-check against the paper
authors' published model"). Ask Claude to generate the pose list + analysis.

**Still open (not blocking the re-run):**
- `tools/test_fk_equivalence.py` validates the joint-AXIS structure only; link
  lengths `a,d` / absolute reach are not yet asserted by a test (add a
  pose-level calibration check — base+tool+joint-zero offsets).
- The ≈0.63 effort-scale anomaly (resolve with the static experiment above).

**Env reminder:** identification needs only numpy/scipy/cvxpy(+CLARABEL);
validation/sim need real Pinocchio 3.9 via `source /opt/ros/humble/setup.bash`.
The `[warn] --fs=50.0 but CSV is 200.0 Hz` is benign (auto-detected).

---

## Historical record (PRE-kinematics-fix — context only; all models below are now INVALID)

> ⚠️ Everything from here down predates the 2026-06-13 kinematics fix. The
> conclusions (May model is the deliverable, the Ia tie, the γ sweeps, the
> "structural 200 Hz defect") were all produced through the buggy standard-DH
> chain and are superseded. Kept for provenance / thesis narrative only.

**Last updated:** 2026-06-13 (Ia experiment COMPLETE — hypothesis confirmed,
incumbent not dethroned). **Phase:** identification **COMPLETE** → **control**;
optional follow-ups on the Ia track are open but the May model stands.

> **2026-06-12 late evening — γ sweep finished, matrix complete: the 200 Hz
> defect is STRUCTURAL. Next concrete task: implement `Ia·q̈`. Pick up here.**
>
> **What's now established** (full numbers: CHANGELOG "2026-06-12 (evening)"
> entry; discussion: THESIS_NOTES "Resolution of the γ question"):
> 1. **γ sweep done** (γ ∈ {0.05…2.0}, 13:16 CSV;
>    `data/logs/gamma_sweep_*.txt`). **No γ dethrones the May model**: best
>    held-out 0.761 (γ=0.2) vs the required 0.645, even though γ ≥ 0.1 beats
>    May held-in (down to 0.416 at γ=2.0). The inertia inflation deflates to
>    blob scale by γ=0.5, but the unexplained torque just migrates into F0
>    offsets and held-out worsens again ⇒ **structural**, consistent with
>    reflected actuator inertia.
> 2. **Matrix complete on the replicate** (`…161025.csv`): May model scores
>    **0.438** — identical to the 13:16 run; repeatability ~0.01 Nm. The
>    replicate-identified γ=0.05 model gets the best-ever held-in (0.375) but
>    reproduces the **broken waist axis** (R² −2.90 vs factory +0.30) on
>    independent data.
> 3. Delivered model **unchanged**: May `cfg-640cb8ef`, now validated on two
>    independent 200 Hz collections. All evening docs are written (CHANGELOG,
>    THESIS_NOTES); `sweep_gamma.sh` was fixed to survive artifact cache hits.
>
> **2026-06-13: the `Ia·q̈` experiment is COMPLETE** (implementation +
> smoke tests + real run + full matrix; CHANGELOG 2026-06-13 "motor-inertia"
> entry incl. Results; THESIS_NOTES "Reflected motor inertia" → Outcome).
>
> **Tooling now in the repo:** `sysid_feasible.py --motor-inertia` (phi →
> (84,), SDP only), `phi_to_urdf.py` handles (84,) (Ia → URDF comment +
> sidecar, never inertials), `compare_urdf_performance.py --fit-ia` (the
> protocol for Ia models — NOT comparable to friction-only numbers),
> `sweep_gamma_ia.sh`, `tests/test_motor_inertia.py` (suite 47/47).
>
> **Outcome.** Ia model `cfg-f512651d` (13:16 CSV, May recipe + Ia):
> feasible, Ia = [0.084, **0.798**, 0.156, 0, 0.040, 0.009] kg·m² (plausible,
> 3 estimators agree on shoulder ≈ 0.6–0.8), upper-arm inflation gone at
> γ=0.05, **waist axis healed** (held-out waist 0.244 beats the May model's
> held-in 0.275). But the matrix (friction+Ia-fitted RMSE/MAE,
> `--drop-glitches`):
>
> | model \ data | 200 Hz 13:16 | May 47 Hz |
> |---|---|---|
> | factory | 0.618 / 0.452 | 1.488 / 0.520 |
> | May `cfg-640cb8ef` | **0.332** / 0.231 held-out | **0.520** / 0.351 held-in |
> | Ia `cfg-f512651d` | 0.343 / 0.239 held-in | 0.579 / 0.379 held-out |
>
> **Not dethroned** (−3 % / −11 %, gap concentrated in the shoulder) → the
> **May model stays the deliverable**. Caveats (THESIS_NOTES self-audit): the
> Ia model ran on May-tuned hyperparameters, and challengers are held-out on
> the dirty May CSV while the incumbent is held-out on clean 200 Hz data.
>
> **γ retune DONE (same day, `sweep_gamma_ia.sh`): STATISTICAL TIE.**
> γ=0.5 (`cfg-ce8e7059`): held-in 0.335 / held-out 0.526 vs May 0.332 / 0.520
> — both gaps below the ~0.01 Nm repeatability resolution. Ia is γ-invariant
> (shoulder 0.798 ± 0.1 % over 25× γ) → strongly identified by the data; the
> γ=0.5 realisation is presentable (CoMs ≤ 2 cm). Strict decision rule: May
> model keeps the crown; honest statement: indistinguishable. **Tuning is
> exhausted — do NOT keep evaluating against the May CSV** (its held-out
> independence is spent; multiple-comparisons). Details: CHANGELOG "γ retune"
> entry; THESIS_NOTES "γ retune result" incl. the pre-registered tiebreak.
>
> **Open paths (pick one):**
> 1. **Tiebreak with new data**: different-seed 200 Hz collection (change the
>    seed in `run_trajectories.py`, run via `collect_200hz.sh`), then ONE
>    pre-registered comparison — May `cfg-640cb8ef` vs Ia-γ0.5 `cfg-ce8e7059`,
>    `--friction --fit-ia --drop-glitches`, lower mean RMSE wins, no retuning
>    afterwards.
> 2. **Start the control phase** — on the May model (strict reading), or
>    argue the Ia-γ0.5 model on qualitative grounds (healed waist, explicit
>    actuator model). Either way the controller should use the Ia
>    feed-forward (sidecar JSON `motor_inertia_Ia`; values ≈ [0.083, 0.798,
>    0.156, 0, 0.040, 0.009] kg·m², stable across all runs).
>
> **Prerequisites:** identification needs only numpy/scipy/cvxpy (+CLARABEL);
> **validation needs real pinocchio 3.9 via ROS 2 Humble**
> (`source /opt/ros/humble/setup.bash` — see Env gotchas below). The
> `[warn] --fs=50.0 but CSV is 200.0 Hz` is benign — `load_and_filter`
> overrides with the detected rate (`sysid_feasible.py:278`). The May run
> (`traj_run_20260518_143818.csv`) remains the only independent held-out set;
> both 200 Hz runs share the seed-42 trajectory.

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
