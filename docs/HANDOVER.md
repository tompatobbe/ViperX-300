# HANDOVER — start here

**Last updated:** 2026-06-12 (late evening; γ sweep + matrix complete). **Phase:**
identification **COMPLETE** → **control**, with one decided identification
experiment queued: the **`Ia·q̈` motor-inertia term**.

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
> **Next actions (in order):**
> 1. **Implement the per-joint motor-inertia term** in the `sysid_feasible.py`
>    regressor: `τᵢ += Ia_i·q̈ᵢ` — 6 extra columns (one per joint, like the
>    friction terms), parameters linear, constraint `Ia_i ≥ 0`; SDP structure
>    unchanged. Bump `PIPELINE_VERSION`. Export note: `Ia` cannot live in a
>    URDF link inertial — keep it sidecar metadata (like F0) for the
>    controller.
> 2. Re-run identification + the cross-validation matrix on the 200 Hz data
>    (reuse `sweep_gamma.sh` machinery). **Success bar unchanged:** beat
>    **0.438** held-in (200 Hz) *and* **0.645** held-out (May), with a sane
>    waist axis. If `Ia·q̈` fails too → next suspects are a richer friction
>    model and excitation conditioning (paper Eq. 11); the May model stays the
>    deliverable and the **control phase starts on it**.
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
