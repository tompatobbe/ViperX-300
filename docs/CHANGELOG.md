# Engineering Changelog

A running, thesis-oriented record of **substantive changes to the code and
methodology** — what changed, *why*, the evidence behind it, and how it affects
results. The intent is that each entry can be cited or paraphrased directly in
the dissertation's "Implementation" / "Methodology" discussion.

Entries are newest-first. Each follows the template at the bottom of this file.

---

## 2026-06-11 — 200 Hz collection tooling: gated one-command run (new files only)

**Area:** data collection · new files `collect_200hz.sh`,
`record_joint_states_200hz.py`, `check_topic_rate.py`, `check_collection.py` ·
no existing file modified

### Problem / Motivation
The 200 Hz re-collection (`docs/COLLECTION_200HZ.md`) is the next experiment. The
previous collection had known fault modes with **no guard** against any of them:
1. Topic published ~47 Hz instead of 200 (FTDI `latency_timer` 16 ms) — discovered
   only *after* the 900 s run was spent.
2. Recorder-throttle halving trap (runbook Step-4 caveat): recorder `--rate` ≈
   publish rate can drop every other message (throttle compares against the last
   *written* row).
3. Recorder/trajectory coverage mismatch: `run_trajectories.sh` gives both the same
   `--duration`, but the trajectory starts ~40 s later (SLSQP optimiser + home move),
   so the recorder can die before the trajectory ends — the delivered CSV spans
   865.6 s of a 900 s request.
4. No machine-checked verdict on a fresh CSV before identification effort is spent
   on it.

### Change
Four new files; past runs and config hashes are untouched:
- **`collect_200hz.sh`** — one-command gated run. Order: (1) env check (rclpy +
  interbotix imports, auto-sources ROS/workspace if needed); (2) FTDI
  `latency_timer` → 1 ms (sudo only if ≠ 1; warns and defers to the rate gate if
  sysfs is missing); (3) **topic-rate gate** — refuses to start below `--min-rate`
  (default 150 Hz); (4) recorder in background with `duration + 360 s` slack (covers the SLSQP
  optimiser + home move; recorder is stopped early when the trajectory exits),
  waits for its first-message sentinel; (5) the **unmodified, proven**
  `run_trajectories.py` with `--rate 200 --seed 42 --stride 4` — the same SLSQP
  condition-number-optimised Eq. 7 excitation that produced the delivered model;
  recorder is SIGINT-stopped when the trajectory exits (fixes fault 3);
  (6) `check_collection.py` verdict, then prints the identify/validate commands.
  `--smoke` = 60 s end-to-end rehearsal through identical gates.
- **`record_joint_states_200hz.py`** — Tier-2 recorder, pulled forward from the
  runbook's deferred list (a new file was being written anyway, and keeping the
  throttle foot-gun in new code would be a deliberate fault): records **every**
  message (no throttle → Step-4 caveat gone), `time` column from
  **`msg.header.stamp`** (driver's sample clock; wall-clock fallback with a loud
  warning if stamps are zero), BEST_EFFORT/KEEP_LAST-50 QoS, watchdogs
  (first-message timeout → exit 1, mid-run stall warning, duration+30 s hard stop),
  live dropout-sentinel counter, extra trailing `recv_time` column (arrival wall
  time, for jitter diagnostics).
- **`check_topic_rate.py`** — scriptable runbook Step-3 gate: measures the actual
  topic rate for 5 s, reports arrival-dt and header-stamp health, exit code 0/1/2.
- **`check_collection.py`** — runbook Step-5 verdict as a gate: median rate,
  dt jitter/gaps, dropout-sentinel rows, empty cells, dead effort columns, span vs
  expected, per-joint motion coverage; exit 0 = usable, 1 = do not identify.

### Evidence
- CSV-format compatibility: `sysid_feasible.load_and_filter` (reused by
  `compare_urdf_performance.py`) selects columns **by name**, so the extra
  `recv_time` column is invisible to the pipeline; `time` keeps its
  seconds-from-first-sample semantics.
- `check_collection.py` validated against the delivered run
  (`traj_run_20260518_143818.csv`): reproduces the known ground truth — median
  46.7 Hz, span 865.6 s, **exactly 22 dropout-sentinel rows** — FAILs at the
  150 Hz gate and PASSes with `--min-rate 40`, exercising both exit paths.
- `bash -n` / `py_compile` clean. The rclpy code mirrors the proven
  `record_joint_states.py` structure (plain subscriber + `spin_once` loop +
  signal flag); hardware-facing paths can only run in the lab — which is what
  `--smoke` rehearses.

### Impact
- Lab procedure: `bash collect_200hz.sh --smoke`, inspect, then
  `bash collect_200hz.sh`. Logs land in `data/logs/<run>.{recorder,trajectory}.log`.
- No existing artifact, script, or config hash affected (new files only; new CSVs
  are `traj_run_200hz_<stamp>.csv`).
- New CSVs gain `recv_time` and a header-stamp time base; downstream pipeline
  unchanged.

### Open questions / assumptions
- Assumes the xs_sdk stamps `/vx300s/joint_states`; if not, the recorder falls back
  to wall clock (warned), and `check_topic_rate.py` reports stamp health before
  anything moves.
- Latency-timer step resolves `/dev/ttyDXL` (or a single usb-serial device); when
  ambiguous it warns and lets the rate gate decide — the gate is authoritative.
- Smoke CSVs (`smoke_200hz_*.csv`) are rehearsal-only, not identification inputs.

---

## 2026-06-11 — Two opt-in data-conditioning improvements (glitch drop + measured velocity)

**Area:** `sysid_feasible.py` (`load_and_filter`, `run_identification`, CLI) · data
preprocessing for identification · opt-in, default-off

### Problem / Motivation
Analysis of the delivered run (`traj_run_20260518_143818.csv`) against the paper's
excitation design surfaced two preprocessing weaknesses (the frequency content and
friction excitation were otherwise good):
1. **22 communication-dropout rows** where the sync-read returns the sentinel −π on
   *all six* joints at once (≈0.05 % of samples, scattered). Left in, each is smeared
   by the zero-phase `filtfilt` into a large spurious velocity/acceleration/torque
   spike around it.
2. **q̇ obtained by differentiating filtered position** (then differentiated again for
   q̈), so the inertial regressor columns ride on double-differentiation noise. The
   encoder already reports velocity directly; the paper differentiates *measured* q̇
   only once. This double-diff noise is a plausible contributor to the per-link
   inertias collapsing to a generic blob (REL stuck at 0.59 vs the paper's 0.43).

### Change
Added two **opt-in** flags to `sysid_feasible.py`, both default-off:
- `--drop-glitches` — removes rows where `all(|q − (−π)| < 1e-3)` before filtering.
- `--use-measured-vel` — uses the `*_vel` columns for q̇ (one fewer differentiation
  stage); q̈ is then a single difference of measured q̇.

The flags are threaded through `run_identification` and added to the config dict
**conditionally** (only when set), so the prior recipe (neither flag) keeps its exact
config hash and still reproduces the delivered artifact byte-for-byte. Enabling either
flag changes the hash → a new artifact filename; nothing is overwritten. `PIPELINE_VERSION`
is deliberately **not** bumped (the default code path is unchanged).

### Evidence
Read-only analysis of the delivered CSV (n=40 338, 865 s, ≈47 Hz):
- 22 rows with all joints = −3.142 exactly (comm dropout), scattered across the run.
- Dominant velocity frequencies on 0.1 Hz fundamental + harmonics to ~0.5 Hz (matches
  paper f_l=0.1 Hz, N_f=5) — excitation design itself is sound.
- Velocity-sign balance ~40–52 % each direction per joint — Coulomb friction
  well-conditioned.

**Attribution runs (delivered recipe `--method cvxpy --entropic 0.05 --w2 100
--solver CLARABEL --stride 1`, REL = mean over joints):**

| Config | Unconstrained REL | Final (feasible) REL | cfg hash |
|---|---|---|---|
| no flags (delivered) | ~0.59 | 0.5997 | 640cb8ef |
| `--drop-glitches` | 0.5806 | 0.6016 | 9ef2c992 |
| `--use-measured-vel` | **0.4786** | 0.6320 | 8c7e81f6 |
| both | 0.4755 | 0.6354 | f740b783 |

Conclusion: **neither flag improves the *feasible* model**, and both are kept
default-off. Specifics:
- `--drop-glitches`: negligible (22/40 316 rows can't move a global fit). Harmless;
  retained as a hygiene option, not a result.
- `--use-measured-vel`: lowers the *unconstrained* REL to 0.48 but the feasible REL
  gets *worse* (0.63), and the gain is an **artifact**. A velocity-consistency check
  (measured `*_vel` vs differentiated position, both filtered) shows the encoder
  velocity is **lagged ≈43 ms (≈2 samples) on every joint and attenuated** — slope
  (measured/differentiated) ≈ 0.41 (waist), 0.48 (forearm_roll), 0.44–0.46 (wrists),
  0.94 (elbow); corr as low as 0.63 (waist). The attenuated/lagged `q̇` shrinks the
  Coriolis/viscous/inertial torque the model must explain (→ lower residual) without
  representing true motion, and the `q`(position)–`q̇`(register) phase mismatch is what
  inflates the shoulder/elbow `F0` offsets to ±2–2.6 Nm. **Differentiated position is
  the more faithful estimator and remains the default.**

A side observation: the shoulder/elbow constant offset `F0` is poorly conditioned
(swings to −3.30 Nm in the drop-glitches-only run, where `q̇` is unchanged from
delivered) — gravity-vs-offset collinearity on the gravity-loaded joints.

### Impact
**No change to the delivered model** — it remains the best feasible model. Both flags
default off; the delivered recipe reproduces byte-for-byte (unchanged config hash).
The four runs are preserved on disk under distinct hashes for the thesis comparison
table. The remaining lever is unchanged: a **higher-rate (200 Hz) re-collection** —
the paper's rate — which both cleans `q̈` from differentiated position *and* yields a
velocity register with far less relative lag, and is also the prerequisite for
identifying per-link inertias (see THESIS_NOTES "Encoder velocity vs differentiated
position").

### Open questions / assumptions
- A lag-corrected + rescaled measured velocity was considered and rejected as fragile;
  a 200 Hz re-collection solves the root cause instead.

---

## 2026-06-10 — FINAL identified model (stride-1): identification phase complete

**Area:** identification deliverable · `outputs/` artifacts

### The model
Production run at full resolution with the settled recipe:
```
python3 sysid_feasible.py data/traj_run_20260518_143818.csv --no-plot --stride 1 \
  --method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL
```
- **phi:**  `outputs/npy/traj_run_20260518_143818__sysid_feasible-v1-4__cfg-640cb8ef.npy`
- **URDF:** `outputs/urdf/traj_run_20260518_143818__sysid_feasible-v1-4__cfg-640cb8ef__phi_to_urdf-v1-0__cfg-3ef0a00c.urdf`
- Identification: REL mean **0.5997** (= lstsq 0.5899), `[8] Overall: [ALL OK]`,
  status `optimal` (CLARABEL). Masses [0.50, 0.50, 0.263, 0.027, 0.027, 0.027] kg.

### Validation (`compare_urdf_performance.py --friction`, vs `vx300s.urdf`)
| friction-fitted, mean | A baseline | **B (final)** |
|---|---|---|
| RMSE [Nm] | 2.682 | **0.822** (−69.3%) |
| MAE [Nm] | 0.753 | **0.386** |

Beats the manufacturer URDF on every meaningful joint (elbow friction RMSE 3.32 →
0.95, REL 0.18). Matches the stride-5 result (0.816/0.386) → stable across
resolution. Only forearm_roll (−0.04) and wrist_rotate (−0.14 Nm, τ_max 0.07)
slightly trail A — negligible, low-torque, the generic inertia blob.

### Impact
**Identification phase complete.** This URDF is the physically-feasible, data-only
dynamic model that was the prerequisite for the control phase (paper §4–5 ERG +
robust law). Control work is now unblocked — see `docs/PAPER_SUMMARY.md` §10 and
`control/trq.py` (torque→current mapping already in place).

### Open questions / assumptions (carry into the thesis Discussion)
- **Held-in** validation only (identified & validated on the same run). A held-out
  trajectory is needed for a defensible generalisation claim — first control-phase
  to-do, or collect a second run.
- Per-link **inertias ≈ the generic blob** (~0.002): this single trajectory does
  not excite the inertia split; only gravity (`mc`) and friction are identified.
  REL floor 0.59 vs the paper's 0.43 → the gap is excitation, not the solver.
  Implementing the paper's Eq. 11 condition-number-optimised excitation trajectory
  is the main lever to identify more and is itself a thesis contribution.

---

## 2026-06-10 — Tuned the realisation (w2 ↑, γ=0.05): best validated model so far

**Area:** `sysid_feasible.py` (cvxpy path, `--w2`/`--entropic`) · identification/validation

### Problem / Motivation
With the bounded log-det divergence (`--entropic`), even tiny γ raised REL to ~0.65
and would not return to the 0.59 baseline. Diagnosis: the reference blob `P0` has
**zero first moments** (CoM=0), so the divergence pulls the *identifiable* first
moments `mc` (which set gravity, esp. elbow/shoulder) toward 0. The standard params
were tied to the data only through the paper's weak coupling weight `w2 = 5e-3`
(Eq. 16a 2nd term), which the divergence (~7 in cost units) easily overwhelmed.

### Change
No code change — **tuned the paper's own Eq. 16a weight `w2`**. Sweeping `w2` at
γ=0.05 confirmed the mechanism: as `w2` rises, the data reclaims the identifiable
directions and REL falls back to baseline.

| `w2` (γ=0.05) | REL mean | elbow REL | masses 4–6 |
|---|---|---|---|
| 1 | 0.628 | 0.381 | 0.12 |
| 10 | 0.616 | 0.370 | 0.07 |
| **100** | **0.599** | 0.273 | 0.029 |

Chosen operating point: **`--w2 100 --entropic 0.05`** (defaults `--ref-mass 0.5`,
`--ref-inertia 1e-3`). Masses: [0.50, 0.50, 0.23, 0.029, 0.029, 0.029] kg —
non-degenerate (smallest = 29 g, ~30000× above the γ=0 `1e-6`), so `phi_to_urdf`
exports with no `--mass-floor`. The waist/shoulder masses pinning at exactly
`ref_mass` reflects that their absolute mass is essentially **unobservable** from
this trajectory; the shrinking distal masses reflect that those light links are
barely excited. Honest "observable-from-data, unobservable-from-prior" split.

### Evidence — beats the manufacturer baseline AND the γ=0 model
`compare_urdf_performance.py --friction` (stride 5 model), friction-fitted mean:

| | A `vx300s.urdf` | γ=0 SDP | **w2=100, γ=0.05** |
|---|---|---|---|
| RMSE [Nm] | 2.682 | 0.945 | **0.816** (−69.6% vs A) |
| MAE [Nm] | 0.753 | 0.406 | **0.386** |

Best model to date. The **elbow improved** in validation (friction RMSE 1.899 →
0.894) despite slightly higher identification REL — the generic-blob inertias
generalise better than the degenerate γ=0 realisation. Minor regressions vs A on
two very-low-torque joints (forearm_roll −0.04, wrist_rotate −0.14 Nm; the 0.002
blob inertia adds slight spurious inertial torque on wrist_rotate, τ_max 0.07 Nm).

### Impact
Settled identification recipe: `--method cvxpy --entropic 0.05 --w2 100 --solver
CLARABEL`. **Next:** final `--stride 1` run at these settings → export → validate →
the control-ready, validated URDF (unblocks the control phase). Setting `w2` strong
is paper-faithful (it is Eq. 16a's coupling weight); the entropy prior is our
documented deviation (see THESIS_NOTES).

### Open questions / assumptions
- Still **held-in** validation (identified and validated on the same run); a
  held-out run is needed for a defensible generalisation claim.
- Per-link inertias are ≈ the generic blob (`~0.002`) — this trajectory does not
  observe the inertia split; only gravity (`mc`) and friction are well-determined.
  Richer excitation (paper Eq. 11) would identify more; REL floor 0.59 vs paper 0.43.

---

## 2026-06-10 — Convex SDP: first run result + entropic regulariser for URDF export

**Area:** `sysid_feasible.py` (`identify_sdp`, feasibility check, CLI) · identification

### Problem / Motivation
First execution of the convex SDP (`--method cvxpy`, SCS) on
`traj_run_20260518_143818.csv` (`--stride 5`). The solve succeeded
(`status: optimal`) and — crucially — **preserved the fit**: REL mean **0.5921**
vs the unconstrained-lstsq baseline 0.5910, i.e. imposing physical feasibility
cost the fit essentially nothing. This is what *neither* NLP solver achieved
(trust-constr: feasible but REL 0.82; SLSQP: REL 0.60 but infeasible). Two
artefacts remained: (i) the feasibility report flagged links 3–4 as FAIL, and
(ii) per-link masses collapsed toward zero (m≈0 for links 1–2), which **blocks
`phi_to_urdf.py`** (it rejects near-zero mass at line ~162).

### Change
1. **Feasibility-check tolerance (correctness, reporting only).** The friction
   check used a strict `Fv,Fc >= 0` while every other check allows `−1e-4` slack.
   SCS (first-order) returns the constrained `Fv` at ≈ −1e-7, so links 3–4 were
   spuriously failed. Aligned the friction tolerance to `−1e-4`. The saved `phi`
   is unchanged; only the printed verdict is corrected.
2. **Entropic (log-det) regulariser — `--entropic γ`.** Added the model-free
   maximum-entropy term `−γ·Σ_i log det P_i(φ)` to `identify_sdp` (off by default,
   γ=0 ⇒ exact paper Eq. 16). `log det P` is concave; minimising `−log det P` is
   convex, so the problem stays a single global SDP. It resolves the
   *standard*-parameter non-uniqueness (the identified model lives in the base
   parameters; the URDF realisation is non-unique) by spreading inertia into the
   unidentifiable null-space → non-degenerate, physically consistent link masses,
   **without any CAD/reference prior** (thesis-integrity constraint preserved).
3. **`--solver` override** (e.g. CLARABEL). CLARABEL is interior-point and more
   accurate near the feasibility margins than the default SCS.
4. Threaded both options through `run_identification` and into the artifact
   config dict (so the config hash / provenance sidecar capture them).
5. **`phi_to_urdf.py --mass-floor` (validation-only escape hatch).** The γ=0 SDP
   model has collapsed (near-zero) link masses, which the export guard rejects.
   `--mass-floor` clamps only the CoM/parallel-axis *divisor*; because those links
   also have near-zero first moment `mc`, `mc (= m·c)` and `J_O` are preserved
   exactly → the emitted URDF is **torque-faithful** (RNEA depends only on `mc, J_O`)
   even though its masses are not a physical realisation. This lets us sanity-check
   the SDP model's torque prediction *before* committing to the entropic-regulariser
   realisation. Captured in the URDF artifact config hash.

### Run result (entropic γ=0.01, CLARABEL) — regulariser needs to be bounded
The first entropic run **failed informatively**: REL mean rose to 0.97 and link
masses 1–2 exploded to ≈ 8200 kg. `−log det P` is monotone increasing and
**unbounded above**, so it drives the unidentifiable (null-space) masses toward
infinity — and this is *not* fixable by lowering γ (a data-free direction's optimum
is +∞ for any γ>0). Decision (with the thesis author): switch to the **bounded
log-det Bregman divergence** (below).

### Sanity validation of the γ=0 SDP model — beats the manufacturer baseline
Before investing in the realisation, the γ=0 cvxpy model was exported (via
`phi_to_urdf --mass-floor`, the masses being ~1e-6 at the LMI's `EPS` margin) and
validated with `compare_urdf_performance.py --friction` on the identification run:

| mean over 6 joints | A `vx300s.urdf` | B (γ=0 SDP) | B improvement |
|---|---|---|---|
| rigid-body RMSE | 3.025 Nm | **1.330 Nm** | −56% |
| friction-fitted RMSE | 2.682 Nm | **0.945 Nm** | −65% |
| friction-fitted MAE | 0.753 Nm | **0.406 Nm** | −46% |

B beats A on **every** joint; the friction-fitted REL 0.582 matches the
identification REL 0.59 → the URDF round-trip is faithful. **This is the best
model so far** (prev. best trust-constr MAE 0.708). The degenerate masses do not
hurt torque prediction (RNEA uses only base parameters), so the SDP model is
validated and only the *realisation* (control-ready masses) remains. Caveat:
training-set error, not held-out — the excitation/held-out question still stands.

### Bounded log-det divergence — implemented
Replaced the unbounded `−γ·Σ log det P` with the convex log-det Bregman divergence
`+γ·Σ_i [ tr(P0⁻¹·P_i) − log det P_i ]` toward a generic isotropic reference
`P0 = diag(ref_inertia·I₃, ref_mass)` (defaults 1e-3 kg·m², 0.5 kg; `--ref-mass`,
`--ref-inertia`). It is ≥0, convex, and minimised at `P_i = P0`, so the `tr` term
bounds masses from above (fixing the explosion) while `−log det P` lifts them off
zero. Verified: an otherwise-free pseudo-inertia converges to `diag(P0)`.
**Next:** run `--entropic γ` (start ~0.1, CLARABEL), tune the smallest γ that gives
plausible non-degenerate masses while REL stays ≈ 0.59, then export (no mass-floor
needed) → re-validate → this becomes the control-ready URDF.

### Evidence
- SDP run: `status: optimal`, cost 2517.752; REL `[0.69 0.76 0.16 0.33 0.61 1.00]`
  mean 0.5921 (≈ lstsq 0.5910). All LMI / Iᶜ / triangle / mcy-sign constraints
  pass; the only FAILs were the friction-tolerance artefact above.
- `cp.log_det(P)` with a 4×4 PSD `P` solves under CLARABEL (`status: optimal`) —
  the regulariser is DCP-valid and supported by the installed backend.

### Impact
- Identification default behaviour is unchanged (γ=0). Adding `entropic`/`solver`
  to the config dict changes the config hash, so a re-run of any method recomputes
  rather than cache-hitting; previously-saved artefacts remain on disk.
- **Next:** re-run cvxpy with a small `--entropic γ` (+ `--solver CLARABEL`),
  tune γ for the smallest value giving plausible masses, then `phi_to_urdf.py`
  (now unblocked) → `compare_urdf_performance.py` vs the 0.753 Nm baseline.

### Open questions / assumptions
- γ is a free hyper-parameter with no data-driven optimum; it trades fit against
  inertia "fullness". Choose the smallest γ that lifts masses off zero so the fit
  (REL) is essentially unperturbed — to be reported in THESIS_NOTES.
- `log det P` mixes parameter scales (mass O(1), moments O(1e-3)); the LMI margin
  `EPS=1e-6` bounds det away from 0, so the term is finite, but the resulting mass
  distribution is an entropy-maximising realisation, **not** a claim about true
  per-link masses (only the base parameters are claimed identified).

---

## 2026-06-10 — Re-identified with corrected units (v1.4): result & diagnosis

**Area:** identification result (`sysid_feasible.py` v1.4) · validation.

### What
Re-ran identification on `data/traj_run_20260518_143818.csv` with the corrected
mA→Nm units (same config as the prior v1.1/v1.2 runs: `fs=50, fc=10, stride=1,
trust-constr, w1=1, w2=5e-3`). Artifact:
`outputs/.../traj_run_20260518_143818__sysid_feasible-v1-4__cfg-f8c3b062.{npy,urdf}`.

### Evidence (RNEA torque-prediction vs measured, all with correct units)
| Model | rigid-body mean RMSE | friction-fitted mean MAE |
|---|---|---|
| old v1-1 (mis-scaled units) | 14.73 Nm | 8.75 Nm |
| **new v1-4 (correct units)** | 4.49 Nm | **0.708 Nm** |
| manufacturer `vx300s` | 3.03 Nm | 0.753 Nm |

- Identified masses now physical: `[0.80, 0.80, 0.51, 0.30, 0.21, 0.11] kg`;
  all feasibility constraints (mass>0, I^c PD, triangle ineq., mass-moment signs)
  pass.
- v1-4 **beats the manufacturer URDF on mean MAE** (0.708 vs 0.753 Nm) — already
  competitive on typical error.

### Open problem — optimizer non-convergence
Both identification stages stopped with *"maximum number of function evaluations
exceeded"*; final constrained REL (mean **0.822**) is **worse than the
unconstrained least-squares baseline (0.590)** and far from the paper's ~0.43.
Symptom in validation: v1-4's RMSE (4.49) > baseline (3.03) despite better MAE —
i.e. large `q̈`-driven torque spikes (waist RMSE 6.7 Nm against a ±0.73 Nm real
signal). Interpretation: gravity/friction parameters are well-identified (→ good
MAE), but **inertia parameters are poorly determined** because `trust-constr`
with 2-point FD constraint Jacobians on a 242k-row, 136-variable problem exhausts
its evaluation budget after few real iterations (`delta_grad == 0.0` warning =
quadratic data term confusing the quasi-Newton Hessian).

### Next
Solver-convergence experiments (smaller problem via `--stride`, `SLSQP`, possibly
raising `maxiter` / zeroing the data-term Hessian). Target: constrained REL ≤ the
0.59 unconstrained value, closing the inertia-driven RMSE gap to baseline.

---

## 2026-06-10 — Added convex SDP solver (`--method cvxpy`, paper-faithful)

**Area:** `sysid_feasible.py` — new `identify_sdp()` + `run_identification`
branch + `--method cvxpy`. Requires `pip install cvxpy`.

### Motivation
Both SciPy NLP solvers fail on the non-convex reformulation (trust-constr stalls
feasible-but-poor; SLSQP fits but goes infeasible / collapses masses). The
paper's problem (Eq. 16) is actually a **convex SDP** — solving it as such gives
a feasible *and* optimal solution.

### Change
`identify_sdp()` formulates Eq. 16 in CVXPY: convex quadratic objective (data
term via the Gram matrix `WᵀW`, so size is independent of #samples) subject to
the **pseudo-inertia LMI** `P_i(phi) ⪰ 0` per link — a single constraint that
subsumes mass>0, `I^c ≻ 0` (16c) and the triangle inequalities (16d–f), per
Wensing et al. [53] — plus `Fv,Fc ≥ 0` and the mass-moment sign constraints
(16i). `trust-constr`/`SLSQP` paths are unchanged; `method` is part of the
config hash so SDP runs get distinct artifact names.

### Justification / honesty
No CAD/reference-model prior is used — feasibility is resolved by physics alone,
so the method remains valid for a robot with no pre-existing model (the thesis
premise). See `THESIS_NOTES.md` "Solver…" for the full argument and the
base-vs-standard representation nuance.

### Impact / to run
`pip install cvxpy`, then
`python3 sysid_feasible.py data/<run>.csv --no-plot --stride 5 --method cvxpy`.
Expected: `[8]` feasibility `ALL OK` *and* REL near the 0.59 unconstrained
baseline (i.e. both fixes at once). Then export + validate as usual.

---

## 2026-06-10 — SLSQP vs trust-constr; non-identifiability is the real blocker

**Area:** identification solver (`sysid_feasible.py`) · diagnosis.

### What
Re-ran identification with `--stride 5 --method SLSQP` to test the
solver-convergence hypothesis. SLSQP reached the data-optimal fit but produced a
physically degenerate, **infeasible** model.

### Evidence
| run | REL (mean) | feasibility | masses [kg] |
|---|---|---|---|
| unconstrained lstsq | 0.591 | n/a | n/a |
| trust-constr v1-4 (`cfg-f8c3b062`) | 0.822 | ALL OK | [.80,.80,.51,.30,.21,.11] |
| SLSQP v1-4 (`cfg-5e871841`) | 0.602 | Links 2,5,6 FAIL | [.80,.81,.33,.02,.0004,.003] |

Conditioning check (column-normalised base regressor, stride-20 subsample):
**numeric rank 58/78 with a clean singular-value cliff; cond over the
identifiable subspace ≈ 197** (good). The `waist`/`shoulder` mass columns are
exactly zero (those masses are non-identifiable).

### Conclusion
Excitation/conditioning of the **base parameters** is fine — *not* the
bottleneck. The blocker is the classic **base-vs-standard non-identifiability**:
only the 58 base combinations are determined by data, so the per-link masses /
inertias are free, and a fit-chasing solver (SLSQP, weak `w2` coupling) collapses
the distal masses to ~0 and slips infeasible. trust-constr stays feasible but
stalls. The paper's SDP/YALMIP solver gets both feasible *and* optimal because
the LMI feasibility set is convex. → Best usable model remains **trust-constr
v1-4**; the fix is better regularisation of the standard parameters and/or a
convex (CVXPY/SDP) solver. See `THESIS_NOTES.md` "Solver…".

---

## 2026-06-10 — Added paper summary; noted paper-vs-code discrepancies

**Area:** documentation (`docs/PAPER_SUMMARY.md`, `Paper.txt`, `CLAUDE.md`).

Condensed the source paper into `docs/PAPER_SUMMARY.md` (method, feasibility
constraints, limits, and the benchmark REL values to beat) so the full text need
not be loaded for context. While summarising, two discrepancies between the
paper and the current code surfaced — flagged here for follow-up:

1. **Link-length split.** Paper: `L3=0.1964, L4=0.10362`; code:
   `L3=0.21981, L4=0.08021`. **Benign** — both give `L3+L4 = 0.30002`, which is
   what enters DH `d4`; the individual values don't otherwise appear.
2. **First-mass-moment sign constraints (Eq. 16i).** Paper uses mixed axes:
   `m2·X2>0, m3·Y3>0, m4·Z4<0, m5·Y5>0, m6·Z6>0`. The `sysid_feasible.py`
   docstring states an all-`y` convention (`m2y2>0 … m6y6>0`). **To verify:**
   whether the implemented constraint matches the paper's axes; a wrong axis
   would bias the identified centre-of-mass and is a plausible contributor to
   the poor identified URDFs.

---

## 2026-06-10 — Corrected effort→torque conversion (mA, with dual-motor joints)

**Area:** data pipeline (`sysid_feasible.py: load_and_filter`) · affects every
downstream result (identification *and* validation).

### Problem
The measured joint efforts were converted to torque with
`EFFORT_SCALE = STALL_TORQUE / 100`, i.e. the `*_effort` columns were assumed to
be a **percentage of stall torque** (−100…100 %). The recorded values, however,
reach ±400–1200, far outside that range. As a result the "measured torque" was
physically impossible — e.g. **122 Nm on the shoulder**, whose motors stall at
~10.6 Nm. Any validation against these numbers was meaningless, and the
identification was fitting parameters to mis-scaled targets.

### Root cause
The `*_effort` field published by the Interbotix driver is the Dynamixel
**present current in milliamps (mA)** of the *master* motor, not a percentage.
The correct conversion uses the motor torque constant (τ = k_t · I):

$$\tau_{\text{joint}}[\text{Nm}] = \frac{\text{effort}[\text{mA}]}{1000}\,\cdot\,k_t\,\cdot\,n_{\text{motors}},\qquad k_t = 2.409\ \text{Nm/A (XM540-W270, geared)}$$

The shoulder and elbow are **dual-motor** joints (master + mirrored shadow
motor); the driver reports only the master's current, so their joint torque is
doubled (`n_motors = 2`). The constants `k_t = 2.409` and the current unit
appear in `control/trq.py`.

### Evidence
On near-static samples the measured torque must equal the URDF gravity term
`G(q)`. Regressing raw effort against Pinocchio's `computeGeneralizedGravity`
for `vx300s.urdf` gave a **shoulder slope of 0.00480 Nm/unit = 2 × 0.002409**,
i.e. exactly `2·k_t/1000`. This simultaneously confirms (a) the unit is mA and
(b) the shoulder carries two motors. After the fix, measured torque ranges
become physically sensible:

| joint | before (max\|τ\|) | after (max\|τ\|) |
|---|---|---|
| shoulder | 122.8 Nm | 5.58 Nm |
| elbow | 81.2 Nm | 3.69 Nm |
| waist | 12.4 Nm | 0.73 Nm |

### Change
```python
# sysid_feasible.py
TORQUE_CONSTANT  = 2.409                          # Nm/A (XM540-W270, geared)
MOTORS_PER_JOINT = np.array([1, 2, 2, 1, 1, 1])   # waist, shoulder, elbow, fr, wa, wr
EFFORT_SCALE     = (TORQUE_CONSTANT / 1000.0) * MOTORS_PER_JOINT   # mA → Nm
```
`PIPELINE_VERSION` bumped `1.3 → 1.4` so cached artifacts recompute.

### Impact
- All previously identified URDFs/`phi` vectors were fit to mis-scaled torque
  and must be **regenerated** before they can be trusted.
- Enabled a meaningful model-validation comparison (see below): with correct
  units, `vx300s.urdf` predicts the recorded torques to ~0.75 Nm mean MAE.

### Open questions / assumptions to verify
- **Elbow motor count** set to 2 from the ViperX-300 spec; the static data here
  was too friction-noisy to confirm independently. Verify against the hardware.
- **wrist_rotate** is an XM430 (slightly different `k_t`); load is ±0.05 Nm, so
  the error is negligible, but the single global `k_t` is an approximation.

---

## 2026-06-10 — Added URDF model-validation harness

**Area:** new tooling (`compare_urdf_performance.py`).

### What
A standalone script that quantifies how well a URDF's dynamic parameters
reproduce reality, and compares two URDFs head-to-head. For each sample of a
recorded trajectory it runs Pinocchio inverse dynamics (RNEA) on `(q, q̇, q̈)`
and compares the predicted torque against the measured torque, reporting
per-joint **RMSE / MAE / relative error / R²**. It reuses
`sysid_feasible.load_and_filter` so filtering, derivatives and effort scaling
are identical to the identification pipeline.

### Why
Identification reports its own training residual, but that does not tell you
whether the resulting URDF is a *good physical model*. RNEA torque prediction is
the standard, model-agnostic way to validate inertial parameters, and it lets us
rank a candidate URDF against the manufacturer's `vx300s.urdf` baseline.

### Notes
- Plain RNEA is rigid-body only (gravity + inertia + Coriolis); joint friction
  appears as an unmodelled residual. The `--friction` flag additionally fits a
  per-joint viscous+Coulomb+offset friction term (independently per model) to
  isolate rigid-body quality.
- Joints are matched **by name**, so the harness handles both arm-only URDFs
  (`nq = 6`) and the full robot incl. gripper (`nq = 10`).

### First finding
With corrected units, the manufacturer baseline (`vx300s.urdf`, ≈0.75 Nm mean
MAE) substantially outperforms the current identified URDF (≈2.4 Nm), whose
parameters are implausible (near-zero link masses; 28 Nm waist error against a
≈0.7 Nm real load). Investigating the identification quality is the next step.

---

## Entry template

```markdown
## YYYY-MM-DD — <short title>

**Area:** <file(s) / subsystem> · <scope of impact>

### Problem / Motivation
<what was wrong or missing, and why it matters>

### Change
<what was changed; code snippet if useful>

### Evidence
<measurements, plots, or reasoning that justify the change>

### Impact
<what results/conclusions this affects; what must be re-run>

### Open questions / assumptions
<anything unverified that a reader/examiner should know>
```
