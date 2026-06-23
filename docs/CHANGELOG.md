# Engineering Changelog

A running, thesis-oriented record of **substantive changes to the code and
methodology** — what changed, *why*, the evidence behind it, and how it affects
results. The intent is that each entry can be cited or paraphrased directly in
the dissertation's "Implementation" / "Methodology" discussion.

Entries are newest-first. Each follows the template at the bottom of this file.

---

## 2026-06-23 — Default command rate 50 Hz → ~6.7 Hz (stride 30): fix comms stalls

**Area:** `collect_200hz.sh` default STRIDE

### Problem / Motivation
Streaming position commands at 50 Hz (stride 4) for a 0.5 Hz-bandwidth trajectory
floods the WSL2/usbipd link and triggers ~100–280 ms comms stalls (schedule
shifts → brief catch-up). A 60 s smoke at `--stride 30` ran with 0 stalls and
passed verification, but the full-run command omitted the flag and re-stalled —
the fix wasn't persisted.

### Change
Default `STRIDE 4 → 30` (~6.7 Hz commands, still 13× the trajectory bandwidth).
The servo's time-based profile interpolates between the sparser waypoints; data is
still recorded at 200 Hz (identification uses the measured motion, not the
commands), so fidelity is unaffected. A comms gap shorter than one ~150 ms command
interval no longer forces a schedule shift.

### Evidence
Smoke at stride 30: `400 commands in 59.9 s, 0 stalls absorbed`, Collection OK,
verification PASS (198 Hz recording). Excitation trajectory itself is clamp-safe
(shoulder stays in [−0.5,+0.3] during the run; sub-−1.3 only at the sleep pose).

### Impact
Collection now completes. Run `bash collect_200hz.sh --design <npz>` (stride 30
default) for the full 900 s.

---

## 2026-06-23 — Reduce excitation speed: full-speed motion moved the (clamped) base

**Area:** `run_trajectories.py` velocity/acceleration limits

### Problem / Motivation
At full speed the extended arm's reaction torque moved the clamped platform
(observed on hardware). That risks tipping AND violates the **fixed-base
assumption** the identification depends on — a moving base corrupts the torque
data regardless of excitation quality. The waist (fast yaw of the extended arm)
was the dominant driver (accel amplitude 8.6, highest of all joints).

### Change
Cut per-joint limits: `VEL_MAX 3.14→[1.8,2.2,2.2,2.5,2.5,2.5]`,
`ACCEL_MAX 10→[3.5,5,5,6,6,6]` — waist hardest, shoulder/elbow retain meaningful
acceleration for inertia/Coriolis excitation.

### Evidence
Re-optimised: waist accel 2.1 (was 8.6), shoulder/elbow ~4.3; feasible, band ✓,
rest ✓, **corr 0.569 unchanged** (first-moment decorrelation is speed-independent).
cond(Φ_b) rose ~150→~239 — the cost is weaker inertia excitation (lean on the
`Ia·q̈` motor-inertia term at identification instead).

### Impact
Keeps the base fixed → valid data. Re-run the design; tune limits further if the
base still moves (or back up if it's rock-solid).

---

## 2026-06-23 — Centre the q0 operating points (fix lopsided waist motion)

**Area:** `run_trajectories.py`

### Problem / Motivation
The free q0 offsets parked dynamically-degenerate joints at extremes — above all
the WAIST (whose angle does not affect the dynamics at all, base vertical-axis
symmetry, so cond(Φ_b) is indifferent to it): q0_waist = −1.41, with the waist
sweeping only the negative half. Lopsided, unintuitive motion the user flagged on
hardware.

### Change
Constrain |q0 − joint_centre| ≤ Q0_MAX (0.30 rad) for every joint. The
gravity-relevant offsets (shoulder ≈+0.2, elbow ≈−0.1) sit inside the band, so
conditioning is essentially unchanged; the degenerate joints are pulled back to a
symmetric, centred sweep.

### Evidence
Re-optimised: waist q0 +0.30 (was −1.41), shoulder −0.02, start velocity ✓ rest,
band ✓, cond(Φ_b) ≈166–178 (within noise of the unconstrained value).

### Impact
Trajectory starts and sweeps about the joint centres. Re-run the design to save
the centred version (the previous .npz still has the off-centre waist).

---

## 2026-06-23 — Rest-to-rest excitation (fix start lurch) — caught on hardware

**Area:** `run_trajectories.py` · safety-critical

### Problem / Motivation
The first smoke run of the optimised design **lurched violently on engage** and
the user aborted it. Cause: the Fourier trajectory's velocity at t=0 is Σ_k a_k,
and only velocity *amplitude* was constrained — the saved design started at
**2.5–2.8 rad/s** (waist/shoulder/wrist_rotate) from a standstill.

### Change
Added **rest-to-rest boundary constraints** to the optimiser: Σ_k a_k = 0 (zero
boundary velocity) and Σ_k k·b_k = 0 (zero boundary acceleration) per joint —
because the motion is periodic these hold at start, end, and every period. The
feasible-start init is projected onto them; the feasibility verdict includes them.
A `print_stats` readout and a hard `main()` **safety gate** abort any design with
|q̇(0)| > 0.05 rad/s (also blocks stale pre-fix designs).

### Evidence
Re-optimised design: q̇(0) max |0.000| rad/s ✓, feasible, band ✓, cond(Φ_b) ≈176
(slightly up from 140 — the equality constraints cost some freedom).

### Impact
Excitation now eases out of and back into standstill. The previously-saved design
is invalidated (lurches) — re-run before collecting.

---

## 2026-06-23 — Excitation optimiser: reliable feasibility (feasible start + buffer)

**Area:** `run_trajectories.py` optimiser robustness

### Problem / Motivation
With the tighter shoulder floor (−1.25) the cond(Φ_b) optimiser stopped finding a
strictly-feasible design — restarts started from infeasible random points, one
diverged (cond 8×10⁵), and the saved design violated limits by 0.5 rad (would trip
the hardware box gate).

### Change
(1) Each restart now starts from `scale_to_limits` (amplitudes feasible for all
pos/vel/accel) with q0=HOME — SLSQP begins feasible and only maintains it.
(2) Inward limit buffer (0.04 rad, 0.10 rad/s) so the full-rate trajectory's
between-grid peaks stay within true limits. (3) Feasibility verdict evaluated
against the TRUE limits + band (the optimiser constraints are buffered), matching
the hardware gates.

### Evidence
Now returns `feasible=True` with all joints ✓ and band ✓ inside true limits;
cond(Φ_b) ≈120–150 (improves with iterations), shoulder·elbow corr ≈0.57 (the
clamp-floored workspace ceiling, still richer/better-conditioned than old seed-42
data at 236).

### Impact
Design step is now dependable. Re-run → smoke → collect.

---

## 2026-06-23 — Shoulder floor −1.25 (anti-tip clamp collision) + stale-design gate

**Area:** `run_trajectories.py` · new hardware collision constraint

### Problem / Motivation
Below shoulder ≈ −1.3 the shoulder link collides with the anti-tip clamp securing
the base (observed on hardware). The float sweep and the first optimised design
reached shoulder −1.78 — unsafe.

### Change
Raised `LIMITS_LO[shoulder]` −1.78 → **−1.25** (0.05 rad below the −1.3 collision
point for execution-overshoot margin). Added a **box-limit safety gate** in
`main()`: a design whose raw trajectory exceeds the current box limits (e.g. a
design saved before this floor) aborts instead of being silently clipped into a
distorted path.

### Evidence
The prior `outputs/excitation_design.npz` (shoulder −1.782) now trips the gate
(box violation +0.532 rad → ABORT). Compiles.

### Impact
The first optimised design is **invalidated** — must re-run
`run_trajectories.py --design-only --save outputs/excitation_design.npz`. The
tighter shoulder range may raise the achievable shoulder·elbow corr (less
decorrelation room); judge on the re-run.

---

## 2026-06-23 — collect_200hz.sh --design: replay a vetted excitation deterministically

**Area:** `collect_200hz.sh` + `run_trajectories.py` (`--save`/`--load`)

### Problem / Motivation
The cond(Φ_b) optimiser is now multi-minute and multistart-random, so re-running
it live inside the gated collection would be slow and non-reproducible (the
collected trajectory wouldn't match the one vetted offline).

### Change
`run_trajectories.py` gained `--save`/`--load` (.npz of a,b,q0). `collect_200hz.sh`
gained `--design PATH`, which passes `--load` to the trajectory script (skipping
the optimiser) so collection replays the exact approved design; the legacy
`--seed` path is unchanged when `--design` is omitted. The band safety gate and
all collection gates still run.

### Evidence
`bash -n` clean; `--load outputs/excitation_design.npz` reproduces the saved
design (cond(Φ_b) 83.6, shoulder·elbow corr 0.362, band ✓).

### Impact
Design-once-offline → collect-deterministically. Workflow: vet with
`run_trajectories.py --design-only --save`, then
`collect_200hz.sh --design <npz>`.

---

## 2026-06-23 — Coupled shoulder–elbow collision band → excitation constraint

**Area:** `run_trajectories.py` · turns the measured reachable envelope into the
optimiser constraint · `control/pd_grav_control.py --float` produced the data

### Problem / Motivation
Opening the shoulder range for the new excitation needs the real collision-safe
shoulder/elbow set, not an independent box (the box's shoulder-up + elbow-extended
corner collides). Mapped it by hand in float mode.

### Change
From the float sweep (`data/float_envelope_20260623_113036.csv`, full-area sweep,
confirmed by the user) the safe set is a **diagonal band** elbow ≈ −0.7·shoulder
+ offset: fitted `elbow_max=−0.71·sh+0.30`, `elbow_min=−0.67·sh−0.26`, ~0.56 rad
elbow freedom at fixed shoulder. Encoded as `SH_EL_BAND_HI/LO` (+0.08 rad inward
margin), enforced on the sampled trajectory in the optimiser, reported in
`print_stats`, and a hard **safety gate** aborts collection if a design breaches
it (the per-joint clip can't enforce a coupled constraint). Shoulder/elbow box
limits opened to the swept extremes (shoulder [−1.78,1.38], elbow [−1.36,1.58]).

### Evidence
A partial design run (2 restarts, 110 iters) inside the band: cond(Φ_b) 236→**89**,
shoulder·elbow m·c_y corr **+0.62→+0.42**, band ✓ (slack +0.08). Not yet strictly
feasible (minor limit overshoots — needs more iterations).

### Impact
Trajectory redesign demonstrably reduces the first-moment collinearity within the
real safe set. **Open question:** the band is itself a near-linear shoulder–elbow
coupling (≈0.56 rad independent freedom), so corr may plateau ~0.4 — a
**workspace-geometry ceiling** on first-moment separability. If the converged
design and re-identified model still lump, the next lever is a known payload to
break the mass symmetry (not trajectory design). Next: full converged design run
→ collect → re-identify.

---

## 2026-06-23 — Float/compliant mode in pd_grav_control (reachable-envelope mapping)

**Area:** `control/pd_grav_control.py` · `--float` + `--go-home` · supports the
shoulder/elbow coupled-limit design for the new excitation

### Problem / Motivation
Designing the new excitation needs the *real* collision-safe shoulder/elbow
reachable set (the lumping fix requires opening the shoulder-forward range, but
only within a coupled envelope, e.g. elbow→1.0 only when shoulder≈−1.0). Best
measured empirically by hand-moving the arm.

### Change
Added `--float`: pure gravity-compensation, **no position term** — the arm holds
its own weight but is freely backdrivable by hand. Disables the position/
tracking-error kill (the operator moves it far on purpose); soft limits + current
caps + a raised velocity backstop remain. `--go-home q…` moves the arm to a start
pose in position mode (timed profile, raw interface) before engaging. On stop it
writes a **CSV of recorded joint angles** (`data/float_envelope_*.csv`) and prints
the shoulder/elbow ranges swept. Reuses the tested current-mode handoff/parking.

### Evidence
Byte-compiles; reuses the validated gravity + safety path (only the position term
and the position kill are gated off in float). Hardware run pending (user).

### Impact
Enables recording the real reachable envelope to fit the coupled shoulder/elbow
limit, which then becomes a constraint in the cond(Φ_b) excitation optimiser.

---

## 2026-06-23 — Implement the cond(Φ_b) excitation objective (+ free q0, multistart)

**Area:** `run_trajectories.py` · fixes the wrong-objective defect from the audit
entry below

### Change
Replaced the excitation optimiser's objective `cond([q̇; q̈])` (kinematic) with
**`cond(Φ_b)`** — the condition number of the actual base identification
regressor (paper Eq. 11), built via `sf.regressor_fast` + `find_base_parameters`
over one fundamental period. Made the per-joint offsets **q0 free design
variables** (paper does this; gravity conditioning depends on the operating
point). Added **multistart** (`--restarts`, q0 jittered per restart), `--maxiter`,
`--design-only` (optimise + report without ROS/hardware), and a startup readout
of achieved `cond(Φ_b)` and the shoulder·elbow `m·c_y` correlation. The hardware
import is now lazy so the design runs ROS-free; the initial move targets the
trajectory start (= q0), not HOME.

### Evidence
A short (24-sample grid, 18-iter, single-restart) check drops `cond(Φ_b)` 807 → 116.
**Caveat:** the shoulder·elbow `m·c_y` correlation barely moved (+0.640 → +0.631)
on that partial/infeasible run — global conditioning ≠ targeted first-moment
decorrelation. A full feasible run (higher `--maxiter`/`--restarts`) is needed to
judge; if the correlation does not drop, add an explicit shoulder/elbow column-
correlation penalty on top of the paper's criterion.

### Performance / who-runs
One objective eval ≈ 0.25 s (Python-loop regressor over 80 samples); SLSQP
finite-diff gradient makes it ~17 s/iteration → the full optimise is multi-minute.
**User runs it** (`python3 run_trajectories.py --design-only` first to vet the
design, then without `--design-only` to collect). Defaults: grid 80 samples,
6 restarts, maxiter 200.

### Impact
Excitation design now targets the regressor conditioning that governs first-moment
identifiability. Next: a full design run, judge `cond(Φ_b)` + shoulder/elbow corr,
then (if needed) add the targeted penalty, then recollect + re-identify.

---

## 2026-06-23 — Excitation-trajectory audit: optimiser conditioned the wrong matrix

**Area:** `run_trajectories.py` (excitation design) · root-cause audit vs the
paper before re-identification · analysis only (fix follows)

### Problem / Motivation
The hardware first-moment lumping (entry above) showed the shoulder/elbow gravity
split is unidentifiable from our data. We audited the excitation generator against
the paper (§3.2, Eq. 7 / Eq. 11) to find why.

### Finding
The kinematic form is faithful to the paper (Fourier, Δf=0.1 Hz, N_f=5,
900 s @ 200 Hz, SLSQP ≈ fmincon). But the **conditioning objective is wrong**:
the paper minimises `cond(Φ_b)` — the condition number of the actual **base
identification regressor** (Eq. 11, which contains the gravity/angle columns) —
whereas `run_trajectories.py` minimises `cond([q̇; q̈])`, a purely **kinematic**
velocity/acceleration matrix. First-moment (gravity) conditioning was therefore
never optimised, leaving the shoulder/elbow first moments collinear → lumping.

### Evidence (measured on the recorded data)
cond(Φ_b) = 236 (200 Hz run) / 2 765 (May run); shoulder·elbow `m·c_y` column
correlation +0.62 (200 Hz) — the collinearity behind the lumping. Secondary
deviations: q0 fixed at HOME (paper frees it); SLSQP does not converge (degenerate
cond→1.0 on re-run); shoulder forward range capped at +0.17 rad (paper ±1.76);
accel limit 10 vs ~200–500 rad/s². Full table in THESIS_NOTES (2026-06-23
excitation-trajectory audit).

### Impact
Reframes re-identification: the model wasn't wrong for lack of a solver — it was
identified from data that never made the first moments separable. Fix (next):
swap the objective to cond(Φ_b), free q0, open the shoulder range, multistart the
optimiser. Re-run: regenerate the excitation, recollect, re-identify.

---

## 2026-06-23 — Held-out cross-validation (post-DH-fix) + a first-moment lumping diagnosis

**Area:** identification validation · `compare_urdf_performance.py` (held-out) +
`control/pd_grav_control.py` (hardware gravity-source swap) · no code change

### Problem / Motivation
The control phase stalled on a ~46 mm EE error dominated by shoulder gravity
over-prediction. Before re-collecting data we tested whether the *existing* v1-5
(post-DH-fix, `cfg-9ef2c992`) models already generalise, and which is the better
gravity source for the controller. CLAUDE.md gates the control phase on a
*validated* URDF, defined by low **held-out** torque-prediction error — which we
had not yet measured for the post-fix models.

### Experiment & result
**(1) Held-out torque cross-validation** (`--friction --drop-glitches`), each
model predicting the dataset it was NOT identified from. Friction-fitted mean
RMSE: 200 Hz→May 1.222 Nm, May→200 Hz **0.303 Nm**; both crush factory
(2.066 / 0.719 Nm) — the DH fix is re-confirmed fully held-out. But the **shoulder
is asymmetric**: May→200 Hz shoulder R² **+0.81** (RMSE 0.76 vs no-model 1.69),
while 200 Hz→May shoulder R² **−1.9** (worse than commanding zero — it
over-predicts out-of-sample). So the 200 Hz shoulder defect is **real and
held-out**, not a held-in artifact (likely high-speed rotor inertia `Ia·q̈`
absorbed into the shoulder first moment).

**(2) Hardware gravity-source swap** — ran the controller on the May URDF
gravity. The two models AGREE on shoulder gravity (−241 vs −239 mA) but disagree
massively on **elbow** (May −94 vs 200 Hz −602 mA; measured holding −683). On
hardware the May model droops the elbow **−0.349 rad** (under-compensates 7×)
while the shoulder droop is unchanged. The swap made things worse.

### Interpretation
Classic **first-moment null-space lumping**: the shoulder and elbow first-moment
regressor columns are nearly collinear for these (shared-excitation) trajectories,
so the optimiser can shuffle gravity between the two joints without hurting the
*total* chain-torque fit. Held-out cross-val doesn't catch it because both
datasets share the excitation structure. The 200 Hz model happens to split it
physically (elbow −602 ≈ measured), the May model doesn't.

### Impact / decisions
- **Controller stays on the 200 Hz model** (best per-joint gravity split: elbow
  correct, shoulder slightly over). The May swap is reverted.
- **Re-identification is now justified with a specific target:** design excitation
  that **decorrelates shoulder vs elbow gravity** (move one while the other is held
  at varied fixed angles, slowly) to break the collinearity. Generic
  "re-identify the shoulder" was the wrong framing.
- No re-run needed yet; next is excitation-trajectory design.

### Open questions / assumptions
- Friction is refit on the eval data (gravity/inertia are the held-out part).
- The ~1.7× constant static over-prediction (stiction, CHANGELOG 2026-06-13) is a
  separate physical effect, not addressed by re-identification.

---

## 2026-06-23 — Integral term (PID + gravity comp) to kill the steady-state droop

**Area:** `control/pd_grav_control.py` · roadmap stage 2 (accuracy) · opt-in
`--ki-scale`

### Problem / Motivation
Pure PD + gravity-comp leaves a **steady-state droop** because the gravity model
is imperfect (the shoulder first-moment over-prediction, the forearm_roll
constant-FF residual). The 2026-06-18 end-to-end run held the EE with ≈46 mm
error, dominated by a +0.105 rad shoulder droop. A re-confirmation hold today
(after the velocity glitch fix — held >40 s, no false kill) showed droop
`[0, +0.058, −0.004, −0.216, −0.153, −0.003]` rad. An integral term is the
standard, lowest-effort way to drive that residual → 0 without needing a better
model.

### Change
Added an optional integral term to the control law:
`u = Kp·err + Ki·∫err·dt − Kd·q̇ + α·G_mA(q)`. Controlled by `--ki-scale`
(default **0 = off**, preserving the verified PD+G behavior; sweep up like α).
Per-joint `KI_BASE` (larger on the proximal gravity joints) and an anti-windup
clamp `I_CAP` bounding each joint's integral *contribution* in mA. Integration is
**gated** until after the setpoint ramp (`frac≥1`) and the grace window, so it
never winds up on the deliberately large, shrinking engage-transient error. The
integral contribution is logged (new cols 32:38) for analysis.

### Evidence
Byte-compiles; logic is offline-reasoned (no hardware yet). Anti-windup cap keeps
a saturated integral well under the gravity load, so it cannot by itself overpower
the arm. Hardware ki-sweep pending.

### Impact
Roadmap stage 2 (the biggest EE-accuracy win) is implemented and ready to sweep
on hardware. Expected: droop → ~0 at the held pose, shrinking the 46 mm EE error.
Re-run: a `--ki-scale` sweep (0 → 0.5 → 1.0) at the standard pose, compare droop.

### Open questions / assumptions
- Integral gains are first guesses; the sweep calibrates them. Watch for slow
  oscillation (too-high Ki against the noisy Dynamixel velocity / Kd damping).
- Also cleaned up unresolved git merge-conflict markers accidentally committed
  into this changelog (the 06-18 vs 06-14 entry seam); both entry sets retained.

---

## 2026-06-18 — END TO END: commanded the EE to a target position on the real arm

**Area:** `control/pd_grav_control.py` (velocity glitch-rejection fix) · full goal
pipeline validated on hardware

### Result
Ran the complete pipeline on the real arm: target EE position [0.30, 0, 0.40] m →
`ik_solve` → joint targets → `pd_grav_control --hold-pose` held the EE there for
**25 s on URDF gravity**. Kinematics are exact (commanded setpoint FK = target to
<0.1 mm). Real-world EE accuracy at the settled pose: **|error| ≈ 46 mm**
([−30, +17, +30] mm), dominated by a **+0.105 rad shoulder droop** — i.e. the
*dynamic* shoulder-gravity over-prediction (first-moment error), not kinematics.

### Fix (this entry's code change)
The 25 s hold ended on a **false velocity kill**: a single sample read 100 rad/s
on all joints (a timer-burst near-zero `dt` makes Δq/dt explode). Added glitch
rejection to the velocity filter — samples with `dt < 4 ms` or a non-physical
estimate (>8 rad/s; the arm tops out ~3–4) reuse the last filtered velocity and
don't advance the filter state. Hold itself was stable (elbow droop +0.004,
jitter 0.010 rad over 25 s).

### Impact
The thesis goal — hold via URDF gravity → FK → IK → command the EE anywhere — is
demonstrated. The accuracy bottleneck is now clearly the shoulder gravity model;
two complementary fixes: (a) re-identify the shoulder first-moment; (b) add an
**integral term** (PID + gravity comp) to drive steady-state droop → 0 despite the
model error. Either would shrink the 46 mm.

### Open questions / assumptions
- 46 mm is at one pose; EE error will vary with the shoulder load (pose-dependent
  bias). Map it across poses for a thesis figure.

---

## 2026-06-18 — Inverse kinematics (`tools/ik_solve.py`): close the EE-pose → joint-target loop

**Area:** `tools/ik_solve.py` (new) · roadmap step 3 · same URDF/Pinocchio model as
the controller

### Problem / Motivation
To "command the end-effector to any position" we need IK: desired EE pose → joint
angles to use as the controller's setpoint.

### Change
New damped-least-squares (Levenberg–Marquardt) IK on the EE frame Jacobian
(`ee_link`), using the identified URDF via Pinocchio — position-only by default
(the 6-DoF arm leaves orientation free), `--rpy` for a full 6-DoF pose. Clamps to
URDF joint limits each iteration and checks the result against the controller's
conservative software limits; prints a ready-to-run `pd_grav_control --hold-pose`
command. Workflow: `ik_solve --xyz X Y Z` → joints → controller holds the EE there.

### Evidence
Round-trip (IK of the test-pose EE position) recovers the pose to **0.10 mm**;
a fresh target [0.30, 0, 0.40] converges to **0.06 mm**, within soft limits. Needs
ROS sourced for real Pinocchio.

### Impact
Roadmap steps 3 and (point-to-point) 4 are functional: the full goal pipeline —
hold via URDF gravity → FK → IK → command EE anywhere — now exists. Smooth
time-parameterized `q_ref(t)` trajectory tracking is the remaining refinement.

### Open questions / assumptions
- Position-only IK picks *an* orientation (redundancy); use `--rpy` to constrain.
- IK does not yet check self-collision; targets near limits should be eyeballed.

---

## 2026-06-18 — URDF-in-the-loop: gravity (and FK) from the identified URDF via Pinocchio

**Area:** `control/pd_grav_control.py` · puts the identified URDF at the centre of
the control stack (gravity now, FK added, IK next) · see `CONTROL_ROADMAP.md`

### Problem / Motivation
The controller computed gravity from the φ vector. The thesis goal is to control
*from the URDF*, and the same URDF model also supplies forward/inverse kinematics
— so loading it once with Pinocchio unifies gravity + FK + IK.

### Change
Added `--gravity-source {urdf,phi}` (default `urdf`) and `--urdf` (default the
validated 200 Hz URDF `…cfg-9ef2c992…cfg-3ef0a00c.urdf`). In URDF mode gravity =
`pin.computeGeneralizedGravity` ÷ `EFFORT_SCALE` (Pinocchio joint order already
matches `ARM_JOINTS` for this URDF). Graceful fallback to φ if real Pinocchio is
unavailable (gated on `hasattr(pin,'buildModelFromUrdf')` — the ROS gotcha).
Added `ee_pose(q)` (forward kinematics of frame `ee_link`); startup now prints the
gravity source, a **URDF↔φ equivalence self-check**, and the **EE pose (FK)**.

### Evidence
URDF gravity matches the φ vector to **1.3e-7 Nm** over 50 random q; gravity at
home/test poses identical to the φ values used in all prior runs (shoulder −791 /
−233 mA). FK gives sensible EE positions (home [0.36,0,0.56] m). So behavior is
unchanged; only the source of the (identical) gravity moved to the URDF.

### Impact
Roadmap steps 1b (URDF gravity) and 2 (FK) are implemented; IK (step 3) is next,
from the same Pinocchio model. Requires ROS sourced for real Pinocchio (the
controller already runs in that environment).

### Open questions / assumptions
- EE frame assumed `ee_link`; the identified URDF is arm-only (nq=6, no gripper).
- Hardware re-confirmation of an unchanged hold in URDF mode is pending.

---

## 2026-06-18 — RESULT: model-based PD+gravity-comp holds the arm (dual-motor joints included)

**Area:** `control/pd_grav_control.py` · first **hardware-confirmed** model-based
control result · resolves the dual-motor current-control question

### Problem / Motivation
First hardware bring-up of the PD+gravity-comp regulator. Early runs looked like
the dual-motor joints (shoulder/elbow, which have shadow motors) could not be
torque-controlled — the elbow dropped under current control. Needed to determine
whether that was a hardware limitation or a controller bug.

### Change / investigation (chronological, all in `pd_grav_control.py`)
Iterative hardware debugging fixed a chain of real bugs, none of which was a
hardware limit:
1. Setpoint captured from a bogus first `joint_states` (all −π) → settle + median
   + joint-limit gate.
2. **Never command zero current** on stop/kill — it dropped the gravity-loaded
   elbow to the floor; park straight to position mode (servo PID holds).
3. **Mode-switch transient**: switching position→current torque-cycles the
   motors (limp window); the heavy elbow free-falls ~0.5 rad before control
   engages and enters at ~2.9 rad/s.
4. Kill debounce — the mode switch emits a 1-sample garbage position reading
   (jumps to the joint limit) and a velocity glitch.
5. **Ramped setpoint** (capture post-switch position, ramp reference to q_d over
   `--recover-time`; kills use tracking error) → gentle recovery, no violent
   over-correction that previously coupled into the shoulder.
6. Removed the current-blend handoff (it muzzled control authority during the
   transient) and switched the Kd damping term to a **filtered finite-difference
   velocity** — the raw Dynamixel velocity register is too noisy/underreported
   (THESIS_NOTES) to damp with.

### Evidence (hardware, α=1.0, gain×0.5, test pose [0,−0.6,0.5,0,0,0])
Two runs held **15–18 s** with the elbow at the setpoint: **droop −0.00/−0.06 rad,
jitter std ≈ 0.0005 rad (0.03°)** — a stable, non-oscillating hold. The elbow
settled at **−594 mA, essentially its model gravity (−602 mA)** with zero droop,
i.e. the identified model carries the dual-motor elbow. Shoulder held with
+0.05–0.07 rad droop. **The "dual-motor can't be torque-controlled / shadow gives
half torque" hypothesis is FALSIFIED.** Remaining kills were all re-runs *without*
re-posing (degraded start) — operational, not a controller fault.

### Impact
The thesis' control half has a working model-based controller. Open items:
(a) the ~0.59 rad entry dip from the mode-switch limp window (benign, recovers in
~2.3 s; polish by engaging from a low-gravity state / trajectory lead-in);
(b) α sweep to read the per-joint gravity scale from steady-state droop;
(c) shoulder/wrist droop suggests their model gravity is slightly off-scale.

### Open questions / assumptions
- Operational: must `set_pos` to a clean pose before each run.
- Base must be physically secured — an early underdamped oscillation nearly
  tipped the platform.
- forearm_roll FF uses its measured holding current (model known-bad there).

---

## 2026-06-18 — PHASE TRANSITION: first model-based controller (PD + gravity compensation)

**Area:** `control/pd_grav_control.py` (new) · starts the thesis' control half ·
feeds forward the identified gravity model · (hardware results: see the
2026-06-18 RESULT entry above)

### Problem / Motivation
The control phase was gated on a validated gravity model. With the DH fix and the
static-gravity confirmation (2026-06-13: shoulder gravity SHAPE r=0.997 in the
real world; verdict "safe to start PD+gravity-compensation"), that gate is
cleared. `control/trq.py` already had a current-mode PID cascade but with gravity
compensation **commented out** (the Pinocchio path, lines ~376–379) — so no
model-based control had actually run.

### Change
New PD + gravity-compensation **regulator** (holds a fixed setpoint, no
trajectory yet). Law per joint, in mA:
`u = Kp·(q_d − q) − Kd·q̇ + α·G_mA(q)`, where `G_mA(q)` is the IDENTIFIED gravity
evaluated exactly as the static experiment validated it — `sysid_feasible`
Newton-Euler ID at q̇=q̈=0, friction zeroed, divided by `EFFORT_SCALE` →
master-motor mA (pure numpy, no Pinocchio in the loop). The gravity source is the
200 Hz model `…cfg-9ef2c992.npy`.

Safety improvements over `trq.py`: (1) **bump-free handoff** — read each joint's
position-mode holding current and blend the command from there to the full law
over `--ramp-in` s, so the arm stays gravity-supported through the mode switch
(trq.py ramped from zero current, momentarily unsupporting it); (2) pure setpoint
regulation; (3) per-joint soft position limits + current caps; (4) velocity /
position-error kill switches; (5) SIGINT parks the arm in **position mode** (its
PID holds the pose) rather than leaving it limp at zero current (which would
collapse a loaded arm).

### Evidence
Offline gravity sanity (no ROS) at the 200 Hz model: waist ≈ 0 mA (gravity-free
axis ✓), shoulder −791 mA at home → −233 mA folded back (q=[0,−0.6,0.5,0,0,0]) ✓,
elbow ≈ −600 mA. Two flags: forearm_roll shows a spurious +150 mA even at home
(the known joint-4 defect, CHANGELOG 2026-06-13) so its FF is untrustworthy;
shoulder at home (791 mA) is near its 900 mA cap → bring-up must start at the
folded test pose, not home.

### Impact
Opens the control half of the thesis. The **α sweep is also a clean closed-loop
resolution of the 0.58 anomaly**: the gravity gain α that zeroes the steady-state
droop (q_d − q) is the true gravity scale — α≈1 ⇒ identified gravity correct
(0.58 was a position-mode stiction artifact) and precision computed-torque is
viable; α≈0.58 ⇒ a real scale error remains. The regulator logs droop per run.

### Open questions / assumptions
- Default PD gains (`KP_BASE`/`KD_BASE`, `--gain-scale 0.5`) are conservative
  guesses pending hardware tuning.
- forearm_roll FF is known-bad; expect ~0.25 rad droop there at α=1 until joint 4
  is fixed — consider zeroing its FF for clean shoulder/elbow tuning.
- Master-mA command convention matches the identification/static-experiment space
  (driver mirrors master→shadow on dual-motor shoulder/elbow).

---

## 2026-06-18 — Breakaway-current (stiction) test tooling

**Area:** `control/breakaway_current.py` (new) · methodology for resolving the
≈0.58 static-current anomaly · no results yet (hardware run pending)

### Problem / Motivation
The static-gravity experiment (2026-06-13) found steady holding current is
~0.58× the identified gravity current with *perfect* gravity shape (shoulder
r=0.997). The leading explanation is **gearbox stiction**: at standstill the
geared XM540 holds part of the load by static friction, so the position-mode
servo PID settles at the **low edge** of a stiction band rather than at true
gravity. This needed a direct, model-free measurement to either confirm stiction
(→ gravity model correct, PD+G safe) or expose a real gravity scale error.

### Change
New single-joint current-control test, built on the safe pattern in
`tools/test_waist_current.py` (one joint in current mode, all others hold
position; clean revert to position mode). For a joint at a fixed pose it ramps
commanded current up from the measured position-mode holding current until the
joint breaks away (`I_break+ = g + f`), then ramps down (`I_break- = g - f`),
giving:
- **stiction** `f = (I_break+ − I_break−)/2`
- **stiction-free gravity** `g = (I_break+ + I_break−)/2`

Safety: ramp starts at the actual holding current (no jump, model-free baseline);
small steps; breakaway caught at a few-degrees position deviation → joint
immediately handed back to its position PID and re-homed; hard current cap
(`--max-current`, default 600 mA) and absolute abort window. Logs a CSV trace +
results JSON to `data/`.

### Evidence
Pending — script byte-compiles; physics/safety reasoning above. Hardware run
sequence: `waist` (gravity-free control, validates the rig with no runaway risk),
then `shoulder` and `elbow` (the dual-motor geared joints the hypothesis targets).

### Impact
Resolves (or refutes) the stiction interpretation of the 0.58 factor and yields
an independent gravity estimate per pose to cross-check the identified model —
the last open question before committing to PD + gravity-compensation control.

### Open questions / assumptions
- `JointSingleCommand` cmd is interpreted as current (mA) in current mode and as
  position (rad) in position mode — same convention `test_waist_current.py` and
  `vel_osc.py` rely on.
- Dual-motor joints (shoulder/elbow): commanding the master drives the shadow;
  present current read is the master's — same mA convention as the static
  analysis. The midpoint `g` should be compared to the identified gravity current
  in the **same** master-mA space.

---

## 2026-06-14 — Control phase started: `control/pdg_control.py` (PD + gravity-compensation, current mode)

**Area:** new control script · uses the identified model in closed loop · control
phase kickoff

### Problem / Motivation
The validated-URDF gate is met, so the control phase begins. The existing
`control/trq.py` was jerky in practice. Root causes on inspection: gravity
compensation was commented out (so an integral term wound up fighting gravity),
it fed back the *double-differentiated* (noisy) encoder acceleration, and used
large P+I gains with no model feed-forward. The paper authors'
`PD_GravityCompensation.py` (external/paper_model) is the opposite and is proven
on this exact hardware: 200 Hz, current mode, per-joint PD + gravity feed-forward
in mA, no integrator.

### Change
New `control/pdg_control.py`: a clean PD+G current controller,
`u(mA) = soft·[ G(q) + Kp·(q_ref−q) + Kd·(q̇_ref−q̇) ]`. It follows the paper's
proven structure (gains default to their tuned vx300s values; commands in mA) but
(a) feeds forward gravity from **our identified model** (champion URDF via
Pinocchio, Nm→mA by 1/EFFORT_SCALE) with the paper's G as a `--gravity paper`
cross-check; (b) damps with the *measured* velocity (no differentiation, no
acceleration-feedback term); (c) drives a smooth cubic reference that **starts at
the current pose** so the error/command begin near zero (no startup lurch);
(d) soft-starts the command 0→1 over 0.5 s; (e) switches the arm back to
**position mode on Ctrl-C** so it holds rather than falls. Defaults to a
gravity-comp **hold** (regulate to the current pose) for safe first bring-up;
`--goal` commands a move. `--gravity-scale` exposed (1.0 = physical; the stiction
finding says holding needs ~0.6× but full G is correct + conservative). Current
clamp default 2000 mA (paper used 3200).

### Evidence
Software only (no hardware yet): syntax-checks; the cubic reference verified
(zero velocity at t=0 and t≥T, smooth in between). Hardware bring-up pending
(HANDOVER).

### Impact
First model-in-the-loop controller; foundation for computed-torque tracking and
the paper's ERG (which adds a reference governor on top of PD+G — needs the
paper's mass matrix M, currently MATLAB-only via their socket server). No
identification results affected.

### Open questions / assumptions
- Gains are the paper's; may need light retuning for our gravity model / current
  limit. Start in hold mode, then small goals.
- The ERG layer (`external/paper_model/ERG.py`) needs M(q); porting M,C to Python
  (or standing up the MATLAB socket) is the next control sub-task if ERG is in
  scope.

---

## 2026-06-14 — Stiction-hysteresis experiment: tooling to test the standstill-stiction hypothesis

**Area:** new tooling (mover + collection + analysis) · the ≈0.63 scale anomaly /
standstill-stiction question

### Problem / Motivation
The static gravity benchmark found the holding current under-reads the
identified/paper gravity by ~1.6× (the "0.63 anomaly"). The leading,
calibration-independent explanation is that gear stiction supports part of the
gravity load at standstill (THESIS_NOTES "Standstill stiction"). It needs a
direct hardware test.

### Change
The mechanism is falsifiable: static friction settles at opposite ends of its
band depending on the *direction of the last motion*. New tooling holds each
target pose from **both** approach directions and measures the difference.
- `control/stiction_hysteresis_poses.py` — mover; 6 gravity-loaded targets, each
  held "ascending" (waypoint q*−δ) and "descending" (q*+δ) on the swept joints
  (shoulder/elbow/wrist_angle, δ=0.20 rad). Same safety envelope as
  `static_gravity_poses.py` (position mode, slow blocking moves, clipped to
  limits, grasped + no payload). Sidecar logs target/approach/dwell-window.
- `collect_stiction_hysteresis.sh` — gated wrapper, reuses the proven 200 Hz
  recorder (mirrors `collect_static_gravity.sh`).
- `tools/analyze_stiction_hysteresis.py` — per joint: band = I(desc)−I(asc) ≈
  2·τ_breakaway, and midpt = ½(asc+desc) which cancels stiction → should ≈ model
  gravity. Reports in raw mA (+Nm via EFFORT_SCALE); optional `--phi/--urdf/paper`
  overlay to check midpt − model ≈ 0. Reuses `compare_gravity` predictors and
  dwell detection; runs without ROS for `--phi`/paper.

### Evidence
Tooling only (no hardware run yet): mover syntax-checks, analysis tool imports
and `--help` works. Predictions to test: (1) a non-trivial band (≈0 would falsify
stiction); (2) midpoint lands on the model gravity (~1604 mA shoulder) while the
single-direction static hold (1005 mA) sits one half-band below.

### Impact
Provides the experiment that settles whether the model's gravity (used for
control feed-forward) is right and the static current is stiction-biased. Does
not change any model or result. Recommended before/early in the control phase.

### Open questions / assumptions
- δ=0.20 rad assumed enough to traverse the stiction band; the analysis prints
  the per-joint band so this is self-checking (tiny/erratic bands ⇒ raise δ).
- Dwells matched to trials by temporal order + q-consistency (two trials share a
  target q, so order is needed); the tool warns on any order/target mismatch.

---

## 2026-06-13 — Champion validated on full dynamics (held-out torque): beats the no-model baseline; validated-URDF gate MET → control unblocked

**Area:** validation results · `compare_urdf_performance.py` · model-selection
gate for the control phase

### Problem / Motivation
The gravity benchmark validated `G(q)` only. The deliverable URDF must also
predict torque on a *moving* trajectory (M·q̈ + C·q̇ + G) and, per the
post-2026-06-13 rule, **beat the no-model baseline** on the rigid-body-only
per-joint metric — the criterion the gravity-deficit reopening introduced.

### Change / Evidence
Held-out torque validation of the champion URDF (`cfg-a92e984c`, exported via
`phi_to_urdf-v1-1`) with `--friction --drop-glitches`.

**First attempt — May 47 Hz run (`traj_run_20260518_143818.csv`): misleading.**
B beat the factory (−43.6 % RMSE) but came out **−52.7 % vs the no-model
baseline** on rigid-body RMSE. Diagnosed as a **low-rate-acceleration artifact**:
shoulder RMSE/MAE ≈ 6 (rare large spikes), and B is the first model with real
link inertia (1.5 kg upper arm), so `M·q̈` with q̈ differentiated from 46.7 Hz
data injects spikes the old ≈0-mass models never produced. The friction-fitted
MAE already showed the model explained the signal (shoulder MAE 0.495 vs baseline
1.307). Not a fair dynamics test on low-rate data.

**Confirming run — 200 Hz replicate (`traj_run_200hz_20260612_161025.csv`,
`--stride 4`, clean q̈): gate passes.** Same model, baseline margin flips
−52.7 % → **+37.4 %**:

| protocol | metric | no-model | factory A | champion B |
|---|---|---|---|---|
| rigid-body (primary) | mean RMSE [Nm] | 0.884 | 1.242 | **0.554** (+37.4 % vs baseline) |
| rigid-body | shoulder R² | −0.13 | −0.55 | **0.671** |
| friction-fitted | mean RMSE [Nm] | 0.477 | 0.707 | **0.212** (+55.5 % vs baseline) |
| friction-fitted | R² shoulder / elbow | — | <0 | **0.939 / 0.860** |

B beats the no-model baseline decisively on both protocols and beats the factory
by 55–70 %. The factory URDF stays *worse* than no-model (−48 %) even with
friction fitted. (Elbow rigid-body R² is still <0 only because elbow torque is
offset-dominated; friction-fitted R² 0.86 once the offset is absorbed.)

### Impact
- **Validated-URDF gate MET.** The champion `cfg-a92e984c` is now validated on
  both gravity (static benchmark) and dynamics (held-out torque, beats baseline
  + factory). **The control phase is unblocked.** HANDOVER flipped.
- Confirms the May-data result was a sampling-rate artifact (same model, clean
  q̈ → passes), reinforcing "use 200 Hz / clean acceleration for any
  inertia-sensitive validation."
- **Corroborates the standstill-stiction reading of the 0.63 anomaly:** B tracks
  *moving* shoulder torque at R² 0.94 yet over-predicts the *static* holding
  current ~1.6× — correct in motion, over in hold = the stiction signature.

### Open questions / assumptions
- The 200 Hz replicate shares the seed-42 trajectory with the training run →
  proves dynamics fidelity + repeatability, **not** full trajectory-independence.
  The only independent set (May) is rate-confounded. A clean *and* independent
  test needs a new different-seed 200 Hz collection (pre-registered tiebreak).
- forearm_roll is unhelped (R² ≈ 0, RMSE ≈ baseline) — friction/stiction-
  dominated, not gravity, as the static benchmark showed.

---

## 2026-06-13 — Gravity deficit CLOSED on v1.5: re-identified models recover gravity; 200 Hz model (`cfg-a92e984c`) reproduces the paper's published gravity → new champion

**Area:** identification results (v1.5 re-run) · model selection · the ≈0.63
scale anomaly · `data/static_gravity_20260613_183554.csv` benchmark

### Problem / Motivation
The 2026-06-13 cross-check reopened identification: every pre-fix model carried
almost no gravity. After the modified-DH root-cause fix (sysid 1.4→1.5) the
models had to be re-identified and re-judged on the gravity-only static
benchmark (`compare_gravity.py`, entry below).

### Change
Re-identified the May and 200 Hz models on the corrected v1.5 kinematics (same
recipe: `--method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL
--drop-glitches`; 200 Hz adds `--stride 4`) and scored pure gravity G(q) against
the 15-pose static dataset, in raw master-motor mA.
- May v1.5: `outputs/npy/traj_run_20260518_143818__sysid_feasible-v1-5__cfg-9ef2c992.npy`
- **200 Hz v1.5 (new champion): `outputs/npy/traj_run_200hz_20260612_131613__sysid_feasible-v1-5__cfg-a92e984c.npy`**

### Evidence (gravity swing [mA] / corr vs the measured static sweep)
| model | shoulder swing | corr | elbow swing | mean corr* | mean RMSE_off* |
|---|---|---|---|---|---|
| measured | 1005 | — | 558 | — | — |
| May v1-4 (pre-fix, invalid) | 204 | 0.86 | 141 | 0.60 | 145 |
| May v1.5 | 1319 | 0.98 | 74.6 | 0.72 | 125 |
| **200 Hz v1.5 `a92e984c`** | **1604** | **1.00** | **441** | **0.76** | **121** |
| paper | 1683 | 0.99 | 489 | 0.71 | 132 |
| factory `vx300s.urdf` | 1883 | 0.88 | 741 | −0.00 | 298 |

*mean over gravity joints shoulder/elbow/wrist_angle.

1. **Shoulder gravity recovered, best-in-class:** May swing 204→1319 (corr
   0.86→0.98); 200 Hz reaches 1604 at corr 1.00, lowest shoulder RMSE_off of all
   models. First model in the project carrying real shoulder gravity magnitude.
2. **Excitation, not structure, explained the missing elbow/wrist:** May elbow
   swing 74.6 → 200 Hz 441 (≈ paper 489, measured 558); wrist_angle 4.2 → 41.8
   (paper 113, still mildly under-excited). The richer paper-rate trajectory
   filled in the elbow.
3. **Independent reproduction of the paper's gravity** (no CAD prior): 200 Hz
   model within 5–10 % of the paper at the dominant joints (shoulder 95 %,
   elbow 90 %) — the affirmative answer the original cross-check lacked.
4. **Factory URDF is the broken reference at elbow/wrist:** our model and the
   paper both correlate positively with measured there; the factory
   anti-correlates (elbow −0.27, wrist −0.62, mean corr ≈ 0). The paper model,
   not the manufacturer CAD URDF, is the proper benchmark.

### Impact
- **Champion changes to the 200 Hz v1.5 model `cfg-a92e984c`** (gravity-valid:
  carries shoulder+elbow gravity, reproduces the paper, beats factory on every
  gravity joint). HANDOVER updated.
- The gravity-deficit reopening is **closed for the 200 Hz model**. The
  identification → validated-URDF gate for the control phase is effectively met
  on gravity, modulo the open items below.

### Open questions / assumptions
- **≈0.63 scale anomaly reinterpreted (see THESIS_NOTES "Resolution").** Both
  the 200 Hz model (from motion) and the paper predict ~1.6× the static holding
  current (1604/1005 ≈ 0.63). Most plausible cause: **gear stiction supports
  part of gravity at standstill**, so the holding current under-reads gravity
  and the model recovers the true value. Treat static amplitude as a lower bound
  and shape/correlation as ground truth. Does not yet fully clear the ×2/k_t
  question (the factor also appears on moving data); falsifiable via
  holding-current hysteresis between approach directions.
- **wrist_angle** still mildly under-excited (swing 41.8 vs paper 113).
- **forearm_roll** measured swing 1641 mA is physically implausible as gravity
  on a roll axis and predicted by no model — stiction/cogging/current artifact;
  excluded from the gravity-joint means.
- May v1.5 elbow stayed flat (74.6) — confirms the May trajectory's poor
  elbow-gravity excitation; the 200 Hz collection is the lever.



**Area:** new tool `compare_gravity.py` · validation methodology · benchmarks
our models, the factory URDF and the paper authors' published model against the
`static_gravity_20260613_183554` dataset

### Problem / Motivation
The reopened defect is that our identified models carry almost no gravity (G(q)
nearly constant over configuration) while the measured holding currents are
strongly pose-dependent — and the friction-fitted torque validation hid it
(a per-joint offset basis fitted on the test data absorbs a flat model's error).
We needed a comparison that isolates **gravity alone**, the broken term, and
that can also score the paper authors' published model — which is not a URDF but
closed-form symbolic G/M/C in master-motor **mA**.

### Change
New `compare_gravity.py`. In a held pose q̇=q̈=0, so M q̈ and C q̇ vanish and the
measured current collapses to `G(q) + offset + stiction`. The script:
- detects static dwells straight from the velocity signal (all joints
  |q̇|<thresh for ≥min-dur, discarding a settle window), averages q and effort
  over each dwell, and matches each to the nearest commanded pose for validation
  (no CSV/JSON time-base alignment needed — match residual ≤ 0.05 rad here);
- compares **pure gravity G(q)** from: measured holding current, the paper model
  (`external/paper_model`, already mA), any URDF (Pinocchio RNEA(q,0,0)), and any
  of our `phi` .npy (regressor `inverse_dynamics_phi` with friction columns
  zeroed). Nm predictions → mA via `1/EFFORT_SCALE`;
- reports everything in **raw master-motor mA** (assumption-free: no k_t, no ×2
  dual-motor factor — sidesteps the ≈0.63 scale anomaly), per joint: gravity
  **swing** (peak-to-peak over poses; the headline — a gravity-free model ≈ 0),
  RMSE/MAE raw and after removing the best per-joint constant offset (= the
  stiction/bias, isolating gravity *shape*), and correlation. Means are taken
  over the gravity-bearing joints (shoulder/elbow/wrist_angle), since
  waist/wrist_rotate carry ~no gravity in this sweep.
- The paper model (`external/`, .gitignored) is cloned from
  <https://github.com/MomaniMutaz/ViperX-300-6DoF-Robotic-Arm-Dynamical-Model>.

### Evidence (smoke run: paper + May model `cfg-640cb8ef`, no ROS)
15 dwells detected, pose-match residual ≤ 0.048 rad. Measured swing [mA]:
shoulder 1005, elbow 558, **forearm_roll 1641**, wrist_angle 190. The May model
reproduces only shoulder 204 / elbow 141 / forearm_roll 0.6 — i.e. **~5× too
flat at the shoulder and essentially zero elsewhere**, quantifying the
gravity-free defect. The paper model tracks the shoulder shape well (corr 0.99)
but its swing (1683) overshoots measured (1005) by ~1.6× — ratio 0.60,
**the ≈0.63 scale anomaly reappearing in raw mA**, assumption-free.

### Impact
Gives the identification phase a direct, friction-independent gravity metric and
a physically-correct external reference (the paper G). No results invalidated;
this is a new diagnostic. Recommended primary check for any re-identified model.

### Open questions / assumptions
- **forearm_roll** shows a huge measured swing (1641 mA) that neither the paper
  (194) nor our model (≈0) predicts and that anti-correlates with paper-G
  (−0.32) — likely stiction/cogging or a frame issue on the roll axis, not
  gravity; worth isolating before trusting that joint.
- The 0.60 measured/paper ratio on the shoulder is the cleanest statement yet of
  the effort-scale anomaly (k_t / dual-motor / current semantics) — now on
  static data with no inertia/Coriolis confound.
- The q→paper-G angle convention is assumed identity (paper applies its own
  q2/q3 offsets internally); the script's paper-vs-factory correlation check
  (needs a URDF run with ROS) is the validation of that assumption.

---

## 2026-06-13 — ROOT CAUSE FOUND & FIXED: modified-DH parameters were run through a standard-DH kinematic chain

**Area:** `sysid_feasible.py` `_dh_transform` + `_ne_forward_pass` (FIXED,
PIPELINE_VERSION 1.4→1.5) · `phi_to_urdf.py` `_dh_transform` (FIXED, 1.0→1.1) ·
`tools/test_fk_equivalence.py` + `tools/test_phi_urdf_consistency.py` (gates) ·
**invalidates all prior identification**

### Root cause (one sentence)
The `DH_PARAMS` table is written in the **modified (Craig) DH** convention
(α, a are the *previous* link's — identical numbers to `sim/sim.py`'s
`DH_TABLE`, which is validated against the real robot), but `_dh_transform`
composed them as **standard (distal) DH** (`Rot_z·Trans_z·Trans_x·Rot_x`), and
`_ne_forward_pass` used the matching standard-DH Newton–Euler recursion. Feeding
modified-DH parameters through a standard-DH chain shifts every link twist to the
wrong side of its joint rotation — which left the **shoulder as a vertical/yaw
axis** (it should be a pitch axis), so it could not carry gravity. (Credit: the
user spotted that `sim/sim.py` uses a different `dh_matrix` than
`sysid_feasible.py` for the same parameters.)

### The decisive evidence (vs the genuine Interbotix `urdf/vx300s.urdf`)
- **Before:** consecutive joint-axis angle signature (convention-free) was
  ∥,⊥,∥,⊥,⊥ — the real robot is ⊥,∥,⊥,⊥,⊥ (waist⊥shoulder, shoulder∥elbow —
  the parallel pitch pair). The DH structure was shifted one joint down the arm.
  A shoulder sweep held the pipeline EE at constant z (yaw) while the real arm's
  EE swept z=+0.62→−0.11 m (pitch, ~4 Nm gravity).
- **After the fix:** axis signature = ⊥,∥,⊥,⊥,⊥, matching the real robot exactly
  (`tools/test_fk_equivalence.py` CHECK 1 PASS).

### The fix (modified/Craig DH throughout — verified, not guessed)
1. `_dh_transform` → `Rot_x(α)·Trans_x(a)·Rot_z(θ)·Trans_z(d)` (both
   `sysid_feasible.py` and `phi_to_urdf.py`; matches `sim/sim.py`).
2. `_ne_forward_pass` → Craig recursion: joint i rotates about z_i=[0,0,1] of
   frame i; the linear-acceleration step uses the PARENT frame's ω, dω and the
   moment arm in the parent frame, then rotates into frame i. The backward pass
   and torque projection (`n_mat[2]` = z_i) were already correct for modified DH
   and are unchanged.
3. `phi_to_urdf` standalone export needed NO frame rewrite: because `Trans_z(d)`
   commutes with the joint rotation, the URDF link frame coincides with the DH
   frame, so joint origin = transform at q=0, axis z, inertia transfers directly.
   (The earlier v1.1 "frame fix" was reverted; only the `_dh_transform` change
   remains.)

### Verification (independent reference = Pinocchio RNEA on the exported URDF)
- `tools/test_fk_equivalence.py` → **PASS** (axis structure matches real URDF).
- `tools/test_phi_urdf_consistency.py`: regressor torque == Pinocchio RNEA on the
  generated URDF over 75 random (q,q̇,q̈) → max diff **3.0e-7 Nm** (float
  round-off; PASS at 1e-6 tol). Gravity also matches an independent
  energy-gradient computation to ~5e-9 Nm.

### Impact
- **All previously identified `phi`/URDFs are invalid** (fit through the wrong
  kinematics) — May `cfg-640cb8ef`, Ia `cfg-ce8e7059`, every γ-sweep model. All
  identification must be re-run with v1.5. The model-selection history is moot.
- This is the single root cause of the documented "gravity-deficit", the
  "structural 200 Hz defect", and the "broken waist axis."
- KEEP also: `compare_urdf_performance.py` no-model-baseline reporting (separate
  entry below).

### Next steps
Re-run identification (v1.5) on existing data; expect gravity to land on the
shoulder/elbow. Then the static-gravity hardware experiment as the physical
check, and add a pose-level FK calibration check (base+tool+joint-zero offsets)
to also validate link lengths a,d (the axis structure is now confirmed; absolute
reach is not yet asserted by a test).

### How this surfaced
Chasing the phi→URDF gravity mismatch (entry below: same phi, different gravity
through the DH regressor vs the exported URDF). Added a consistency unit test
(`tools/test_phi_urdf_consistency.py`): random phi → torque via the regressor
must equal Pinocchio RNEA on the exported standalone URDF. It failed at ~18 Nm.

### Two distinct problems found
1. **Standalone URDF export (FIXED).** `generate_standalone` placed each joint
   origin at the *full* DH transform `T_rel[i]` (q=0) with axis z. Classic DH
   applies the joint rotation `Rot_z(θ_off+q)` **before** the constant tail
   `K_i = Tz(d)·Tx(a)·Rx(α)`, whereas URDF rotates **after** the origin — so
   for any twisted joint (α≠0) the URDF rotated about the wrong axis. Fix: the
   URDF link frame is the *pre-K* frame `H_i = T[i-1]·Rot_z(θ_off+q)`; joint
   origin `= K_{i-1}·Rot_z(θ_off_i)`, axis z (= true motion axis z of frame
   i-1); each link's inertial is mapped from DH frame i into `H_i` by `K_i`
   (`c'=K_i·c`, `J'=R_{K_i}·J·R_{K_i}ᵀ`). PIPELINE_VERSION 1.0 → 1.1.
2. **The regressor itself disagrees with physics (NOT fixed).** Built an
   independent ground truth — gravity torque as the gradient of potential
   energy `U=−Σ mₖ g·p_com,k` (COMs placed via the regressor's own forward
   kinematics, frame T[k+1]). The **fixed URDF reproduces this to 1.4e-7**
   (so the URDF is correct physics), but **`regressor_fast`/`inverse_dynamics_phi`
   differ from it by up to ~13 Nm.** Per-joint mean gravity error
   (regressor vs truth): waist 0.000, shoulder **7.93**, elbow **7.09**,
   forearm_roll 0.92, wrist_angle 0.81, wrist_rotate 0.09. The regressor reads
   joint torque as `n_mat[2]` — the z-component of the link moment in DH
   frame i — but the joint physically moves about z of frame i-1 (verified by
   FK finite-difference); projecting onto the true motion axis fixes 5/6 joints
   in a quick test, but the shoulder+elbow magnitudes show at least one further
   effect (not a clean single-axis swap), so the exact anatomy is still open.

### Impact (significant)
- **The export fix is correct and verified** — but the identified `phi` were
  fit by the *buggy* regressor, so every delivered parameter set (May
  `cfg-640cb8ef`, Ia `cfg-ce8e7059`, all γ-sweep models) is contaminated:
  identification drove parameters to match measured torque through wrong joint
  axes. This is the leading candidate root cause for the documented
  "gravity-deficit", the "structural 200 Hz defect", and the "broken waist
  axis" — they may all be downstream of this.
- **All identification must be re-run after the regressor is fixed.** Until
  then, no identified URDF can be trusted, and the model-selection history is
  moot.
- Net: tasks reordered — fixing the regressor (`sysid_feasible`) now precedes
  the SDP/γ "first-moment collapse" work (that diagnosis was performed with the
  buggy regressor and must be revisited).

### Next diagnostic step
Isolate the regressor discrepancy precisely: compare `inverse_dynamics_phi`
term-by-term against Pinocchio RNEA on the (now-correct) URDF for a single
non-trivial link at a time; determine whether it is purely the torque-axis
projection, a COM/first-moment frame placement, or both. Then fix
`regressor_fast` to match and re-assert `test_phi_urdf_consistency.py` plus a
new regressor-vs-energy-gradient gravity test.

---

## 2026-06-13 — Validation protocol fix: no-model baseline + rigid-body-primary in `compare_urdf_performance.py`

**Area:** `compare_urdf_performance.py` (reporting only; no numbers change)

### Problem / Motivation
The cross-check entry below showed the friction-fitted mean RMSE has almost no
power against a missing gravity model (nuisance basis alone: 0.381 vs 0.374
with the best model). The protocol needed a built-in control.

### Change
Every report now includes a **no-model baseline** — τ_pred = 0 pushed through
the *identical* protocol (including the friction/Ia nuisance fit) — plus a
`baseline_margin` line showing how much A and B beat it (flagged when <10 %).
The rigid-body-only section is labelled **primary criterion**; the
friction-fitted section is labelled a secondary diagnostic. Docstring updated.

### Evidence / first run
200 Hz run, `--friction --fit-ia --drop-glitches --stride 20`: baseline mean
RMSE 0.344; champion B margin **+2.0 %** (not meaningful); factory A margin
**−80 %** (worse than no model — its full-scale gravity over-predicts the
≈0.63-scaled measured torque, consistent with the scale anomaly below).

### Impact
No recomputation needed (pure reporting). All future model selection must
quote the rigid-body-only per-joint table and the baseline margin.

---

## 2026-06-13 — Cross-check against the paper authors' published model: our identified URDFs carry almost no gravity; the `--friction`/`--fit-ia` validation masked it

**Area:** validation methodology · identification results (both champions) ·
external benchmark (paper authors' repo)

### Problem / Motivation
The paper authors publish their identified model at
<https://github.com/MomaniMutaz/ViperX-300-6DoF-Robotic-Arm-Dynamical-Model>
(no URDF — generated MATLAB/Python functions for M, C, G and friction with the
identified base parameters baked in, **in master-motor current units, mA**).
Question: does our pipeline, run on our data, reproduce a comparable model?

### Evidence (read-only analysis, 2026-06-13; mA→Nm via `(I/1000)·k_t·n_motors`, k_t=2.409, n=2 shoulder/elbow)
- **The paper's gravity model is physically sane.** Over the configurations of
  our 200 Hz run, paper-G vs factory-CAD-G at the shoulder: r ≈ 0.95, similar
  magnitude (RMS 2.60 vs 2.81 Nm).
- **Our measured efforts contain that gravity signal.** Measured shoulder
  torque vs paper-G: r = 0.92 (200 Hz run) / 0.95 (May run); vs factory-G:
  0.87 / 0.91. Gravity is in the data, on both datasets.
- **Our identified URDFs do not contain it.** Gravity RMS over the same
  configurations, shoulder/elbow: Ia-γ0.5 `cfg-ce8e7059` **0.24 / 0.01 Nm**,
  May `cfg-640cb8ef` **0.46 / 0.37 Nm** — vs ≈1.9 Nm of gravity-correlated
  signal in the measurements. Masses sit at the γ-prior values (0.5, 0.5,
  0.26/0.03, 0.03 kg…) with CoMs ≈ 0: the entropic prior, not the data, set
  them. Bare-RNEA residual ≈ measured RMS (shoulder 1.75 of 1.92 Nm) —
  the rigid-body part of the model explains almost nothing.
- **Why validation didn't catch it:** with `--friction --fit-ia`, the
  per-joint nuisance basis `b·q̇ + c·sign(q̇) + d (+ Ia·q̈)` fitted on the
  *test* data scores mean RMSE **0.381 with no rigid-body model at all**
  (τ_pred = 0); the Ia model improves that to only **0.374**. The
  friction-fitted mean RMSE used for all recent model selection (γ sweeps,
  May-vs-Ia matrix) therefore measured the nuisance fit, not the model.
  The joint-mean also hides the shoulder (post-fit shoulder RMSE ≈ 1.2 Nm).
- **Scale anomaly (open):** LS scale of measured torque on factory/paper
  gravity is consistently ≈ **0.63** on both datasets — with our ×2 dual-motor
  scaling, measured gravity-correlated torque is ~35 % smaller than CAD/paper
  gravity. EFFORT_SCALE (n_motors=2, k_t) deserves a dedicated check.

### Impact
- **Answer to the reproduction question: no.** The paper's identification
  recovered gravity; ours collapsed the gravity-bearing parameters (masses,
  first moments) onto the regulariser prior. Root cause not yet isolated —
  candidates: w1/w2/γ balance letting the prior dominate identifiable
  directions (w2=100, γ=0.5), a defect in the gravity columns of the
  regressor, or the effort-scaling anomaly above.
- **All friction-fitted model-selection conclusions are suspect**, including
  the May-vs-Ia "statistical tie" (entry below): both candidates are within
  0.01 Nm of the *no-model baseline*. Rigid-body-only (no `--friction`)
  per-joint metrics must become the primary criterion.
- The identification phase is **not** complete; the validated-URDF gate for
  the control phase is reopened (a gravity-free model cannot do gravity
  compensation).

### Root-cause triage (same day, read-only)
1. **Regressor is OK.** Unconstrained LS on the same stacked regressor
   (200 Hz run, stride 20) recovers configuration-dependent gravity:
   shoulder/elbow gravity RMS 1.35/1.80 Nm, corr 0.61/0.42 with measured.
2. **The identified phi's "gravity" is a constant, not gravity.** May-model
   phi evaluated through the pipeline's own regressor at zero velocity:
   mean −1.43/−2.42 Nm but std only **0.087/0.053 Nm** over the recorded
   poses (LS: std 0.50/0.41). The SDP+γ solution zeroed the first moments
   (CoMs≈0), so all pose dependence is gone; the constant comes from the
   prior masses sitting at the DH frame origins. A constant is exactly what
   the validation's offset nuisance term absorbs — hence invisible.
3. **Separate export-frame inconsistency.** The *same* phi gives different
   gravity through the DH regressor vs. through Pinocchio on the exported
   standalone URDF (zero pose: shoulder −1.49 vs −0.62, elbow −2.39 vs
   +0.39), although masses/CoMs transfer verbatim — the DH model and the
   generated URDF disagree kinematically (suspect: DH θ-offsets / frame
   conventions in `phi_to_urdf.py` standalone mode). Needs a unit test:
   random phi → regressor-G vs URDF-G must match.
4. Even LS recovers only ~⅓ of the expected gravity variation (std 0.50 vs
   ~1.5+ Nm implied by paper/factory G) — consistent with the ≈0.63 scale
   anomaly and/or gravity-direction collinearity in the excitation.

### Open questions / assumptions
- Why does the SDP/γ solution zero the identifiable first moments when the
  data term should resist? (Check the SDP's φ_b against the LS solution in
  the gravity-bearing base directions; check w1·data-term scaling.)
- The ≈0.63 amplitude factor: load sharing between dual motors, effective
  k_t, or current-reading semantics?
- Factory-URDF *elbow* gravity anti-correlates with measured (r ≈ −0.2…−0.3)
  while the paper's correlates (+0.44) — frame/sign convention at the elbow
  worth checking before using factory as baseline A there.

---

## 2026-06-13 — γ retune of the Ia recipe: statistical tie with the May model; Ia is γ-invariant

**Area:** identification results (`sweep_gamma_ia.sh`) · model-selection status

### Problem / Motivation
The Ia model lost the matrix narrowly on May-tuned hyperparameters (entry
below). Question: does its own γ close the gap (the 3 % held-in deficit on
borrowed settings was not a closed case)?

### Evidence (`--friction --fit-ia --drop-glitches` mean RMSE; targets: May model 0.332 / 0.520)
| γ | held-in (200 Hz) | held-out (May) | shoulder Ia [kg·m²] |
|---|---|---|---|
| 0.02 | 0.352 | 0.914 | 0.7985 |
| 0.05 | 0.343 | 0.579 | 0.7982 |
| 0.1 | 0.340 | 0.529 | 0.7980 |
| 0.2 | 0.339 | 0.527 | 0.7979 |
| 0.5 | **0.335** | **0.526** | 0.7977 |

### Impact
- **Statistical tie.** γ=0.5 (`cfg-ce8e7059`) is within the ~0.01 Nm
  run-to-run repeatability resolution of the May model on *both* arenas
  (−0.003 / −0.006). Strictly by the decision rule the May model keeps the
  crown; honestly stated, the models are indistinguishable on the available
  data — and the Ia model carries the healed waist and an explicit actuator
  model. Its γ=0.5 realisation is also presentable (CoMs ≤ 2 cm, no 1 m
  outliers), unlike γ≤0.05.
- **Ia is γ-invariant** (0.1 % drift over a 25× γ range, all joints) while
  the link realisation changes drastically — strong identifiability evidence
  that Ia is determined by the data, not the prior. Thesis-grade result.
- **Tuning is exhausted as a tiebreaker**: both curves flatten (held-out
  → ~0.525 > 0.520), and each further evaluation against the May CSV erodes
  its held-out independence (multiple-comparisons). **The tie can only be
  broken by a different-seed 200 Hz collection** — a neutral arena neither
  model has seen — with the decision rule fixed *before* looking
  (pre-registered: lower mean friction+Ia-fitted RMSE on the neutral set
  wins; candidates: May `cfg-640cb8ef` vs Ia-γ0.5 `cfg-ce8e7059`).

### Open questions / assumptions
- Elbow Ia (0.156) vs shoulder Ia (0.798) asymmetry persists at every γ —
  stable, but unexplained for identical drivetrains.
- forearm_roll Ia pinned at 0 across the sweep (axis nearly aligned with its
  own rotor at low load?) — note, not investigated.

---

## 2026-06-13 — Implemented the per-joint reflected motor-inertia term (`--motor-inertia`)

**Area:** `sysid_feasible.py` (regressor + SDP) · `phi_to_urdf.py` ·
`compare_urdf_performance.py` (new `--fit-ia` protocol) · model structure

### Problem / Motivation
The γ sweep (2026-06-12 evening entry) established that the 200 Hz
re-identification defect is **structural**: the data contains real
acceleration-correlated torque that the rigid-body link parameterisation can
only caricature. The physical candidate is **reflected actuator inertia**
(rotor + gearhead at ≈270:1; Ia = N²·J_rotor, plausibly 0.1–1 kg·m² seen from
the joint), which is local to each joint axis — a link-inertia stand-in wrongly
couples into other axes via RNEA (the reproducing waist defect).

### Change
- **`sysid_feasible.py --motor-inertia`** (SDP path only): 6 extra regressor
  columns appended *after* the 78 link parameters — column 78+i is q̈ᵢ on joint
  i's row (`np.hstack([W, np.diag(ddq)])`), structurally like the friction
  columns. 6 new linear params `Ia_i` with the single constraint `Ia_i ≥ 0`;
  the per-link pseudo-inertia LMIs, coupling term and entropic regulariser are
  untouched, so the problem stays a convex SDP. phi becomes shape (84,).
  Config key `motor_inertia` is **conditional** (same idiom as
  `--drop-glitches`), so the delivered recipe's hash/artifacts reproduce
  byte-for-byte; no version bump needed.
- **`phi_to_urdf.py`** accepts (84,): URDF is built from phi[:78] unchanged
  (Ia is *deliberately* not representable in `<inertial>` — putting it there
  would re-introduce the false cross-axis coupling); Ia is reported in the
  summary, an XML comment in the URDF, and the artifact sidecar JSON
  (`config.motor_inertia_Ia`) for the controller feed-forward `Ia·q̈` /
  pinocchio `model.armature`.
- **`compare_urdf_performance.py --fit-ia`** (requires `--friction`): adds a
  per-joint `q̈` column to the nuisance LS basis, fitted per model per dataset
  — symmetric across A and B, exactly like friction — and prints the fitted Ia.
  This is the protocol for validating Ia-identified URDFs (their rigid-body
  part deliberately excludes the q̈-proportional torque). **Numbers under
  `--fit-ia` are not comparable to friction-only numbers**; the cross-model
  matrix must be redone under one protocol.
- New unit tests `tests/test_motor_inertia.py`: regressor ↔ scalar
  Newton–Euler agreement for (78,) and (84,) phi, Ia block = diag(q̈), link
  columns unchanged by the flag. Suite: 47/47 pass.

### Evidence (smoke tests; real runs pending)
- Stride-40 in-process identification on the 13:16 200 Hz CSV (γ=0.05 recipe +
  `--motor-inertia`): solver optimal, **[ALL OK]**, and — the headline — the
  **upper-arm link inertia stays at blob scale (~0.002)** with Ia absorbing
  the acceleration torque: shoulder Ia = 0.80 kg·m² (dual motor), waist 0.074,
  elbow 0.098.
- Independent cross-check via `--fit-ia` on the **May model's** 200 Hz
  residuals: fitted shoulder Ia = 0.64 kg·m² — same scale from a completely
  different estimator. (Factory model's fitted shoulder Ia is *negative*,
  −0.77: its CAD link inertias are too large, consistent.)
- Hash stability: re-running `phi_to_urdf` on the delivered npy cache-hits the
  existing `cfg-3ef0a00c` artifact.

### Impact
- Ready for the real experiment: re-identify on the 200 Hz data with
  `--motor-inertia` (stride 4) and run the cross-validation matrix under
  `--friction --fit-ia --drop-glitches`. Success bar (from HANDOVER): beat the
  May model on **both** datasets *under the same protocol*, with a sane waist
  axis and link inertias at sane scale.
- NLP solvers (`trust-constr`/`SLSQP`) intentionally do not support the flag
  (legacy path); they raise with a clear message.

### Results (same day) — Ia model loses the matrix narrowly; waist healed; May model stays the deliverable
Real identification (13:16 CSV, May recipe + `--motor-inertia`, artifacts
`cfg-f512651d`): optimal, **[ALL OK]**, unconstrained REL 0.510→0.476 (shoulder
0.569→0.437 — the q̈ columns explain real signal), upper-arm inertia at blob
scale (no inflation), Ia = [0.084, 0.798, 0.156, 0.000, 0.040, 0.009] kg·m².
Full cross-validation matrix under the new `--friction --fit-ia
--drop-glitches` protocol (mean RMSE / MAE [Nm]):

| model \ data | 200 Hz 13:16 | May 47 Hz |
|---|---|---|
| factory vx300s.urdf | 0.618 / 0.452 | 1.488 / 0.520 |
| May model `cfg-640cb8ef` | **0.332** / 0.231 (held-out) | **0.520** / 0.351 (held-in) |
| Ia model `cfg-f512651d` | 0.343 / 0.239 (held-in) | 0.579 / 0.379 (held-out) |

- **Decision rule: not dethroned** (−3 % and −11 %). The May model remains the
  deliverable.
- **The structural defect is nevertheless fixed**: Ia-model waist R² +0.733
  held-in and RMSE 0.244 held-out on May data — *better than the May model's
  own held-in waist* (0.275); all previous 200 Hz models were at R² −3…−12.
  The residual gap is concentrated in the **shoulder** (1.754 vs 1.549 on May).
- Validation-fitted Ia cross-checks: on 200 Hz data both non-factory models'
  residuals want shoulder Ia ≈ 0.64–0.74 (factory: −0.77, its CAD inertias
  overshoot); on the May CSV all fitted Ia ≈ 0 — the 47 Hz pipeline filtered
  the q̈ signal away, which is *why* the May-era identification never saw the
  term.
- **Caveats on the margin** (THESIS_NOTES): the Ia model ran on May-tuned
  hyperparameters (γ=0.05 etc.), and challengers are tested held-out on the
  *dirty* May CSV while the incumbent is tested held-out on the *clean* 200 Hz
  data. A γ retune for the Ia recipe (`sweep_gamma_ia.sh`) and/or a
  different-seed 200 Hz collection (neutral held-out arena) are the follow-ups
  before the thesis fixes a final winner.
- Realisation cosmetics: the Ia model's distal masses collapsed (4–18 g,
  forearm CoM 0.98 m) — torque-faithful (RNEA uses mc, J_O) but not
  presentable; a ref-mass/γ retune item, only relevant if the Ia model wins.

### Open questions / assumptions
- Elbow Ia (0.156) ≪ shoulder Ia (0.798) despite the same dual-XM540
  drivetrain — identifiability or real? The shoulder is also where the Ia
  model still loses; possibly the Ia column absorbs shoulder torque that
  belongs elsewhere.
- Wrist joints are XM430 (not XM540) with different gearing — per-joint Ia
  magnitudes should differ; treat large deviations from N²·J_rotor expectations
  as an identifiability flag, not truth.
- Whether validation should use the *identified* Ia as fixed feed-forward
  instead of re-fitting it per dataset is a protocol choice (see THESIS_NOTES
  "Reflected motor inertia"); fitting was chosen for symmetry with the
  friction treatment.

---

## 2026-06-13 — Workspace reorganisation: pipeline at root, everything else categorised

**Area:** repository layout only — **no code, results, or methodology changed**

### Problem / Motivation
The repo root had accumulated ~35 scripts; the active pipeline was buried among
superseded sysid variants, May-era collection scripts, one-off diagnostics, and
scratch files. Goal: root shows the working pipeline, the rest is categorised
but preserved (traceability — examiners may ask about earlier attempts).

### Change (all moves via `git mv`, history preserved)
- **Root now holds only the active pipeline**: `sysid_feasible.py`,
  `phi_to_urdf.py`, `compare_urdf_performance.py`, `pipeline_artifacts.py`,
  `run_trajectories.py`, `record_joint_states_200hz.py`, `check_collection.py`,
  `check_topic_rate.py`, `collect_200hz.sh`, `identify_200hz.sh`,
  `sweep_gamma.sh`.
- **`tools/`** (working utilities off the critical path): `volt_watch.py`,
  `diagnose_comm.py`, `diagnose_syncread.py`, `diagnose_phi.py`,
  `monitor_servos.py`, `test_waist_current.py`, `plot_arm_data.py`,
  `plot_simple.py`, `visualize_arm_data.py`.
- **`archive/identification/`** (pre-SDP sysid variants): `sysid_paper.py`,
  `sysid_fast.py`, `sysid_subsample.py`, `sysid_19th.py`,
  `sysid_feasible_original.py`, `run_sysid_{cur,pos,pos_paper}.py`, plus
  `dynamic_model.py` (moved from `trash/`, it is `sysid_paper.py`'s import).
- **`archive/collection/`** (May-era 47 Hz flow): `record_joint_states.py`,
  `run_trajectories.sh`, `collect_arm_data.py`,
  `collect_joint_torque_vel_accel.py`, `collect_sysid_pos.sh`,
  `record_arm_data_manual_movement.py`, `record_joint3_test.py`.
- **`archive/scratch/`**: `test.py`, `test2.py`, former `trash/` contents,
  `self_host/` (unrelated LLM experiments). **`archive/npy/`**: the legacy
  `npy/` phi files (already migrated to `outputs/legacy/` by
  `pipeline_artifacts.migrate_legacy`).
- `Paper.txt` → `docs/Paper.txt` (references in CLAUDE.md / PAPER_SUMMARY.md
  updated). README.md rewritten (was May-era stale — it still recommended
  `pip install pin`, the documented gotcha). Added `.gitignore`
  (`__pycache__/`, `*.pyc`, `.pytest_cache/`) and untracked the 12 committed
  `.pyc` files.

### Evidence nothing broke
- Reference map checked before moving: the active root cluster references only
  itself; archived scripts reference only each other (`record_joint_states.py`
  is functionally used only by the archived `run_trajectories.sh`; mentions in
  active files are docstrings).
- Post-move: core modules import cleanly; `bash -n` passes on all three root
  `.sh` scripts; `pytest tests/` 43/43 pass. (`run_trajectories.py` import
  fails only on missing `interbotix_xs_modules` — pre-existing; lab machine
  only.) `COLLECTION_200HZ.md` mentions of old scripts are all in sections
  already marked superseded.

### Impact
- No re-runs needed; artifact paths (`data/`, `outputs/`) unchanged.
- Historical CHANGELOG/THESIS_NOTES entries cite old root paths (e.g.
  `volt_watch.py`, now `tools/volt_watch.py`) — left as written; they are
  records of their date. This entry is the path map.

---

## 2026-06-12 (evening) — Replicate 200 Hz run, γ sweep, matrix completed: the 200 Hz defect is structural → motor-inertia term next

**Area:** data collection (replicate run) · identification methodology (γ sweep,
`sweep_gamma.sh`) · validation results

### Problem / Motivation
The morning matrix (entry below) left two open questions: (1) is the 200 Hz
collection *repeatable* — i.e. are model-to-model RMSE differences real or
within run-to-run noise; (2) is the 200 Hz re-identification defect (upper-arm
inertia inflation + waist degradation) *regularisable away* by the entropic
weight γ, or *structural* (⇒ implement the `Ia·q̈` motor-inertia term)?

### Change
- **Second 200 Hz collection** `data/traj_run_200hz_20260612_161025.csv`
  (185 000 rows, 926.9 s, 199.6 Hz, 25 sentinel rows, check_collection PASS,
  2 stalls absorbed, worst gap 173 ms). Same seed-42 trajectory as the 13:16
  run ⇒ a **replicate**, not an independent excitation; the May run remains the
  only independent held-out set.
- **`sweep_gamma.sh`**: γ ∈ {0.1, 0.2, 0.5, 1.0, 2.0} on the 13:16 CSV, recipe
  otherwise unchanged (`--stride 4 --drop-glitches`, w2=100, CLARABEL). Bug fix
  along the way: the sweep's output parsing died under `set -e` on artifact
  **cache hits** (scripts print `[cache] …` instead of `Saved →`); path
  extraction now matches both, and the upper-arm-inertia indicator is read from
  the URDF file (not printed on cache hits).
- **Replicate-CSV identification** under the γ=0.05 recipe (artifacts
  `…161025__…cfg-a92e984c…`), plus the two missing matrix cells.

### Evidence 1 — repeatability (friction-fitted mean RMSE [Nm], `--drop-glitches`)
| model | 13:16 run | 16:10 replicate |
|---|---|---|
| factory vx300s.urdf | 0.719 | 0.707 |
| May model `cfg-640cb8ef` | 0.438 | **0.438** |
| 200 Hz 13:16 model (γ=0.05) | 0.460 (held-in) | 0.468 |

Run-to-run drift ≤ 1.7 % (the May model reproduces to three decimals) ⇒ the
collection is reproducible and differences ≳ 0.01 Nm between models are real.

### Evidence 2 — γ sweep (13:16 CSV; targets: held-in < 0.438 **and** held-out < 0.645)
| γ | held-in (200 Hz) | held-out (May) | upper-arm Ixx [kg·m²] |
|---|---|---|---|
| 0.05 | 0.460 | 2.355 | 0.051 |
| 0.1 | 0.437 | 0.807 | 0.0043 |
| 0.2 | 0.433 | **0.761** | 0.0024 |
| 0.5 | 0.429 | 0.765 | 0.0021 |
| 1.0 | 0.424 | 0.786 | 0.0020 |
| 2.0 | **0.416** | 0.825 | 0.0020 |

- **No γ meets the held-out bar** (best 0.761 at γ=0.2 vs required 0.645),
  although every γ ≥ 0.1 matches or beats the May model held-in.
- The inertia inflation itself **is** regularisable: by γ=0.5 the upper-arm
  inertia sits at the blob floor (~0.002). But held-out is U-shaped and worsens
  again past γ=0.5 while the dataset-specific torque **migrates into other
  parameters** (shoulder F0 0.228→0.340 Nm, elbow F0 −0.886→−0.938 Nm as γ
  rises). The inflated inertia was a symptom, not the disease: the 200 Hz data
  contains real torque content that no feasible *link*-parameter assignment
  represents without hurting generalisation.
- Metric caveat: γ=2.0 has the best held-in RMSE (0.416) yet the *worst* mean
  REL (0.670 vs 0.509 unconstrained) — REL and friction-fitted RMSE weight
  joints differently; cite the RMSE matrix, not REL, for model comparisons.

### Evidence 3 — matrix completion on the replicate (friction-fitted, `--drop-glitches`)
- May model on the replicate: **0.438 / 0.309** (RMSE/MAE) — identical to the
  13:16 run.
- Replicate-identified γ=0.05 model held-in on its own data: **0.375 / 0.254**
  — the best held-in fit of any model — but the **waist defect reproduces on
  independent data**: waist RMSE 0.314, R² −2.90 (vs factory 0.133, +0.30).
  Same structural signature as the 13:16-identified model (waist R² −7.1).

### Impact
- **Decision rule resolved: the defect is structural.** Per the standing plan
  (THESIS_NOTES "Cross-run validation and the 200 Hz re-identification
  puzzle"), the next identification experiment is the **per-joint reflected
  motor-inertia term `Ia·q̈ᵢ`** in the regressor (6 new linear parameters,
  Ia ≥ 0; SDP structure unchanged), then re-run the sweep/matrix.
- The delivered May model `cfg-640cb8ef` remains the deliverable, now backed by
  two independent 200 Hz validations (0.438 on both).
- The 13:16 vs 16:10 pair gives the dissertation a clean **repeatability**
  statement for the validation methodology.

### Open questions / assumptions
- Whether `Ia·q̈` absorbs the structural torque (and fixes the waist axis) is
  the open experiment; if it does not, next suspects are friction model
  richness (velocity-dependent beyond viscous+Coulomb) and excitation
  conditioning (paper Eq. 11).
- The torque migration into F0 with rising γ is observational; the mechanism
  (why a *constant* offset trades against inertia terms here) is not pinned
  down.

---

## 2026-06-12 — Cross-validation matrix: delivered model passes held-out; 200 Hz re-id regresses

**Area:** `compare_urdf_performance.py` (new `--drop-glitches` flag) · validation
methodology · identification results

### Problem / Motivation
With the 200 Hz dataset in hand, two questions: (1) does the delivered May model
(`cfg-640cb8ef`) survive held-out validation — the project's standing credibility
gap; (2) does re-identifying on 200 Hz data improve the model?

### Change
`compare_urdf_performance.py` gained `--drop-glitches` (plumbed to the existing
`load_and_filter` support). Without it, sentinel dropout rows smear spurious q̈
spikes through the filter and dominate RMSE: the first (contaminated) held-out
run returned a misleading "61 % worse" verdict with RMSE/MAE ≈ 8 and R² ≈ −3300
on the waist. New 200 Hz identification artifact: `cfg-a92e984c` (recipe
unchanged + `--stride 4 --drop-glitches`; stride 4 keeps W within WSL2 RAM —
subsampling happens after 200 Hz filtering/differentiation, so q̈ quality is
preserved).

### Evidence — the matrix (friction-fitted mean RMSE / MAE [Nm], `--drop-glitches`)
| model \ data | May 47 Hz | 200 Hz |
|---|---|---|
| factory vx300s.urdf | 2.066 / 0.649 | 0.719 / 0.569 |
| May model (delivered) | 0.645 / 0.355 (held-in) | **0.438 / 0.313 (held-out)** |
| new 200 Hz model | 2.355 / 0.488 (held-out) | 0.460 / 0.323 (held-in) |

### Impact
- **The delivered model is now cross-validated** (better held-out than held-in,
  +0.52 mean R², 39 % ahead of factory on unseen data). Control phase proceeds
  on it unchanged.
- **200 Hz re-identification under the unchanged recipe is strictly dominated**
  (worse than the May model even on its own training data). Root-cause
  hypothesis (reflected actuator inertia absorbed into link inertias, degrading
  cross-axis coupling — see THESIS_NOTES "Cross-run validation and the 200 Hz
  re-identification puzzle") defines the next identification experiment:
  per-joint motor-inertia term `Ia·q̈` in the regressor + γ sweep.
- Earlier same-day validation numbers (and the 2026-06-10 FINAL entry's
  0.822/2.682) were computed **without** glitch dropping; cite the matrix above
  going forward.

### Open questions / assumptions
- Motor-inertia hypothesis untested; `Ia` extension not yet implemented.
- The May CSV's waist channel keeps strongly negative R² for all models even
  glitch-dropped (residual single-joint glitches?) — minor, but unexplained.

---

## 2026-06-12 — First lab day at 200 Hz: rate fix, three collapses, anti-burst pacing fix

**Area:** driver config (`interbotix_xsarm_control/config/vx300s.yaml`, outside repo) ·
`run_trajectories.py` command pacing · data collection

### Problem / Motivation
First on-site attempt at the 200 Hz collection (runbook `COLLECTION_200HZ.md`).
Two distinct problems surfaced:

1. **Topic-rate gate failed at exactly 50.0 Hz** despite `latency_timer` = 1 ms.
   Root cause was **not** the FTDI timer (the runbook's prime suspect): the
   interbotix driver config `vx300s.yaml` caps `joint_state_publisher:
   update_rate: 50` — uniquely among all models (others ship 100). This also
   retro-explains the May run's ~47 Hz (50 Hz cap minus 16 ms-latency losses).
2. **Three consecutive smoke runs collapsed** (arm fell mid-trajectory) after
   the rate fix: `smoke_200hz_20260612_{114256,121101,122641}.csv`.

### Evidence (collapse forensics, from the 200 Hz CSVs)
- In every run: ~70 ms publisher stalls at ≈1 s and ≈9 s after motion start,
  then a **fatal 144–276 ms stall at ≈16 s**, after which **all motors except
  the waist read 0 mA** (torque off) and the arm falls (shoulder −4 rad/s into
  its hard stop). The waist — first in the power daisy-chain — stays
  torque-enabled, jams against the fallen arm at stall current (~5.4 A,
  ≈13 Nm), and trips its own overload ~5 s later. The big waist spike is
  therefore an *effect*, not the cause.
- **Falsified: mechanical overload** — pre-event currents ≤ 0.84 A (~10–15 %
  of capacity), no spike before cutoff at 5 ms resolution.
- **Falsified: posture-triggered** — run 3 (halved VEL/ACCEL limits ⇒
  different optimized trajectory) failed at the same *relative time* in a
  different posture, never visiting the waist extreme.
- **Environment**: the U2D2 runs over usbipd/vhci (USB-over-IP into WSL2;
  `dmesg`: `vhci_hcd`, one FTDI `urb stopped: -32`). No USB re-enumeration
  during failures. The May 865 s run (different machine/path) has zero stalls.
- **Suspected mechanism**: the pacing loop in `run_trajectories.py` only slept
  when *ahead* of schedule; after a stall it **burst the backlog** (~14
  waypoints at `moving_time=0.08`) — a violent multi-joint catch-up jerk
  (run 1: waist measured 2.6 rad/s > commanded profile). Hypothesis: the
  simultaneous inrush dips the supply and reboots the downstream motors
  (2–9), while the waist (first in chain) rides through. Pending confirmation
  via driver log / `Hardware_Error_Status` registers.

### Change
- `vx300s.yaml` (driver config): `update_rate` 50 → 200. Verified 196.1 Hz
  effective with clean header-stamp timing — the runbook's Tier-1 goal is met.
- `run_trajectories.py`: on falling ≥ 2 command intervals behind schedule, the
  schedule start is shifted by the deficit instead of bursting stale waypoints
  (trajectory resumes seamlessly, finishing late by the stall time); stalls are
  counted and printed. Waypoint generation is untouched. VEL_MAX/ACCEL_MAX
  restored to 3.14 / 10.0 after the halved-limits diagnostic run.

### Impact
- 200 Hz collection is viable; smoke must be re-run to confirm the anti-burst
  fix prevents the collapse. If the collapse persists *without* a command
  burst, the cause is purely electrical (PSU sag / chain cabling) and the
  stalls are a symptom, not a trigger.
- The seed-42 waypoints are unchanged (same coefficients), so the excitation
  remains comparable to the delivered-model trajectory.

### Resolution (same day) — root cause confirmed: power-path voltage sag
The anti-burst fix absorbed stalls but a 5th run still collapsed ⇒ command
path exonerated entirely. A 100 s passive listen with the arm **idle** showed
200 Hz with *zero* gaps ⇒ host/USB path exonerated; stalls occur only under
motor load. A live `Present_Input_Voltage` poller (**`volt_watch.py`**, new
diagnostic; logs in `data/logs/voltwatch_*.log`) then provided the smoking
gun: 11.9–12.1 V at idle, but **9.8–10.2 V under dynamic load** and **8.7 V
while the power connector was handled** — vs the XM540's 9.5 V operating
floor. Root cause: a **resistive/loose connection where the 12 V/20 A brick
feeds the U2D2 power-hub board**, sagging ~2 V under load; deep sags
brown out and reboot the motors furthest down the power chain (everything
but the waist), which is the observed collapse. The shallow sags are the
~70 ms publisher stalls. After reseating the connector, the full smoke
**PASSED** (`smoke_200hz_20260612_130906.csv`, 198.9 Hz effective, 86 s,
check_collection PASS) with one absorbed 126 ms stall and dips bottoming at
9.8 V — *still marginal*: the connector must be mechanically secured before
the 900 s run (15 min of vibration).

**The 900 s collection then succeeded**: `data/traj_run_200hz_20260612_131613.csv`
— 185 238 rows, 927.8 s, 199.7 Hz effective, 1 absorbed stall, 33 sentinel
rows (0.018 %), check_collection **PASS**. volt_watch over the full run:
min 9.7 V (never below the 9.5 V floor), worst dips during gripper grasp +
trajectory spin-up — which retroactively explains why all collapses occurred
in that early high-demand window. The paper-rate dataset for identification
now exists; supply headroom remains thin (~0.2 V at worst), so secure the
brick→PHB connection (screw terminals) before the control-phase campaigns.

### Open questions / assumptions
- Power headroom is still thin (9.8 V min under load post-reseat). If stalls/
  sentinels persist in the real run, the barrel-jack→PHB joint needs a proper
  fix (screw terminals / re-crimp), not another reseat.
- Source of the recurring ~7–8 s spacing between sags (suspected: recurring
  high-demand phases of the f_l = 0.1 Hz excitation) not pinned down.

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
