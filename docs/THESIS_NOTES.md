# Thesis discussion notes

Raw material for the dissertation's **Methodology** and **Discussion** chapters:
non-trivial design decisions, **deviations from the source paper** and their
justification, trade-offs, and open questions an examiner might raise.

Distinct from the other docs:
- `CHANGELOG.md` — *what changed, when* (chronological engineering log).
- `PAPER_SUMMARY.md` — *what the paper says* (condensed reference).
- **`THESIS_NOTES.md` (this file)** — *what to argue/discuss* (decisions &
  reasoning, organised by topic, not by date).

Each entry: the topic, what the paper did, what we did and why, the trade-off,
and how to frame it in the thesis.

---

## Solver for the physically-feasible identification (Eq. 16)

**The paper's tooling.** Momani & Hosseinzadeh use two *different* optimizers for
two *different* problems:
- **Parameter identification** (Eq. 16 — minimise data-fit + coupling subject to
  the feasibility constraints): solved with **YALMIP** (§3.4). YALMIP is a MATLAB
  modelling layer that dispatches to an underlying **semidefinite-programming
  (SDP) / convex solver**. This is natural because the feasibility constraints
  `J_i − m_i·CᵀC ≻ 0` (16c) and `Iᶜ_i ≻ 0` are **linear matrix inequalities
  (LMIs)** — convex semidefinite constraints.
- **Excitation-trajectory design** (Eq. 11 — minimise `cond(Φ_b)` under joint
  limits): solved with **MATLAB `fmincon`, active-set algorithm** (§3.2). *(This
  repo does not yet implement trajectory optimisation; see open questions.)*

**What this repo does.** `sysid_feasible.py` reformulates Eq. 16 as a general
**nonlinear program (NLP)** and solves it with **SciPy** (`scipy.optimize.minimize`),
selectable between `trust-constr` (default) and `SLSQP`. The PD/LMI constraints
are imposed as **nonlinear inequality constraints** — concretely via the
eigenvalue triangle inequalities (16d–f) and a pseudo-inertia positive-eigenvalue
test — rather than as native semidefinite constraints.

**Is this a deviation? Yes — but at the *solver* level, not the *problem* level.**
- The **optimization problem is the same** as the paper's: identical objective
  (16a) and the same feasibility set (16b–16i). Fidelity to the paper's
  *methodology* lives in the problem formulation, which is preserved.
- The **solver/representation differs**: NLP (SciPy) vs SDP (YALMIP). This is a
  legitimate implementation choice, motivated by avoiding a MATLAB + YALMIP +
  commercial-SDP-solver dependency and keeping the pipeline pure-Python/ROS.
- Within that already-made choice, **`trust-constr` vs `SLSQP` is a free
  selection** — neither is the paper's solver, so swapping between them
  introduces *no additional* divergence. If anything, `SLSQP` (a Sequential
  Quadratic Programming method) is the closer relative of the paper's
  `fmincon` active-set/SQP family.

**Trade-off to discuss.**
- *SDP route (paper):* the LMI constraints are convex, so a single global optimum
  is found reliably; needs MATLAB/YALMIP + an SDP backend.
- *NLP route (this repo):* no convexity guarantee — the eigenvalue/triangle
  reformulation is non-convex, so the solver can stall in poor local minima or
  exhaust its iteration budget (observed: `trust-constr` stopped on "max function
  evaluations", giving a constrained fit *worse* than unconstrained least squares;
  see `CHANGELOG.md` 2026-06-10 re-identification entry). This is the practical
  cost of trading SDP for NLP and is the motivation for the solver-tuning /
  `SLSQP` / `--stride` experiments.

**Resolution (implemented).** The two SciPy NLP solvers were found to fail in
complementary ways on this non-convex reformulation: `trust-constr` stays
feasible but stalls at a poor fit (REL 0.82); `SLSQP` reaches the data-optimal
fit (REL 0.60) but exits the feasible set (negative inertia eigenvalues) and
collapses the distal-link masses. This is the practical cost of NLP-vs-SDP. We
therefore implemented the **convex SDP route with CVXPY** (`--method cvxpy` in
`sysid_feasible.py`, function `identify_sdp`), which mirrors the paper's
YALMIP/SDP approach in pure Python. Key simplification: the physical-consistency
condition is imposed as the single **pseudo-inertia LMI** `P_i(phi) ⪰ 0`
(Wensing, Kim & Slotine 2017, paper ref [53]), which provably subsumes mass>0,
`I^c ≻ 0` (16c) and the triangle inequalities (16d–f). Being convex, it has a
unique global optimum — feasible *and* optimal — so it removes both NLP failure
modes.

**Critical: no CAD/reference-model prior is used (and why that matters).**
A tempting shortcut to fix the mass collapse is to regularise the standard
parameters toward the manufacturer `vx300s.urdf` inertials. We deliberately do
**not** do this. The thesis premise is that the paper's method can build a
dynamic model **from data alone for a robot that has no pre-existing model**
(the paper's own motivation). Regularising toward an existing model would:
(i) assume the very answer being sought — unavailable for a genuinely model-less
robot, so the demonstration would not transfer; and (ii) make the validation
circular (the output would be "equivalent to the prior" by construction). The
SDP instead resolves the non-identifiable directions using **physical
feasibility constraints only** (physics, not a reference model), keeping the
demonstration honest and transferable. `vx300s.urdf` is used in this project
**only** as (a) a validation baseline and (b) the source of *kinematics* (DH
frames / link lengths, which are legitimately known), never as a prior on the
*inertial* parameters being identified.

**Representation nuance (worth a paragraph in the thesis).** The identified
*model* lives in the **base parameters** — what the data determines and what
`M(q), C, G` depend on. A URDF's per-link masses/inertias are a *non-unique
realisation* of those base parameters; their non-uniqueness is a known property,
not an identification failure. If a "nicer" URDF realisation is wanted without a
CAD prior, the principled, model-free option is the **entropic / uniform-density
regulariser** (log-det of the pseudo-inertia, Wensing [53]) — kept separate so the
baseline result stays the exact paper formulation (Eq. 16).

**Status: implemented (2026-06-10)** as `identify_sdp(..., entropic_gamma=γ)`,
exposed as `--entropic γ` (γ=0 ⇒ exact Eq. 16). The realisation choice is forced:
the γ=0 SDP fits well (REL 0.59 = baseline) and is feasible, but its standard
masses collapse toward zero (the unidentifiable null-space), and `phi_to_urdf.py`
rejects near-zero masses — so *some* prior-free realisation selector is unavoidable.

*First attempt — pure max-entropy `−γ·Σ log det P_i` — and why it fails.* `log det P`
is concave, so `−log det P` is convex and keeps the SDP global; the intent was to
"spread" inertia into the null-space. But `−log det P` is **monotone increasing and
unbounded above**: for a data-free mass direction the optimum is `+∞` for *any* γ>0.
Empirically (γ=0.01, CLARABEL) the fit broke (REL 0.97) and masses 1–2 ran to
≈8200 kg. This is an instructive negative result worth a paragraph: maximum entropy
alone is not a usable realisation selector here — it needs a scale.

*Adopted form — bounded log-det (Bregman) divergence.* The objective term is
`+γ·Σ_i [ tr(P0⁻¹·P_i) − log det P_i ]`, the Bregman divergence of `−log det` to a
reference `P0` (Wensing [53]). It is convex, ≥0, and minimised at `P_i = P0`; the
extra linear `tr(P0⁻¹·P)` is the missing **upper bound** that pulls masses *down*
toward the `P0` scale, while `−log det P` still keeps them *off* zero. `P0 =
diag(ref_inertia·I₃, ref_mass)` is a **generic isotropic blob — the same
uninformative shape for every link, explicitly NOT the manufacturer/CAD inertials**;
it supplies only an order-of-magnitude scale (`--ref-mass` 0.5 kg, `--ref-inertia`
1e-3 kg·m²), so the "from data alone" claim holds (cf. the no-CAD-prior argument
above). The data still determines the identifiable directions; the divergence only
fixes the *non-unique realisation* of the rest.

*Reporting policy.* γ is a free hyper-parameter (no data-driven optimum): pick the
**smallest γ** that lifts all masses to plausible, non-degenerate values while the
torque fit (REL) stays ≈ unperturbed, and state explicitly that only the *base*
parameters are claimed identified — the per-link masses are an entropy-regularised
realisation at a generic scale, not a measurement. Independent corroboration that
the underlying model is sound: the γ=0 model already **beats** the manufacturer
`vx300s.urdf` baseline on held-in torque prediction (friction-fitted mean RMSE
0.945 vs 2.682 Nm, −65%; MAE 0.406 vs 0.753 Nm) — see the 2026-06-10 changelog.

**Refinement (2026-06-10): pure −log det is unbounded → use a bounded divergence
to a generic reference.** The first formulation added the raw maximum-entropy term
`−γ·Σ_i log det P_i`. This has a failure mode opposite to the γ=0 collapse:
`−log det P → −∞` as the eigenvalues of `P` grow, so against a *soft* (finite-`w1`)
data penalty the objective is **unbounded below** — the solver lowers cost simply by
inflating `P`, and the link masses/inertias **explode** rather than settle. γ=0
collapses masses to zero; large γ on pure −log det blows them up. Neither yields a
trustworthy per-link realisation; both are symptoms of the same underlying fact
(the standard parameters are free in the data null-space, so *something* must pin
their scale).

The principled fix is to pin the scale with a **generic isotropic reference** rather
than leave it free: replace `−log det P` with the **log-det Bregman / Stein
divergence to `P₀ = s·I₄`** (Wensing & Slotine, *Geometric robot dynamic
identification: A convex programming approach*, T-RO 2018),

`D(Pᵢ, P₀) = tr(P₀⁻¹Pᵢ) − log det(P₀⁻¹Pᵢ) − 4`,

which is convex, ≥ 0, and uniquely minimised at `Pᵢ = P₀`. It keeps the entropy
(−log det) curvature that lifts masses off zero but adds the `tr(P₀⁻¹Pᵢ)` term that
**bounds them from above**, so the realisation is non-degenerate *and* finite for any
γ. Crucially `P₀ = s·I₄` is the **same generic unit-scale blob for every link** — a
single regularisation scale `s ≈ 0.5 kg`, **not** a per-link manufacturer/CAD prior —
so the thesis-integrity argument above is untouched: physics + a neutral scale, never
the answer we are trying to identify. The reporting policy is unchanged: pick the
smallest γ giving plausible, non-degenerate masses, with REL essentially unperturbed.

**Why we do not hard-constrain Σmᵢ to a weighed total (the obvious alternative).**
Constraining the summed link mass to a *measured* total arm mass would be the most
defensible scale source — it is a real measurement of *this* robot, not a model — and
would still let the entropy term distribute mass across links. We keep it as an
optional upgrade rather than the default for one practical reason: it requires a
physical measurement we do not currently have. If/when the arm is weighed, adding
`Σ_i m_i = M_measured` as a single linear equality is a strict improvement and should
be reported as such.

**Sequencing: validate torque before tuning the realisation.** Because RNEA
inverse-dynamics torque depends *only* on the base parameters, the realisation choice
(γ, `s`, or a mass constraint) **cannot change the validation metric (REL / torque
RMSE)** — two URDFs with identical base parameters predict identical torques. The
realisation is therefore a *control-phase* artifact (needed to emit a usable URDF),
not an identification-phase one. The agreed order is: (1) γ=0, floor the degenerate
masses just to emit a URDF, and confirm torque prediction (REL ≈ 0.59) vs the 0.753 Nm
baseline — proving the base identification is sound and that realisation is downstream
of validation; then (2) enable the bounded divergence with small γ, verifying REL is
unmoved. If REL shifts when γ is switched on, γ is too large.

### Open question / candidate future-work (DEFERRED — discussion material, not doing now)
Two distinct excitation facets, both worth a thesis paragraph:

1. **Base-parameter conditioning is fine.** The unconstrained least-squares fit
   reaches REL ≈ 0.59 (vs the paper's ≈ 0.43) with base-regressor cond ≈ 197, so
   the *identifiable combinations* are well conditioned — the REL gap is data/model
   fidelity (friction model, `q̈` estimation noise, workspace coverage), not
   base-param excitation.
2. **Per-link inertia excitation is insufficient (new finding, 2026-06-10).** The
   final model's per-link inertias and several masses sit at the generic
   regulariser blob (`~0.002 kg·m²`, `0.5 kg`) because this single trajectory does
   **not excite the individual inertial parameters** — only gravity (first moments
   `mc`) and friction are actually determined by the data. So while the model
   predicts *this* trajectory's torque well (and beats the manufacturer baseline),
   the inertia split is a prior-chosen realisation, not identified. Sufficient
   acceleration-rich, workspace-covering excitation is what would change that.

**Candidate next contribution (kept as potential, not started):** implement the
paper's **Eq. 11 condition-number-minimising excitation trajectory** (the one part
of the paper this repo deliberately skips) and/or a richer friction model. This is
the natural lever to (a) push REL 0.59 → ~0.43, (b) actually *identify* the per-link
inertias rather than fill them from the blob, and (c) enable a **held-out**
validation (identify on one excitation run, validate on another) — the single
biggest credibility upgrade, since all current validation is held-in. Strong
methodological discussion either way: even reporting *why* a non-optimised
trajectory under-identifies inertias (and how the feasibility + entropy machinery
degrades gracefully to a torque-faithful model regardless) is a result in itself.

---

## Encoder velocity vs differentiated position; and dropout removal (2026-06-11)

An examiner will reasonably ask two things about the data pipeline: *"why obtain q̇
by differentiating position rather than using the joint's measured velocity?"* and
*"did you clean the communication dropouts?"* Both were investigated empirically.

**What the paper does.** Momani & Hosseinzadeh measure q and q̇ from
`/vx300s/joint_states` and obtain q̈ by a single forward difference of the *measured*
q̇, then zero-phase low-pass (`filtfilt`, 10 Hz). Their data is **200 Hz**.

**What we do, and why it differs.** Our identification differentiates *filtered
position* once for q̇ and again for q̈ (both stages low-passed). We tested switching to
the encoder's reported `*_vel` register (`--use-measured-vel`, opt-in, default off).
A direct consistency check on the delivered run (measured `*_vel` vs differentiated
position, both filtfilt-10 Hz) showed the encoder velocity is **not** a faithful
derivative at our sample rate:

- **Uniform lag ≈ 43 ms** (≈2 samples at 46.7 Hz) on every joint — the servo computes
  velocity over a trailing window, so it trails position.
- **Magnitude attenuation**: slope(measured/differentiated) ≈ 0.41 (waist), 0.48
  (forearm_roll), 0.44–0.46 (wrists), 0.94 (elbow); correlation as low as 0.63
  (waist). The register underreports velocity by ~50–60 % on the distal joints.

Consequence: using `*_vel` *lowers* the unconstrained REL (0.59 → 0.48) but this is an
**artifact** — an attenuated, lagged q̇ reduces the Coriolis/viscous/inertial torque
the model must account for (smaller residual) without representing true motion, and
the q(position)/q̇(register) phase mismatch inflates the shoulder/elbow constant
offsets F0 to ±2–2.6 Nm. The *feasible* REL actually worsens (0.60 → 0.63). **We
therefore keep differentiated position** as the more physically faithful estimator,
accepting the differentiation noise as the lesser evil at this sample rate. This is a
deliberate, evidenced deviation from the paper's measured-q̇ approach, forced by the
quality of the Dynamixel velocity register at ~47 Hz.

**Dropout removal.** The sync-read occasionally returns the sentinel −π on all six
joints at once (22 of 40 338 rows in the delivered run). Removing them
(`--drop-glitches`, opt-in) is **negligible** — 0.05 % of scattered samples cannot
move a global least-squares/SDP fit (unconstrained REL 0.5806 vs ~0.59; feasible
0.6016 vs 0.5997). It is retained as a defensible data-hygiene option, not as a result.

**A related conditioning observation.** The constant offset F0 on the gravity-loaded
shoulder and elbow is poorly identified — it swings to −3.30 Nm merely from removing
the 22 dropout rows (q̇ unchanged). This is gravity-vs-offset collinearity on those
joints, and is a candidate for a richer/better-constrained friction-offset treatment.

**The unifying takeaway → 200 Hz re-collection.** Both the differentiation-noise
problem (drives the q̈ → inertia estimate) and the encoder-velocity lag/attenuation
share one root cause: the **~47 Hz sample rate**, far below the paper's **200 Hz**.
At 200 Hz, differentiated position is far cleaner (so q̈ — hence the inertial terms —
becomes identifiable rather than collapsing to the regulariser blob) *and* the
velocity register's fixed ~2-sample lag becomes a much smaller fraction of the signal
period. A higher-rate re-collection is therefore the single change that addresses the
preprocessing limits *and* the per-link-inertia under-identification (see the
"excitation" open-question entry above) at once — the recommended next experiment.

## Recorder instrumentation for the 200 Hz collection (2026-06-11)

For the 200 Hz re-collection a new recorder (`record_joint_states_200hz.py`) was
written rather than reusing `record_joint_states.py`. The original is kept untouched
(reproducibility of past runs); the new one changes three things, each defensible to
an examiner as *instrumentation* fidelity — none of them touch the model or the
identification pipeline:

1. **Time base: publisher header stamp, not subscriber wall clock.** The old
   recorder timestamped rows at callback execution time, which adds DDS transport +
   executor scheduling jitter on top of the true sample time. That jitter matters
   here specifically because q̇ and q̈ are obtained by *differentiating against the
   time column*: an error ε in a timestamp perturbs the local dt and is amplified
   twice on the way to q̈ — at 200 Hz (dt = 5 ms), a 1 ms scheduling delay is a 20 %
   local dt error. `msg.header.stamp` is written by the driver's serial read loop,
   i.e. as close to the actual sample instant as is available without firmware
   changes. (Fallback to wall clock, loudly warned, if a driver does not stamp.
   The arrival time is still recorded in a trailing `recv_time` column so the
   stamp-vs-arrival jitter is auditable in the thesis.)
2. **No throttle — every message is recorded.** The old recorder's rate throttle
   compared each message against the last *written* row and could lock into
   dropping every other message when recorder rate ≈ publish rate (the runbook
   Step-4 caveat). Decimation, if ever wanted, belongs in analysis (`--stride`),
   not at capture time: data discarded at capture is unrecoverable.
3. **QoS matched to sensor data** (BEST_EFFORT, KEEP_LAST 50). A BEST_EFFORT
   subscription is compatible with either publisher reliability setting, and the
   deeper queue (~250 ms at 200 Hz) rides out transient subscriber stalls instead
   of silently dropping the newest messages.

The raw-data principle is preserved: the recorder writes what the driver reports,
unmodified (dropout sentinels included — it *counts* them live but does not remove
them; removal remains the analysis-side `--drop-glitches` decision). Likewise the
collection still uses the unmodified `run_trajectories.py` excitation (seed 42) that
produced the delivered model, so a rate-only comparison between the 47 Hz and 200 Hz
datasets stays clean: same excitation, same pipeline, different sample rate — the
defensible A/B for the "was 47 Hz the bottleneck?" question.

## Cross-run validation and the 200 Hz re-identification puzzle (2026-06-12)

**The experiment.** With the 200 Hz dataset collected
(`traj_run_200hz_20260612_131613.csv` — same seed-42 excitation, same pipeline,
4× the sample rate of the May run), we ran the full cross-validation matrix:
{factory `vx300s.urdf`, delivered May model `cfg-640cb8ef`, new 200 Hz model
`cfg-a92e984c` (same recipe, `--stride 4 --drop-glitches`)} × {May data, 200 Hz
data}, all with `compare_urdf_performance.py --friction --drop-glitches`
(the flag was added to the validation harness this day; sentinel dropout rows
otherwise smear spurious q̈ spikes through the zero-phase filter and dominate
RMSE — they cost the earlier *contaminated* held-out run a misleading
"61 % worse" verdict driven almost entirely by outliers, RMSE/MAE ≈ 8).

**Results (friction-fitted mean RMSE / MAE, Nm):**

| model \ data        | May 47 Hz run        | 200 Hz run            |
|---------------------|----------------------|-----------------------|
| factory vx300s.urdf | 2.066 / 0.649        | 0.719 / 0.569         |
| May model (delivered)| 0.645 / 0.355 held-in| **0.438 / 0.313 held-out** |
| new 200 Hz model    | 2.355 / 0.488 held-out| 0.460 / 0.323 held-in |

**Finding 1 — the delivered model generalises.** Held-out on 15 min of unseen,
cleaner data it beats its own held-in numbers (plausible: the 200 Hz q̈ is less
noisy, so the *evaluation* is cleaner) and the factory model by 39 %. Mean R²
is positive (+0.52). This is the dissertation's cross-validation evidence; the
control phase proceeds on this model.

**Finding 2 — re-identifying on the 200 Hz data under the unchanged recipe
produced a worse model**, dominated even on its own training data (0.460 vs
0.438). The new fit moved the upper-arm inertias from blob scale to
0.17–0.2 kg·m² and acquired a 0.46 m forearm CoM — and its waist-axis
prediction degrades *everywhere* (held-in waist R² −4.95 vs factory +0.47).

**Interpretation (hypothesis, untested).** The 47 Hz data low-passed away most
genuine acceleration-correlated torque, so inertia-direction parameters were
regulariser-dominated. The 200 Hz data restores that signal, and the fit
attributes it to *link* inertia. But the physically dominant acceleration-
correlated effect at the joint is likely **reflected actuator inertia**
(rotor + gearhead at ≈270:1; τ_measured accelerates the rotor too), which is
*local to each joint axis*. A rigid-body parameterisation can only absorb it
by inflating link inertia, which then couples (wrongly) into other axes via
RNEA — precisely the observed waist degradation. The standard remedy is a
per-joint diagonal motor-inertia term `Ia·q̈ᵢ` in the regressor (linear in the
parameter, Ia ≥ 0; fits the SDP structure unchanged). This is the natural next
identification experiment, alongside a γ sweep to check how much of the
inertia inflation is regularisable away.

**Status.** Deliverable unchanged (May model, now cross-validated). The 200 Hz
dataset stands as the better validation set and the substrate for the
motor-inertia experiment.

### Resolution of the γ question (2026-06-12 evening): the defect is structural

The γ sweep (`sweep_gamma.sh`, γ ∈ {0.05…2.0} on the 200 Hz data; full table in
the CHANGELOG) answered the "regularisable or structural?" question cleanly,
and the answer is **structural** — but with a more interesting shape than
expected:

1. **The inertia inflation per se is regularisable.** By γ=0.5 the upper-arm
   inertia is pinned to the blob scale (~0.002 kg·m²), and held-in RMSE keeps
   *improving* monotonically with γ (0.460 → 0.416 at γ=2.0, beating the May
   model's 0.438 on the same data).
2. **Generalisation is not recovered.** Held-out RMSE on the May data is
   U-shaped with a floor of ~0.76 at γ=0.2–0.5 — never near the May model's
   0.645. So suppressing the inflated inertia does not remove whatever the fit
   was using it for; the unexplained torque simply **migrates into other
   parameters** (shoulder/elbow F0 offsets drift by ~0.1 Nm and 0.05 Nm across
   the sweep) and held-out worsens again at large γ.
3. **The waist defect reproduces on independent data.** Identifying on the
   16:10 replicate run under the same recipe again yields the best-ever
   held-in fit (0.375) *and* the same broken waist axis (R² −2.90 vs factory
   +0.30).

Reading: the 200 Hz data contains genuine acceleration-correlated torque that
the feasible rigid-body link parameterisation cannot represent — it can only
caricature it (inflated link inertia at low γ, distorted offsets at high γ),
and each caricature fits the training run while damaging transfer. This is
exactly the predicted signature of **reflected actuator inertia** (rotor +
≈270:1 gearhead, local to each joint axis): a link-inertia stand-in wrongly
couples into other axes through the RNEA, which is where the waist damage
comes from. For the dissertation, the sweep is a nice negative result: it
rules out "just tune the regulariser" and motivates the model-structure
extension on physical grounds.

A repeatability bonus from the replicate: the May model scores **0.438** mean
RMSE on *both* independent 200 Hz collections (identical to 3 decimals), and
factory drifts only 0.719→0.707 — so the validation methodology resolves
model differences down to ~0.01 Nm.

**Next experiment (decided):** add the per-joint motor-inertia term
`τᵢ ← … + Ia_i·q̈ᵢ` to the regressor — 6 extra columns, parameters linear and
constrained `Ia_i ≥ 0`, so the SDP structure is unchanged — then re-run the
identification + cross-validation matrix on the 200 Hz data. Success criterion
unchanged: beat 0.438 held-in *and* 0.645 held-out, with a sane waist axis.

## Reflected motor inertia: model extension and validation protocol (2026-06-13)

**The model.** With high gear ratios (≈270:1 on the ViperX-300) the
current-derived torque must also accelerate the motor's own rotor and gear
train. Seen from the joint, this is `Ia_i·q̈ᵢ` with `Ia = N²·J_rotor` —
plausibly 0.1–1 kg·m², i.e. potentially *larger* than the link inertias
(0.002–0.05 here). Crucially it is **local and configuration-independent**:
torque on its own axis only, proportional to its own q̈, identical in every
posture — whereas link inertia is posture-dependent and couples across joints
through the Newton–Euler recursion. A fit lacking the term can only mimic it
by inflating link inertia, which drags in spurious cross-coupling — the
mechanism behind the 200 Hz waist defect.

**Implementation choice.** Appended as 6 trailing regressor columns
(diag(q̈)), parameters linear with `Ia ≥ 0` — the pseudo-inertia LMIs and the
entropic regulariser are untouched, so the identification stays a convex SDP
with a unique global optimum. Opt-in (`--motor-inertia`), conditional config
key so the delivered May artifacts keep their hashes. This is a *deviation
from the source paper*, which does not model actuator inertia; the
justification is the γ-sweep negative result (above) plus the standard
treatment in the identification literature (motor/drive inertia as a diagonal
addition to the mass matrix).

**URDF export decision.** `Ia` is deliberately **kept out of the URDF
inertials**: a URDF can only attach inertia to a *link*, and a link-inertia
encoding of rotor inertia is exactly the falsehood the term exists to remove.
It travels in the artifact sidecar + a URDF comment (like F0) and belongs in
the controller as feed-forward `Ia·q̈` (pinocchio: `model.armature`).

**Validation protocol decision.** `compare_urdf_performance.py --fit-ia` adds
a per-joint q̈ column to the same least-squares nuisance basis as friction,
fitted per model per dataset — **symmetric across both models**, so the
comparison still isolates rigid-body (link-parameter) quality. The
alternative — using each model's *identified* Ia as fixed feed-forward — is
closer to deployment but asymmetric (the factory and May models have no
identified Ia) and entangles the link-parameter comparison with Ia-estimation
quality. Caveat to report: `--fit-ia` numbers are a *different protocol* and
must never be mixed with friction-only numbers in one table.

**First evidence (smoke runs, 2026-06-13; full runs pending).** A stride-40
identification on the 200 Hz data with the term came out feasible with the
upper-arm inertia at blob scale (no inflation) and shoulder Ia = 0.80 kg·m²
(dual motor). Independently, *fitting* Ia on the May model's 200 Hz residuals
gives shoulder Ia = 0.64 kg·m² — the same scale from a different estimator —
while the factory model's fitted shoulder Ia is negative (−0.77), as expected
if its CAD link inertias already overshoot. Two estimators agreeing on a
physically plausible magnitude is good preliminary support for the
hypothesis.

### Outcome (2026-06-13): hypothesis confirmed, incumbent not dethroned

The full experiment (CHANGELOG 2026-06-13 "Results") split the verdict in an
instructive way:

- **As physics, the term is vindicated.** The q̈ columns lower the
  unconstrained REL (shoulder 0.57→0.44), the identified Ia magnitudes are
  plausible and reproducible across three estimators, the upper-arm inertia
  inflation disappears at the *original* γ=0.05, and — the acid test — the
  **waist axis is healed**: the Ia model's waist beats even the May model's
  held-in waist on May data. The chain γ-sweep-negative-result → structural
  diagnosis → model extension → defect cured is a complete, defensible
  methodology arc for the dissertation.
- **As a model, it still loses the matrix narrowly** (0.343 vs 0.332 on
  200 Hz; 0.579 vs 0.520 on May), with the gap concentrated in the shoulder.
  Per the decision rule the May model remains the deliverable.

**Evaluation-bias self-audit (examiner material).** The margin must be read
with two asymmetries in mind, raised during the analysis: (1) every
hyperparameter (γ, w2, ref-mass, stride) was tuned on/for the May data — the
challengers run on borrowed settings, and the γ sweep proved the recipe is
dataset-sensitive; (2) the held-out arenas are unequal — the incumbent is
tested held-out on the clean 200 Hz data, the challengers on the May CSV with
its known residual channel glitches (waist, wrist_rotate R² −48 for *all*
models). Counterweights: tuning-overfit would predict the May model fails
held-out, and it does the opposite; and the γ sweep was a genuine tuning
campaign for the 200 Hz side that still produced no winner. Conclusion: the
May model's generalisation is real, but its *margin* is partly an artefact of
asymmetric tuning and asymmetric test sets.

**Open paths (either is defensible):** (a) a γ retune of the Ia recipe
(`sweep_gamma_ia.sh`) — the 3 % held-in deficit under borrowed
hyperparameters is not a closed case; (b) a different-seed 200 Hz collection
as a neutral held-out arena both models have never seen — the clean way to
crown a final winner; (c) accept the May model (twice cross-validated) and
start the control phase, reporting the Ia experiment as a confirmed
structural finding whose identified Ia feed-forward the controller can use
regardless.

### γ retune result (2026-06-13, later): statistical tie; Ia is γ-invariant

Path (a) was run (CHANGELOG "γ retune" entry). Two findings:

1. **The gap closes to within measurement resolution.** At γ=0.5
   (`cfg-ce8e7059`): 0.335 held-in / 0.526 held-out vs the May model's
   0.332 / 0.520 — both gaps below the ~0.01 Nm replicate repeatability. The
   borrowed-hyperparameter explanation of the earlier margin was therefore
   largely correct. Strictly, the decision rule is still not met (no point
   beats both numbers), so the May model remains the deliverable; the honest
   dissertation statement is that the two models are **indistinguishable on
   the available data**, with qualitative advantages on the Ia side (healed
   waist axis, explicit actuator model, presentable γ=0.5 realisation).
2. **Ia is invariant to the regulariser** — shoulder Ia drifts 0.1 % over a
   25× γ range while the link realisation changes drastically (γ=0.02: 1.9 m
   CoMs; γ=0.5: ≤ 2 cm). The data, not the prior, pins Ia. This cleanly
   separates the well-identified subspace (base parameters + Ia) from the
   realisation nuisance the regulariser resolves, and is the strongest
   identifiability evidence in the project so far.

**Methodological stop-rule.** Further tuning cannot break the tie: both
curves flatten short of the targets, and every additional evaluation against
the May CSV spends its held-out independence (this project has now queried it
many times — a multiple-comparisons liability an examiner may probe). The
tiebreaker must be **new data**: a different-seed 200 Hz collection, with the
decision rule **pre-registered** before any model touches it — lower mean
friction+Ia-fitted RMSE on the neutral set, May `cfg-640cb8ef` vs Ia-γ0.5
`cfg-ce8e7059`, one shot, no retuning afterwards.
