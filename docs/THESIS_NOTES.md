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
