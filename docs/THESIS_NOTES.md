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
regulariser** (log-det of the pseudo-inertia, Wensing [53]) — a candidate Phase-2
addition to `identify_sdp`, kept separate so the baseline result stays the exact
paper formulation (Eq. 16).

### Open question
The unconstrained least-squares fit on the current dataset already reaches only
REL ≈ 0.59 (vs the paper's ≈ 0.43) — and the conditioning of the base regressor
is good (cond ≈ 197), so this is **not** a base-parameter excitation/conditioning
problem. The residual gap points at data/model fidelity (friction model, `q̈`
estimation noise, workspace coverage). Implementing the Eq. 11
condition-number-minimising excitation trajectory (the part of the paper this
repo skips) and/or a richer friction model are the likely levers, and are
themselves strong methodological contributions to discuss.
