# CLAUDE.md — Project context & working agreement

Context for Claude Code (and collaborators) when working in this repository.
This is a **master's-thesis research codebase**, so correctness, traceability,
and clear documentation matter more than shipping speed.

## What this project is

Physically-feasible **dynamic model identification** of the Trossen Robotics
**ViperX-300 6-DoF** arm. The work is **based on a scientific paper**:

> Momani & Hosseinzadeh, *"Physically feasible dynamic model identification and
> constrained control of robotic arms: A case study on the ViperX-300 6-DoF
> robotic manipulator"*, **Mechatronics 112 (2025) 103419**.

The full paper text is at **`Paper.txt`** (~14.5k tokens, full incl. ERG control
§4–5 and references). **Do not load it for routine context** — read the
condensed **`docs/PAPER_SUMMARY.md`** instead
(method, equations, constraints, limits, benchmark REL values). Only open
`Paper.txt` when you need exact wording, a derivation, or a number not in the
summary. If you rely on something from `Paper.txt` that the summary lacks, add
it to the summary so the next session needn't reopen the paper.

The pipeline goes from recorded joint data → identified dynamic parameters → a
URDF usable in Pinocchio for model-based control/simulation:

```
τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + τ_friction
```

## Thesis scope & current phase

Master's thesis: **"Model Identification and Control of a ViperX-300 Robotic
Manipulator."** Scope spans **both** halves of the source paper:
1. **Identification** — produce a physically-feasible dynamic model / URDF.
2. **Control** — a model-based controller (the paper's robust law + ERG, see
   `docs/PAPER_SUMMARY.md` §10) built *on top of* that model.

**Current phase: identification.** The control work is deferred until a
**validated URDF** exists — a model good enough to trust is the prerequisite for
model-based control. So for now, prioritise getting identification right
(correct units, good excitation/conditioning, low held-out torque-prediction
error via `compare_urdf_performance.py`); treat ERG/control as the next phase,
not the current one. `control/trq.py` already does the torque→current mapping the
controller will need.

## Pipeline at a glance

1. **Collect** trajectory data on hardware (`control/`, `collect_*`, `run_trajectories*`)
   → CSV in `data/` with columns `<joint>_{pos,vel,effort}` for the 6 arm joints (+ fingers).
2. **Identify** parameters with feasibility constraints — `sysid_feasible.py`
   (`run_identification`). Produces a parameter vector `phi` (13 params × 6 links).
3. **Export** `phi` → URDF — `phi_to_urdf.py`.
4. **Validate** the URDF against recorded torque — `compare_urdf_performance.py`
   (RNEA inverse-dynamics prediction vs. measured torque).

`pipeline_artifacts.py` caches stage outputs by a config hash; bump the relevant
`PIPELINE_VERSION` when an algorithm or its inputs change, so artifacts recompute.

## Conventions & hardware facts (do not silently violate)

- **Joint order** (always): `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`.
- **Effort units:** the CSV `*_effort` columns are Dynamixel **present current in
  mA** (master motor). Torque: `τ = (effort/1000)·k_t·n_motors`,
  `k_t = 2.409 Nm/A` (XM540-W270, geared). **Dual-motor joints:** shoulder & elbow
  (`n_motors = 2`). See `EFFORT_SCALE` in `sysid_feasible.py` and the 2026-06-10
  changelog entry. Never reintroduce the old `STALL_TORQUE/100` scaling.
- **Base:** the arm is fixed-base. The full `vx300s.urdf` has `nq=10` (incl.
  gripper/fingers); identified URDFs are arm-only (`nq=6`). Match joints **by
  name**, not index.
- **Units:** SI everywhere (m, kg, rad, s, Nm, A).

## How to run

```bash
# Identify (writes phi + URDF artifacts)
python3 sysid_feasible.py --csv data/<run>.csv

# Validate a URDF against recorded torque
python3 compare_urdf_performance.py --urdf-b urdf/<candidate>.urdf \
    --csv data/<run>.csv --friction --plot
```

Dependencies: `numpy scipy matplotlib pandas` + `pinocchio` (sim/validation).

**Pinocchio / ROS gotcha (important).** The validation (`compare_urdf_performance.py`)
and `sim/` need the *real* Pinocchio (3.9.0, with `buildModelFromUrdf` / URDF
support), which is provided by **ROS 2 Humble**. Source it first:
```bash
source /opt/ros/humble/setup.bash
```
Without ROS sourced, `import pinocchio` resolves to a **bogus `~/.local`
`pinocchio 0.4.3`** (an unrelated PyPI name-squat) and fails with
`AttributeError: ... has no attribute 'buildModelFromUrdf'`. Recommended:
`python3 -m pip uninstall pinocchio` to remove the junk package. Note that
`sysid_feasible.py` does **not** use Pinocchio (own Newton–Euler regressor), so
identification runs fine without ROS — only validation/sim need it.

## Documentation rule (important)

This repo keeps thesis-facing docs under `docs/`:
- **`docs/CHANGELOG.md`** — chronological engineering log: *what changed, when*.
- **`docs/THESIS_NOTES.md`** — discussion material organised by topic: design
  decisions, **deviations from the source paper** and their justification,
  trade-offs, and open questions an examiner might raise.
- **`docs/PAPER_SUMMARY.md`** — condensed reference of the source paper.

**Whenever you make a substantive change, add a `CHANGELOG.md` entry** using the
template at the bottom of that file. "Substantive" = anything that changes
results, methodology, units/constants, model structure, or that an examiner might
ask about. Each entry must state the **motivation, the change, the evidence/
justification, and the impact (what must be re-run)**. Skip trivial edits.

**When a decision involves a methodological choice or a divergence from the
paper** (solver, model structure, constants, what we implement vs skip), also add
or update a topic entry in **`docs/THESIS_NOTES.md`** — that file is the raw
material for the dissertation's Methodology/Discussion chapters.

When in doubt, write it down — these docs are meant to become dissertation text.

## Working style here

- Prefer correctness and physical sanity-checks over speed; show the evidence
  (a regression, a plot, a magnitude check) for non-obvious claims.
- Keep changes minimal and in the existing style; this is research code others
  (examiners) will read.
- Flag assumptions explicitly rather than guessing silently (e.g. motor counts,
  torque constants).

## Who runs what (important)

**The pipeline scripts are run by the user, not by Claude.** This includes
identification (`sysid_feasible.py`), URDF export (`phi_to_urdf.py`), validation
(`compare_urdf_performance.py`), and anything else long-running or hardware-
facing (`control/`, `collect_*`, `run_trajectories*`). The user wants to watch
their progress live, so:

- **Do not launch these yourself** (no foreground, no background jobs). Instead,
  **give the exact command(s) to copy-paste**, note the relevant flags, and say
  what output to expect and what to send back.
- Claude *may* run quick, read-only analysis/sanity-check snippets (e.g. loading
  a CSV to print magnitudes, inspecting a URDF, regressions) — anything short and
  non-mutating. When in doubt, hand the command to the user.
- After the user runs a script and shares the result, Claude interprets it and
  proposes the next command.
```
