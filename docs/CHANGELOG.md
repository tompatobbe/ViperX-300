# Engineering Changelog

A running, thesis-oriented record of **substantive changes to the code and
methodology** — what changed, *why*, the evidence behind it, and how it affects
results. The intent is that each entry can be cited or paraphrased directly in
the dissertation's "Implementation" / "Methodology" discussion.

Entries are newest-first. Each follows the template at the bottom of this file.

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
