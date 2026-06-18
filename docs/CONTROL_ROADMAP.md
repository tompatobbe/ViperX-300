# Control Roadmap — commanding the ViperX-300 to any position

Forward-looking plan for the **control half** of the thesis. Companion to
`CHANGELOG.md` (what changed, when) and `THESIS_NOTES.md` (what to argue). This
file is the *target pipeline and the decision points*.

## The goal (user, 2026-06-18)

1. **Use the identified URDF to hold the arm in any position without collapsing.**
   → model-based **gravity compensation** (+ feedback). *Done in joint space —
   see `CHANGELOG.md` 2026-06-18 RESULT; stable holds, dual-motor joints included,
   and it even holds untrained poses.*
2. **Compute where the end-effector is** from the joint angles → **forward
   kinematics (FK)**.
3. **Command the end-effector to any position** → **inverse kinematics (IK)**:
   desired EE pose → joint targets, then drive/hold those joints with the
   controller from step 1.

> **Terminology note (for the thesis):** "calculate where the EE *is*" from joint
> angles is **forward** kinematics. "Move the EE *to* a desired pose" needs
> **inverse** kinematics. The user's step 2 is FK, step 3 is IK.

## Pipeline & status

| Stage | What | Status |
|---|---|---|
| 0 | Model-based current control on real hardware (dual-motor joints, mode-switch transient, filtered-velocity damping, safe stop) | **Done** — the hard infrastructure (CHANGELOG 2026-06-18) |
| 1 | Hold **any joint pose** without collapsing (gravity comp + PD) | **Done** in joint space; uses the identified φ (≡ URDF to 3e-7 Nm) |
| 1b | Drive gravity from **the URDF via Pinocchio** (so it literally uses *your* URDF) | **Code done** (`--gravity-source urdf`, default); matches φ to 1.3e-7 Nm; hardware re-confirm pending |
| 2 | **FK**: joint angles → EE pose | **Code done** — `ee_pose()` (`framesForwardKinematics`, frame `ee_link`); printed at startup |
| 3 | **IK**: desired EE pose → joint targets | **Code done** — `tools/ik_solve.py` (damped least-squares on the frame Jacobian); round-trip 0.1 mm; checks joint limits |
| 4 | Feed IK targets to the controller and **track** them | **Working (point-to-point)** — `ik_solve` xyz → `--hold-pose`; controller ramps the setpoint there. Smooth `q_ref(t)` trajectories = future refinement |

## Key unification: one URDF + Pinocchio gives all three pieces

The whole pipeline is exactly what **Pinocchio on your URDF** provides:
- `computeGeneralizedGravity` → the **gravity** term for step 1 (and the full
  `M, C, G` later if precise/fast tracking is wanted).
- `forwardKinematics` / `framesForwardKinematics` → step 2 (FK).
- Jacobians / IK loop → step 3 (IK).

So adopting **URDF-in-the-loop (Pinocchio)** isn't just narrative polish — it is
the natural backbone for steps 1b–3. (Numerically identical gravity to the φ
vector we use now, so it won't change the validated behavior.) Requires sourcing
ROS for the real Pinocchio (see `CLAUDE.md`).

## Decision: continue with the PD + gravity-compensation controller? — YES

PD+G is the right foundation for this exact goal:
- It **is** step 1 (already holds any pose without collapsing).
- It is the **tracking controller** for steps 3–4: IK produces joint targets, PD+G
  drives the joints to them and holds. No need to switch controllers.
- For holding and slow/moderate point-to-point moves (the stated goal), PD+G is
  sufficient. **Computed-torque** (adding `M(q)q̈ + C(q,q̇)q̇` from the same model)
  is an *optional later upgrade* only needed for fast, high-accuracy tracking —
  not required to "command the EE to any position" at sensible speeds.

## Known risks / caveats to carry forward
- **Shoulder first-moment error** (pose-dependent gravity over-prediction,
  THESIS_NOTES) → caps EE accuracy near the shoulder until the identification is
  improved there. Affects step 3 precision, not the ability to hold/move.
- **Mode-switch entry transient** on gravity-loaded joints → engage from a
  low-gravity pose / via a trajectory (already the working pattern).
- IK gives joint targets that may be near limits or in self-collision → the IK
  layer needs limit/collision awareness before commanding step 4.
- **`set_pos` to a clean pose before each run** remains an operational must.

## Immediate next step
Promote step **1b**: switch the controller's gravity source to Pinocchio on the
URDF, verify the hold is unchanged, then add `forwardKinematics` (step 2). That
single move puts the URDF at the center and sets up FK/IK in the same framework.
