# DEMO PREP — supervisor meeting, Friday 2026-11-13

Goal of the meeting: (1) explain **how the calculations are made** (method),
(2) **show the controller running on the robot**, (3) show **how it maps to the paper**
(Momani & Hosseinzadeh, *Mechatronics* 112 (2025) 103419).

Talking points only. Numbers come from `docs/CHANGELOG.md` / `docs/RESULTS_INDEX.md`;
the entry each number comes from is named so you can open it if asked.

---

## 0. The 60-second story (open with this)

1. The ViperX-300 has **no manufacturer dynamic model**, and the CAD URDF predicts torque
   badly. Model-based control needs a model you can trust.
2. Following the paper, I **identified a physically feasible dynamic model from data**:
   excite the arm, record q and motor current at 200 Hz, fit the 13 parameters per link
   with a convex optimisation that only allows physically possible parameters.
3. I export the result as a **URDF** and check it on a **held-out run**:
   it is **55 % better than the factory CAD** on rigid-body torque and roughly
   **matches the paper's benchmark** (mean error 0.426 vs the paper's 0.392).
4. That URDF drives a **model-based controller on the real arm**: PD + gravity
   compensation in current (torque) mode, with forward kinematics, inverse kinematics
   and keyboard Cartesian teleop all computed from the **same URDF** via Pinocchio.
5. Next: computed-torque feed-forward and the paper's ERG constraint layer. M(q) and
   C(q,q̇) are already built and checked for that.

---

## 1. Workspace map — what lives where

### Folders in `~` (WSL2, Ubuntu 22.04, ROS 2 Humble)

| Path | What it is | Used in demo? |
|---|---|---|
| `~/interbotix_ws/` | Trossen/Interbotix ROS 2 workspace (driver `xs_sdk`, `interbotix_xs_msgs`, `interbotix_xs_modules`, descriptions, RViz). Installed by `~/xsarm_amd64_install.sh`. | **Yes**, it is the hardware driver |
| `~/interbotix_ws/src/cont/` | Early ROS 2 package (`cont.py`, cubic-polynomial to-target-and-back trajectory, April 2026). Not part of the current pipeline. | No |
| `~/Vx300s/` | **The thesis repo**: identification pipeline, controller, docs, thesis .tex. | **Yes** |
| `~/ViperX300-review/` | Trimmed review package (champion model + validation + controller + README) made for someone else to re-run. | Maybe: it is a good "here's what you can run yourself" handout |
| `~/apriltag_ws/` | AprilTag ROS workspace (not used in the current pipeline). | No |

### Inside `~/Vx300s/`

| Path | Role |
|---|---|
| `collect_200hz.sh`, `run_trajectories.py`, `record_joint_states_200hz.py`, `check_topic_rate.py`, `check_collection.py` | **Step 1, collect**: excitation trajectory + 200 Hz recorder + pre/post checks |
| `sysid_feasible.py` | **Step 2, identify**: Newton–Euler regressor, base parameters, convex SDP fit (no ROS needed) |
| `phi_to_urdf.py` | **Step 3, export**: parameter vector φ → URDF |
| `compare_urdf_performance.py`, `compare_gravity.py` | **Step 4, validate**: predicted vs measured torque (Pinocchio RNEA) |
| `pipeline_artifacts.py` | Caches every stage output by a config hash, so results never overwrite each other |
| `control/pd_grav_control.py` | **Step 5, the controller** (PD + gravity comp + optional I-term, FK, IK, teleop, float mode) |
| `control/dynamics.py` | Explicit M(q), C(q,q̇), G(q) of the identified model (built and verified, not yet in the loop) |
| `control/set_pos.py` | Move the arm to a joint pose in position mode (use before every controller run) |
| `tools/ik_solve.py` | Standalone IK preview/check |
| `tools/check_dynamics_matrices.py`, `tools/paper_dynamics.py` | M/C property checks + comparison with the paper's MATLAB M/C |
| `urdf/champion.urdf` | **The deliverable** (copy, with provenance header). `urdf/vx300s.urdf` = factory CAD baseline |
| `outputs/npy/`, `outputs/urdf/` | All identified φ vectors and URDFs (with `.json` recipe sidecars) |
| `data/` | Recorded runs (CSV) + controller logs (`pdg_*.npy`) |
| `external/paper_model/` | The paper authors' code: MATLAB M/C/G, their PD+G and ERG ROS nodes |
| `docs/` | HANDOVER, CHANGELOG, THESIS_NOTES, PAPER_SUMMARY, RESULTS_INDEX, CONTROL_ROADMAP |
| `thesis/` | LaTeX chapters (Overleaf is the master copy) |

### The champion model (say these file names if asked)

- φ: `outputs/npy/traj_run_200hz_20260624_124955__sysid_feasible-v1-5__cfg-27904c2e.npy`
- URDF: `outputs/urdf/…cfg-27904c2e__phi_to_urdf-v1-1__cfg-451881cf.urdf` (= `urdf/champion.urdf`)
- Identified on `data/traj_run_200hz_20260624_124955.csv` (900 s tour run), validated
  on `data/traj_run_200hz_20260623_145333.csv` (different run, never seen in the fit).
- Recipe: `--method cvxpy --w2 10 --entropic 0.005 --motor-inertia --solver CLARABEL --drop-glitches --stride 1`

---

## 2. System architecture — how the code talks to the robot

```
 Windows                       WSL2 (Ubuntu 22.04, ROS 2 Humble)
 ───────                       ──────────────────────────────────────────────
 U2D2 USB ─► usbipd attach ─►  /dev/ttyUSB0
                                   │
                                   ▼
                     xs_sdk driver  (ros2 launch interbotix_xsarm_control …)
                        │  publishes  /vx300s/joint_states   (q, q̇, current in mA)
                        │  listens    /vx300s/commands/joint_group  (6 currents, mA)
                        │  service    /vx300s/set_operating_modes  (position ↔ current)
                        ▼
              control/pd_grav_control.py  (rclpy node, timer @ 100 Hz)
                 ├─ Pinocchio on champion URDF: G(q), FK, Jacobian → IK
                 └─ control law → mA → publish
```

- 6 arm joints: `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`.
  Shoulder and elbow have **two motors** each (master + mirrored "shadow").
- Motors: Dynamixel XM540/XM430. No torque mode exists, so torque is commanded as
  **current**: `τ = (mA/1000) · 2.409 Nm/A · n_motors` (`EFFORT_SCALE` in `sysid_feasible.py`).
  The paper does the same (its Remark 8).

---

## 3. How the calculations are made (the method, step by step)

### 3.1 Kinematics
- Modified (Craig) D-H table, `DH_PARAMS` in `sysid_feasible.py`. Link lengths from the
  paper's Table 1 (L3/L4 split differently, but L3+L4 = 0.30002 m is what enters).
- Gotcha worth telling: the table was modified-DH but ran through a standard-DH chain at
  first, so the shoulder was a yaw axis and gravity could not be identified. Found and
  fixed 2026-06-13. Now proven: our regressor equals Pinocchio RNEA on the URDF to 3e-7 Nm
  (`tools/test_phi_urdf_consistency.py`), and FK axes match the real robot
  (`tools/test_fk_equivalence.py`).

### 3.2 Dynamic model (paper Eq. 1–4)
- `τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + τ_f`, friction `τ_f = Fv·q̇ + Fc·sign(q̇) + F0`.
- **13 parameters per link**: m, m·c (3), inertia tensor at the link frame (6), Fv, Fc, F0
  → 78 parameters. My extension: + 6 reflected motor inertias Ia (`--motor-inertia`) → 84.
- Key fact: the model is **linear in the parameters**: `τ = Φ(q,q̇,q̈)·ϑ` (Eq. 5).
  Φ (the regressor) is built by my own Newton–Euler recursion (`regressor_fast`).

### 3.3 Base parameters (Eq. 6)
- Not all 78 parameters affect torque independently. QR decomposition of the stacked
  regressor finds the identifiable combinations: **78 → 58 base parameters**
  (`find_base_parameters`). Fit those; the matrix L maps base ↔ standard parameters.

### 3.4 Excitation trajectory (Eq. 7, 11)
- Each joint follows a finite Fourier series, **Δf = 0.1 Hz, 5 harmonics**, 900 s
  (same as the paper). Coefficients chosen by SLSQP to **minimise cond(Φ_b)** under
  position/velocity/acceleration limits (the paper used `fmincon`).
- My additions: an **operating-point tour** (`--tour`) so the slow centre point also sweeps
  the workspace (waist coverage 64 → 93 %), a shoulder/elbow collision band, and rest-to-rest
  start. Commands are sent at ~6.7 Hz (servo interpolates); data is **recorded at 200 Hz**.

### 3.5 Data processing (paper §3.3)
- Current mA → Nm with `EFFORT_SCALE`. Dropout rows (all joints = −π) removed.
- 4th-order Butterworth, **zero-phase `filtfilt`, 10 Hz cutoff** on q and τ (paper: 10 Hz).
- q̇ and q̈ by numerical differentiation of filtered position (each filtered again).
  The paper uses the encoder velocity directly; I found that register lagged and
  ~50–60 % low on distal joints, so I differentiate position instead.
- Each joint's rows normalised by its max |τ| so small joints count (as in the paper).

### 3.6 Physically feasible identification (Eq. 16), the core
Solve (`identify_sdp`, CVXPY + CLARABEL solver):

```
min   w1·‖Φ_b ϑ_b − τ‖²  +  w2·‖ϑ_b − Lᵀϑ‖²   (+ γ·entropic regulariser)
s.t.  P_i(ϑ) ⪰ 0  for every link        (4×4 pseudo-inertia: covers m>0, I^c≻0, triangle ineq.)
      Fv_i ≥ 0, Fc_i ≥ 0, Ia_i ≥ 0
      sign constraints on first mass moments (16i)
```
- The paper lists the constraints separately (16b–f); I use the single **pseudo-inertia LMI**
  (Wensing, Kim & Slotine 2017, the paper's ref [53]), which implies all of them and keeps the
  problem **convex**, so there is one global optimum. The paper used YALMIP, also an SDP.
- **w2** (coupling weight) sets how hard the standard parameters must match the fitted base
  parameters. Sweep showed **w2 = 10** is the sweet spot (w2 = 100 overfits: shoulder mass 3.4 kg).
- **Entropic regulariser γ** (my addition): picks a non-degenerate realisation of the
  unidentifiable directions using a generic "blob" reference, **not the CAD model**, so the
  model stays data-only. γ = 0 is exactly the paper's Eq. 16.
- Result check: every link passes feasibility; total mass 2.62 kg vs CAD 2.54 kg.

### 3.7 Export and validation (paper §3.5, Eq. 19)
- `phi_to_urdf.py` writes mass, CoM and inertia per link into a URDF (M ≻ 0 follows from
  feasibility; also checked numerically: λ_min ≈ 0.009 kg·m², cond ≤ 48).
- `compare_urdf_performance.py` predicts torque with Pinocchio RNEA on the held-out run and
  compares with measured. Metric: the paper's REL `mean |τ − τ̂| / max(|τ|, |τ̂|)`.
- **Results (CHANGELOG 2026-06-24):**
  - Rigid-body torque RMSE: ours 0.602 Nm vs factory CAD 1.334 Nm → **−54.8 %**
    (CAD is only 9 % better than predicting zero).
  - REL per joint, ours vs paper's validation:

    | | waist | shoulder | elbow | forearm_roll | wrist_angle | wrist_rotate | mean |
    |---|---|---|---|---|---|---|---|
    | Ours (held-out) | 0.547 | 0.260 | **0.120** | **0.321** | 0.544 | 0.767 | 0.426 |
    | Paper | 0.333 | 0.168 | 0.200 | 0.540 | 0.381 | 0.731 | 0.392 |

  - Held-out (0.426) ≈ in-sample (0.477), so no overfitting.

### 3.8 The controller (`control/pd_grav_control.py`)

Per joint, in **mA**, every 10 ms (100 Hz):

```
u = Kp·(q_ref − q)  +  Ki·∫(q_ref − q)dt  −  Kd·q̇_filt  +  α·G(q)/EFFORT_SCALE
```
- **G(q)**: `pin.computeGeneralizedGravity` on the **champion URDF** (startup self-check:
  equal to the φ-vector gravity to ~1e-7 Nm). α = 1 = full identified gravity.
  forearm_roll uses its measured holding current as constant feed-forward instead
  (`--hold-ff`), because the model is weak there.
- **q_ref**: ramps from where the arm actually is after the mode switch to the
  target q_d over 3 s (`--recover-time`), so the controller never sees a large step.
- **q̇_filt**: low-pass (τ ≈ 30 ms) finite difference of position, with glitch rejection.
  Not the raw velocity register (too noisy/low for damping).
- **Ki**: optional (`--ki-scale`, default 0). Only after the ramp; anti-windup clamp.
- Gains at the default `--gain-scale 0.5`, converted to Nm for comparison:
  Kp ≈ [3.0, 8.4, 7.2, 1.4, 1.4, 0.7] Nm/rad, Kd ≈ [0.14, 0.39, 0.36, 0.07, 0.07, 0.04] Nm·s/rad.
- **Targets**: hold current pose (default), `--hold-pose` (joints), `--xyz` (Cartesian →
  **IK**), or `--teleop` (keyboard jog → IK each step).
- **FK**: `framesForwardKinematics`, frame `ee_link`. **IK**: damped least squares
  (Levenberg–Marquardt) on the frame Jacobian of the same URDF, clamped to joint limits.
  Round-trip 0.1 mm.
- **Safety layer**: soft joint limits (current toward a limit fades out over 0.2 rad),
  current caps [700, 1400, 1400, 400, 400, 300] mA, kill on tracking error > 0.5 rad or
  |q̇| > 2.5 rad/s (debounced). A kill or Ctrl-C **parks the arm in position mode**,
  never zero current (it would fall).

### 3.9 Hardware results so far (CHANGELOG 2026-06-18, THESIS_NOTES control section)
- Stable hold, elbow droop ≈ 0, jitter ≈ 0.0005 rad (0.03°). Elbow settles at −594 mA vs
  model −602 mA, so the identified gravity is right there.
- Dual-motor joints **can** be current-controlled (early "half torque" theory was false;
  the problems were controller bugs).
- Held an **untrained** pose (shoulder 0.52 rad, outside the training range) for 65 s →
  the physics-based model generalises.
- End-to-end: target [0.30, 0, 0.40] m → IK → hold. EE error ≈ **46 mm**, mostly a
  0.105 rad shoulder droop (shoulder gravity over-predicted, α* ≈ 0.4 at that pose). The
  I-term is the fix, implemented, sweep on hardware pending.
- Note: these runs used the previous champion (`cfg-9ef2c992`/`a92e984c`). The controller now
  defaults to `cfg-27904c2e`.
- **2026-11-13: teleop works as it should on the current champion (`cfg-27904c2e`).**
  Keyboard jogging of the end-effector with the identified-gravity hold underneath.
  This closes the 2026-06-24 open item.

### 3.10 Code walkthrough: how the model controls the robot (hold and teleop)

The model enters in **one place**: every control tick it computes the **gravity torque**
G(q) for the current joint angles and adds it as motor current to a PD controller.
Teleop uses the **same loop**; it only changes the target q_d, using IK on the **same URDF**.
All line numbers are in `control/pd_grav_control.py` unless another file is named.

**1. Loading the model**

| What | Where |
|---|---|
| Default φ vector (champion) | `:76` |
| Default URDF (champion) | `:81` |
| Load φ, zero the friction parameters (gravity only) | `:187-190` |
| Load the URDF into Pinocchio | `:202-204` |
| Gains Kp, Kd, Ki (mA/rad), scaled by `--gain-scale` / `--ki-scale` | `:89-99`, `:207-209` |
| Current caps and joint limits | `:103`, `:116-118` |

**2. How gravity is calculated**

- Torque in Nm, `gravity_Nm()` at `:270-274`:
  `pin.computeGeneralizedGravity(model, data, q)` = G(q) from the identified URDF.
- Converted to current, `gravity_mA()` at `:276-277`: `G(q) / EFFORT_SCALE`.
- `EFFORT_SCALE = 2.409/1000 · [1, 2, 2, 1, 1, 1]` in `sysid_feasible.py:276-278`
  (torque constant × motors per joint; shoulder and elbow have 2 motors).
- Fallback without ROS: `sysid_feasible.py:215` (`inverse_dynamics_phi`), own
  Newton–Euler with q̇ = q̈ = 0. At startup both methods are compared (`:631-633`) and
  agree to ~1e-7 Nm.

**3. Startup (same for hold and teleop)**

| Step | Where |
|---|---|
| Measure pose + holding current for 1.5 s (median) | `:602-611` |
| Setpoint q_d = current pose (unless a target is given) | `:612-613` |
| Abort if the setpoint is outside the joint limits | `:616-621` |
| Print gravity, URDF↔φ self-check, FK of the EE | `:622-636` |
| Switch the arm to **current mode** | `:651` |
| Send the measured holding current | `:653-655` |
| Start the control timer, 100 Hz → `step()` | `:656` |

**4. Every control tick: `step()` at `:396`**

1. Velocity: filtered finite difference of position (`:422`), not the raw velocity register.
2. Reference ramps from the start pose to q_d over 3 s (`:453`); `err = q_ref − q` (`:454`).
3. Safety kill if the error is > 0.5 rad or the speed is too high (`:459`).
4. **Model feed-forward** (`:477-478`): `ff = α · G(q)` in mA. forearm_roll uses its
   measured holding current instead.
5. Integral (optional, off by default): `:493`.
6. **Control law** (`:499`): `u = Kp·err + Ki·∫err − Kd·q̇ + ff`
7. Shaping (`:503-509`): blend from holding current, dual-motor gain, soft limits, current caps.
8. Send 6 currents (mA) to `/vx300s/commands/joint_group` (`:510`, `publish()` at `:381`).

**5. What teleop adds (only changes q_d)**

| Step | Where |
|---|---|
| Start target = FK of the current pose | `:675` |
| Read a key | `:703` |
| Key → jog direction (X/Y/Z) | `:327` |
| Move the target by one step (10 mm) | `:342` |
| **IK** on the same URDF | `:288-321` |
| Reject if unreachable or outside limits | `:344` |
| Accept: `self.q_d = q_ik` | `:347` |

- IK: FK `:308` → Jacobian `:315` → damped least squares
  `dq = Jᵀ(JJᵀ + λI)⁻¹·err` at `:320` → update and clamp `:321`.
- After the first 3 s ramp, each jog changes q_d directly. The PD term moves the arm a
  few mrad while gravity compensation keeps it up.

**6. Stopping**

Ctrl-C, `q`, or a kill → arm switches back to **position mode** (`:738`). Never zero current.

**Not used by the controller**
- Inertia and Coriolis (M, C): built in `control/dynamics.py`, not in the loop.
- Friction parameters: zeroed (`:190`).
- forearm_roll: measured holding current instead of the model.

**If asked about the engage dip:** the comment at `:648-650` says the holding current is
sent *before* the mode switch, but the code sends it *after* (`:651` switches, then
`:653-655` publishes). This may contribute to the known elbow dip at engage. Not changed yet.

---

## 4. Paper ↔ my implementation (the mapping table)

| Paper | What the paper does | What I do | Same / deviation, and why |
|---|---|---|---|
| Eq. 1–4 | Rigid body + viscous/Coulomb/offset friction | Same, + reflected motor inertia Ia·q̈ | Extension: improves fit, Ia stays physical |
| Eq. 5–6 | Linear regressor, QR base params | Own Newton–Euler regressor, QR (78→58) | Same |
| Table 1 | D-H parameters | Modified D-H, same lengths | Same (verified against URDF/Pinocchio) |
| Eq. 7, 11 | Fourier excitation, Δf 0.1 Hz, N=5, min cond(Φ_b), fmincon | Same series + SLSQP + workspace tour | Tour added: cond alone left the waist under-covered |
| §3.3 | 200 Hz, measured q̇, Euler q̈, 10 Hz filtfilt, joint normalisation | 200 Hz, q̇ from position, 10 Hz Butterworth filtfilt, normalisation | Deviation on q̇: velocity register lags/under-reads |
| Eq. 16 | Constrained LS, YALMIP | Convex SDP (pseudo-inertia LMI), CVXPY/CLARABEL, + entropic γ | Equivalent constraints in one LMI; γ only picks a realisation, no CAD prior |
| Eq. 16i | First-moment signs (mixed axes) | y-axis convention in the code | Flagged in PAPER_SUMMARY; check if asked |
| §3.5 | M ≻ 0 checked by 1° gridding | Checked on 200 random states + guaranteed by feasibility | Equivalent purpose |
| Eq. 19, Table 4 | REL benchmark | Same metric, held-out run | 0.426 vs 0.392; better on elbow and forearm |
| Eq. 20–21 | τ = −Kp e − Kd q̇ + Ĝ + F̂v q̇ + F̂0 + ν(q̇) | u = Kp e + Ki∫e − Kd q̇ + αĜ | No friction FF or ν term yet; ramped reference instead; optional I-term |
| Remark 8 | Torque → current mode | Same, plus bump-free handoff, ramped setpoint, safe park | Real-hardware engineering the paper does not describe |
| §4.2 | ERG (needs M(q), via MATLAB/TCP) | **Not yet.** M(q), C(q,q̇) built in Python and verified | Next step |
| §4.3 | 200 Hz control loop | 100 Hz | WSL2/usbipd bandwidth; enough for regulation |

Observation (CHANGELOG 2026-10-06): the paper's published M(q) is 10–29× ours on the diagonal and
makes held-out prediction worse on our arm. Present as an observation, **not** "the paper is wrong".

---

## 5. Demo runbook (Friday)

### Before you leave home / Thursday rehearsal
- [ ] Do one full rehearsal of section 5 on the real arm (see section 7 for open risks).
- [ ] Base clamped to the table (an early oscillation nearly tipped it).
- [ ] Charge laptop; power supply for the arm; U2D2 cable.
- [ ] Open these files in tabs: `docs/DEMO_PREP.md`, `docs/PAPER_SUMMARY.md`,
      `control/pd_grav_control.py` (scroll to `step()`), `sysid_feasible.py` (`identify_sdp`),
      `figures/urdf_compare_traj_run_200hz_20260623_145333.png`,
      `figures/gravity_compare_static_gravity_20260613_183554.png`.
- [ ] Backup if hardware fails: a phone video of a hold + teleop from the rehearsal, and
      the offline checks in step 5.6.

### 5.1 Connect the USB (PowerShell as administrator)
```powershell
usbipd list                       # find the U2D2 bus id (was 1-2 or 2-1)
usbipd attach --wsl --busid <id>
```

### 5.2 Terminal 1: start the driver
```bash
ls /dev/ttyUSB0 && sudo chmod 777 /dev/ttyUSB0
source ~/interbotix_ws/install/setup.bash
ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s port:=/dev/ttyUSB0
```
Expect: RViz with the arm, no `Failed syncRead` spam. If spam: re-seat USB / re-attach.

### 5.3 Terminal 2: environment + pre-flight
```bash
source ~/interbotix_ws/install/setup.bash   # also sources /opt/ros/humble → real Pinocchio 3.9
cd ~/Vx300s
ros2 topic hz /vx300s/joint_states           # expect ~200 Hz; Ctrl-C
```
(Without ROS sourced, `import pinocchio` finds a broken PyPI package and the controller
falls back to φ, and `--xyz`/`--teleop` refuse to run.)

### 5.4 Demo sequence (safest first, stop when you have shown enough)

**A. Clean start pose** (always, before every run):
```bash
python3 control/set_pos.py 0 -0.6 0.5 0 0 0
```

**B. Gravity-compensated hold**, the core result:
```bash
python3 control/pd_grav_control.py
```
- Point at the startup print: setpoint, holding current, `gravity_mA(q_d)`, the URDF↔φ
  check (~1e-7 Nm), FK EE position. **If any joint's setpoint differs from the real pose by
  ~0.5 rad, Ctrl-C.**
- Say: "it's now in current mode; every 10 ms it computes gravity from my URDF and adds PD."
- Gently push a link by hand: it resists and returns. Ctrl-C → parks in position mode,
  prints the steady-state droop.

**C. (Optional) Show the model is doing the work, α comparison**:
```bash
python3 control/set_pos.py 0 -0.6 0.5 0 0 0
python3 control/pd_grav_control.py --alpha 0.0 --duration 15   # PD only → visible droop
python3 control/set_pos.py 0 -0.6 0.5 0 0 0
python3 control/pd_grav_control.py --alpha 1.0 --duration 15   # with model gravity
```
Compare the printed droop. This is the clearest visual argument for the identified model.

**D. Cartesian target via IK**:
```bash
python3 control/set_pos.py 0 -0.6 0.5 0 0 0
python3 control/pd_grav_control.py --xyz 0.30 0 0.40
```
Startup prints IK converged, error in mm and the joint solution; the arm ramps there.

**E. Keyboard teleop** (confirmed working 2026-11-13):
```bash
python3 control/set_pos.py 0 -0.6 0.5 0 0 0
python3 control/pd_grav_control.py --teleop        # no --xyz/--hold-pose!
```
Arrows = ±X/±Y, space = up, `x` = down, `+`/`-` = step size, `q` = quit. Wait for the
3 s ramp before jogging. The key help stays on screen; one status line underneath shows
the step size and EE target. "BLOCKED" = target outside reach or limits (this is a safety
check, not an error).

**Wrist cable:** always start with wrist_rotate = 0 (as in the commands above). Do not
move the wrist toward ±π to relieve cable tension: that broke the cable on 2026-11-13.

### 5.5 If something goes wrong
| Symptom | Cause | What to do |
|---|---|---|
| `KILL: |q_ref−q|>0.5` on one joint at start | Setpoint ≠ actual pose, or joint resting against a stop | Ctrl-C done already; `set_pos` to clean pose; run without a target |
| `KILL: |q̇|>2.5` | Real fast motion or timing glitch | Re-pose, re-run; keep gains default |
| Elbow dips ~0.5 rad at engage | Known mode-switch transient (motors briefly limp) | Recovers in ~2 s, explain it (THESIS_NOTES) |
| Gripper warm | Gripper stays energised while the arm runs (open TODO) | Keep runs short |
| `waiting for set_operating_modes …` forever | Driver not running / not sourced | Check terminal 1 |
| `syncRead` errors in driver | USB/usbipd link | Re-attach USB, restart driver |

### 5.6 Offline part (no robot needed, good while explaining the method)
```bash
source /opt/ros/humble/setup.bash && cd ~/Vx300s
python3 tools/test_phi_urdf_consistency.py            # regressor == URDF RNEA (~3e-7 Nm)
python3 tools/ik_solve.py --xyz 0.30 0 0.40           # IK preview
python3 tools/check_dynamics_matrices.py --csv data/traj_run_200hz_20260623_145333.csv
python3 compare_urdf_performance.py --friction --drop-glitches --stride 4 \
    --csv data/traj_run_200hz_20260623_145333.csv --urdf-b urdf/champion.urdf --plot
```
(The last one can take a while. Run it before the meeting and keep the plot open.)

---

## 6. Likely questions, with short answers

- **Why is the model "physically feasible" and why does it matter?** Each link's pseudo-inertia
  matrix is PSD, so masses are positive, inertias realisable, M(q) ≻ 0. An infeasible model
  can fit data but give a non-invertible M or nonsense extrapolation, which breaks control.
- **Why base parameters?** Some parameters never affect torque or only in combinations;
  fitting them directly is ill-posed. QR finds the 58 combinations the data determines.
- **How do you know it's not overfitting?** Validated on a different recording; held-out
  error ≈ in-sample error; and the controller holds poses outside the training range.
- **Why not use the CAD URDF?** On held-out data it is only 9 % better than predicting zero torque.
- **Why PD + gravity and not computed torque?** For holding and slow moves, PD+G is provably
  stable for setpoints (Takegaki & Arimoto) and needs only G. Inertia + Coriolis are ~13 % of
  gravity on shoulder/elbow even at excitation speeds. M and C are ready for the
  computed-torque comparison.
- **Why current, not torque?** Dynamixels have no torque mode. Current ≈ torque/k_t,
  shoulder/elbow split over 2 motors. Same as the paper's Remark 8.
- **Why is the EE off by ~46 mm?** Shoulder gravity is over-predicted (pose-dependent, points to
  the shoulder first mass moment). Fixes: I-term (implemented) or re-identify the shoulder.
- **What's the 0.58 / 0.63 scale anomaly?** Static holding current reads below model gravity;
  hypothesis: gear stiction holds part of the load at rest. Test tooling exists
  (`collect_stiction_hysteresis.sh`), not yet run.
- **Where do you differ from the paper?** Section 4 table. Short version: convex SDP with one LMI,
  entropic regulariser (no CAD prior), motor inertia, position-differentiated velocity,
  workspace tour, 100 Hz loop, no ERG yet.
- **How does your M compare with theirs?** Ours passes symmetry, M ≻ 0 and skew-symmetry checks, and
  improves held-out prediction (scale ≈ 1). Theirs is ~10× larger and worsens prediction on our arm.

---

## 7. Open issues to be upfront about

1. **Teleop: now works as it should** (2026-11-13, current champion). The 2026-06-24 kill on
   forearm_roll was a bad start pose (target ≠ actual pose), not a controller fault.
   Also on 2026-11-13: the wrist cable broke after running with wrist_rotate near ±π; it has
   been replaced and the start pose is back to wrist_rotate = 0.
2. **Gripper heats up** during control (stays energised). Not fixed yet.
3. **Shoulder gravity over-prediction** → EE error ~46 mm; I-term sweep pending.
4. **ERG and computed torque not implemented** yet (M, C ready).
5. Waist and wrist_angle REL trail the paper (likely friction model).
6. Gravity benchmark (`compare_gravity.py`) not re-run on the newest champion.

---

## 8. Suggested meeting order (~30 min)

1. Story (section 0), 2 min.
2. Architecture diagram + workspace (sections 1–2), 3 min.
3. Method (section 3.1–3.7), with the paper table (section 4) beside it, 10 min.
4. Live demo B → C → D → (E), 10 min.
5. Open issues + next steps (section 7), ask for direction on: ERG vs computed torque first,
   and how much more identification work is worth doing on the shoulder. 5 min.
