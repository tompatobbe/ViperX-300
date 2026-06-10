# Paper summary — condensed reference

Compact summary of the method this project implements, so the full text
(`Paper.txt`, ~8k tokens) does **not** need to be loaded for routine context.
Read this first; open `Paper.txt` only when you need exact wording, a proof, or
a number not captured here.

> Momani & Hosseinzadeh, *"Physically feasible dynamic model identification and
> constrained control of robotic arms: A case study on the ViperX-300 6-DoF
> robotic manipulator"*, **Mechatronics 112 (2025) 103419**. Open access (CC BY-NC).

## 1. Goal & contribution
A systematic, step-by-step procedure for **closed-loop identification** of a
robot arm's dynamic model that is guaranteed **physically feasible**, validated
on the ViperX-300 6-DoF (which has no manufacturer dynamic model). The paper
also designs an Explicit Reference Governor (ERG) constrained controller from
the identified model (Section 4, summarised in §10 below) as a second validation
of model quality.

**Authors' code & model** (ref. [29]):
`github.com/MomaniMutaz/ViperX-300-6DoF-Robotic-Arm-DynamicalModel` — their
MATLAB/Python `M, C, G` and identified parameters live here.

## 2. Dynamic model
Joint-space dynamics:  **τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + τ_f**   (Eq. 1)

Friction per joint (Eq. 2–4): viscous + Coulomb
**τ_fi = Fv_i·q̇_i + Fc_i·sign(q̇_i)** , with `Fv_i, Fc_i ≥ 0`.

## 3. Regression form & base parameters
Linear-in-parameters: **τ = φ(q,q̇,q̈)·ϑ** (Eq. 5), `φ ∈ R^(n×13n)`, `ϑ ∈ R^(13n)`.
**13 parameters per link**: mass (1) + first mass moments m·[X,Y,Z] (3) +
inertia tensor at link frame J (6) + viscous Fv (1) + Coulomb Fc (1) + motor
torque offset F0 (1). The F0 offset captures break-away/constant torque when
dynamic terms vanish.

Many standard parameters are redundant or non-identifiable → reduce to the
minimal **base parameters** ϑ_b via QR/linear regrouping: **τ = φ_b(q,q̇,q̈)·ϑ_b**
(Eq. 6). This makes the regressor full-rank for reliable least squares.

## 4. Excitation trajectory (Eq. 7)
Per joint, a finite Fourier series:
**q_i(t) = q0_i + Σ_{j=1}^{N_i} (1/(j·2π·Δf))·( a_ij·sin(j2π Δf t) − b_ij·cos(j2π Δf t) )**

Coefficients `q0_i, a_ij, b_ij` are chosen to **minimize cond(Φ_b)** (Eq. 11)
subject to joint angle / velocity / acceleration limits — good conditioning is
the key to accurate, non-degenerate estimates. Paper settings: **Δf = 0.1 Hz,
N_i = 5**, solved with MATLAB `fmincon` (active-set).

## 5. Physical feasibility constraints (Eq. 16)
Identify by solving (16a):
**min_{ϑ_b,ϑ}  w1·‖τ − Φ_b ϑ_b‖² + w2·‖ϑ_b − L·ϑ‖²**
where the 2nd term ties base params to standard params ϑ (so constraints
expressed in ϑ can be enforced); `L` is the base↔standard dependency map.
Subject to, for every link i:
- `m_i > 0`                                                  (16b)
- `J_i − m_i·(C_i)^×ᵀ(C_i)^× ≻ 0`  (pseudo-inertia / parallel-axis PD)  (16c)
- **Triangle inequalities** on eigenvalues λ of the inertia-at-CoM `I^c_i`:
  `λ1+λ2≥λ3`, `λ2+λ3≥λ1`, `λ1+λ3≥λ2`  → realizable mass distribution  (16d–f)
- `Fv_i ≥ 0`, `Fc_i ≥ 0`                                     (16g–h)
- **First-mass-moment sign constraints** (16i), ViperX geometry (§3.4):
  **m2·X2 > 0, m3·Y3 > 0, m4·Z4 < 0, m5·Y5 > 0, m6·Z6 > 0**
  *(note: these mix X/Y/Z axes per the paper; verify against the code, which
  uses a y-only convention).*

`M(q) ≻ 0` is **not** enforced in the optimization (would need fine gridding of
config space — intractable). Instead it's **checked afterward** (§3.5) by
gridding joint angles in 1° steps and confirming `M(q) ≻ 0` everywhere.

## 6. ViperX-300 specifics (§3)
- 6 revolute joints, open serial chain, ROS2 (`interbotix`). Identification data
  collected in **position control mode** (internal PID), subscribing to
  `/vx300s/joint_states` for q, q̇, τ.
- **D-H** (Table 1), link lengths `[m]`:
  `L1=0.12675, L2=0.30594, L3=0.1964, L4=0.10362, L5=0.07, L6=0.13658`.
  *(The code splits L3/L4 differently — 0.21981/0.08021 — but `L3+L4 = 0.30002`
  matches, which is what enters DH `d4`.)*
- Joint angle limits: `q1∈[−π,π], q2∈[−0.56π,0.56π], q3∈[−0.56π,0.51π],
  q4∈[−π,π], q5∈[−0.59π,0.72π], q6∈[−π,π]`.
- Vel/accel limits (Table 2, rad/s & rad/s²): J1 ±3.38 / ±277; J2 −1.58..3.36 /
  −186..349; J3 ±~3 / ±298; J4 ±3.36 / ±291; J5 −3.02..3.43 / ±407;
  J6 ±~4.8 / ±510.

## 7. Data processing
q, q̇ measured directly; **q̈ via first-order forward (Euler) difference of q̇**,
then **zero-phase low-pass filter (`filtfilt`, 10 Hz cutoff)**. Paper dataset:
`T = 180,000` samples at **200 Hz**. Torque rows + regressor rows are
**joint-wise normalized** by each joint's max |τ| so no single joint dominates
the fit. Solved with YALMIP.

## 8. Validation metric & results
**Relative Error Lenient (REL)** (Eq. 19) — robust when torques span large/small
magnitudes:
**REL = (1/N) Σ |τ_actual − τ_pred| / max(|τ_actual|, |τ_pred|)**

Reported REL (Table 4), excitation / validation trajectory (N_i=3 for validation):
| Joint | Excitation | Validation |
|---|---|---|
| 1 | 0.415 | 0.333 |
| 2 | 0.226 | 0.168 |
| 3 | 0.126 | 0.200 |
| 4 | 0.646 | 0.540 |
| 5 | 0.541 | 0.381 |
| 6 | 0.637 | 0.731 |

Joints 1–3, 5 predict well; joints 4 & 6 worse (small torques → noise-sensitive).
Error distributions ≈ zero-mean Gaussian (good for model-based control).
These REL values are the **benchmark to beat** for our own identified model.

## 9. Identified parameters (reference points, Table 3)
Friction (per joint): `Fv ≈ [0.126, 0.101, 0.040, 0.023, 0.017, 0.0016]` Nm·s/rad;
`Fc ≈ [0.239, 0.412, 0.261, 0.030, 0.061, 0.0030]` Nm;
offsets `F0 ≈ [−0.053, −0.203, −0.766, 0.012, −0.013, −0.0047]` Nm.
(Inertial base parameters are regrouped combinations — see Table 3 in `Paper.txt`
for the full list if exact values are needed.)

## 10. Constrained control via Explicit Reference Governor (§4)
A second validation of the identified model: build a model-based controller and
show it works on hardware. Two parts — a robust pre-stabilizing law, then an ERG
add-on for constraint satisfaction.

**Pre-stabilization (§4.1, Theorem 1).** Robust control law (Eq. 20):
**τ = −Kp(q − qv) − Kd·q̇ + Ĝ(q) + F̂v·q̇ + F̂0 + ν(q̇)**
where `Ĝ, F̂v, F̂0` come from the **identified** model, and the robustness term
`ν(q̇) = −β·q̇/‖q̇‖` (Eq. 21, zero at q̇=0). A Lyapunov/LaSalle proof gives
asymptotic convergence to `q†` with bounded steady-state error
`‖q† − qv‖ ≤ σ = ‖G̃(q†)+F̃0‖ / λmin(Kp)` (G̃, F̃ = identification errors). Notes:
identified **Coulomb** friction is deliberately *not* fed forward (sign → chatter);
`ν` covers robustness instead. Gains used: `Kp = diag{11.78, 20.95, 20.95, 7.85,
6.55, 0.57}`, `Kd = diag{0.52, 0.91, 0.52, 0.026, 0.13, 0.014}`,
`β = diag{0.5,0.5,0.5,0.1,0.1,0.1}`.

**Torque→current (Remark 8).** The ViperX has no native torque mode: desired
torques from (20) are mapped to **current-control** mode via
`/vx300s/set_operating_modes`, currents published to `/vx300s/commands/joint_group`.
(This is exactly what `control/trq.py` does in this repo.)

**ERG constraint enforcement (§4.2).** Linear joint constraints
`aᵢᵀ[qᵀ q̇ᵀ]ᵀ + hᵢ ≥ 0` (Eq. 29, time-invariant). The applied reference is
filtered: **q̇v = κ·Δ(q,q̇,qv)·ρ(qv,qr)** (Eq. 30), with
- **Dynamic Safety Margin** `Δ = Γ(qv) − V(q,q̇,qv) − ε` (Eq. 31); closed-form
  threshold `Γ` (Eq. 32–33) using a constant lower bound `M ⪯ M(q)` — chosen for
  real-time tractability (Remark 11).
- **Navigation Field** `ρ = ρ_a (attraction to qr) + ρ_r (repulsion from
  constraints)` (Eq. 35–37).

**Experiment (§4.3).** Constraints `q1 ≤ 80°`, `q3 ≥ −50°`. ERG params
`ε=κ=1e-3, η=0.01, ξ=0.8, δ=0.5`. Controller at **200 Hz**, reference updated
every **0.2 s**. Infeasible target `qr=[90,0,−60,0,0,0]°` → robot settles at the
best feasible `[79.5,0,−49.5,0,0,0]°` without violating constraints, confirming
the identified model is accurate enough for safety-critical control. Setup: i9-
13900K, WSL2 (Ubuntu 22.04 ROS2 node ↔ Windows MATLAB via TCP/IP for `M(q)`).
Demo video: `youtu.be/pkxozXyPYPs`.
