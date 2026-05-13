# ViperX-300 Dynamic Model Identification

Python implementation of physically feasible dynamic model identification for the
[Trossen Robotics ViperX-300 6-DoF robotic manipulator](https://www.trossenrobotics.com/viperx-300),
following the methodology in:

> Momani & Hosseinzadeh, *"Physically feasible dynamic model identification and constrained
> control of robotic arms: A case study on the ViperX-300 6-DoF robotic manipulator"*,
> Mechatronics 112 (2025) 103419.

The pipeline goes from raw joint data (position, velocity, torque) to a physically consistent
dynamic model `τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + τ_friction`, suitable for use in model-based
control and Pinocchio/URDF-based simulation.

---

## Prerequisites

### Python packages

```bash
pip install numpy scipy matplotlib pandas
```

For simulation with Pinocchio:
```bash
pip install pin  # or follow https://stack-of-tasks.github.io/pinocchio/download.html
```

### Hardware (data collection only)

- ViperX-300 arm with Interbotix ROS 2 driver
- `interbotix_xs_modules` Python package (part of the Interbotix ROS 2 SDK)

---

## Repository structure

```
ViperX-300/
├── data/                          # Recorded CSV files
│   └── sysid_run1.csv             # Default input for sysid scripts
│
├── npy/                           # Saved identified parameter vectors (.npy)
│
├── urdf/
│   ├── vx300s.urdf                # Original Interbotix URDF
│   └── viper300_sysid.urdf        # URDF patched with identified inertials
│
├── Control/                       # Hardware control scripts (require ROS 2 + arm)
│   ├── vx300s_test.py             # Basic position control test
│   ├── pos_osc.py                 # Position oscillation
│   ├── vel_osc.py                 # Velocity oscillation
│   ├── trq.py                     # Torque control
│   ├── set_pos.py                 # Set joint positions
│   └── set_current.py             # Set motor current
│
├── sim/                           # Simulation and visualization (no hardware needed)
│   ├── sim.py                     # Interactive 3D DH visualizer (educational)
│   ├── robot_arm_sim.py           # 3D arm with joint sliders
│   ├── pinocchio_sim.py           # Pinocchio-based simulation
│   └── pinocchio_sim_clean.py     # Clean Pinocchio example
│
│── collect_arm_data.py            # Excitation trajectory mover + recorder (v1)
│── collect_arm_data_v2.py         # Excitation trajectory mover only (pair with recorder)
│── record_arm_data_manual_movement.py  # Record while manually back-driving the arm
│── record_joint_states.py         # Standalone joint state recorder
│── collect_joint_torque_vel_accel.py   # Torque/vel/accel collector
│
│── sysid_paper.py                 # Reference implementation (paper-accurate, slow)
│── sysid_subsample.py             # Option A: subsample fix — fast, minimal change
│── sysid_fast.py                  # Option B: vectorized NE — fastest, full dataset
│
│── phi_to_urdf.py                 # Convert phi.npy → URDF with identified inertials
│── plot_arm_data.py               # Plot joint data from CSV
│── plot_simple.py                 # Simple 4-panel position/velocity/accel/effort plot
└── visualize_arm_data.py          # Interactive joint data viewer with toggles
```

---

## Workflow

### 1. Collect excitation data (hardware required)

Start the recorder in one terminal:
```bash
python3 record_joint_states.py --duration 90 --rate 50 --output data/sysid_run1.csv
```

In a second terminal, run the sum-of-sinusoids excitation trajectory:
```bash
python3 collect_arm_data_v2.py --duration 90 --rate 50
```

Or use the combined mover+recorder (v1):
```bash
python3 collect_arm_data.py --duration 90 --rate 50 --output data/sysid_run1.csv
```

The resulting CSV has columns:
`time, waist_pos, waist_vel, waist_effort, shoulder_pos, ...` (one row per sample).

### 2. Run system identification

Pick one of three sysid scripts depending on your hardware:

```bash
# Option A — subsample (fast, uses every 10th point after filtering)
python3 sysid_subsample.py data/sysid_run1.csv --stride 10

# Option B — vectorized Newton-Euler (fast, uses all points)
python3 sysid_fast.py data/sysid_run1.csv

# Reference — paper-exact but slow on large datasets
python3 sysid_paper.py data/sysid_run1.csv
```

Both save the identified parameter vector to `npy/phi_<variant>_<timestamp>.npy`.

### 3. Convert to URDF

Patch the original URDF with the identified inertial parameters:
```bash
python3 phi_to_urdf.py npy/phi_fast_<timestamp>.npy \
    --template urdf/vx300s.urdf \
    -o urdf/viper300_sysid.urdf
```

Or generate a minimal standalone URDF (no mesh files):
```bash
python3 phi_to_urdf.py npy/phi_fast_<timestamp>.npy -o urdf/viper300_sysid.urdf
```

### 4. Simulate / visualize

```bash
# Interactive DH visualizer — no hardware needed
python3 sim/sim.py

# Pinocchio forward dynamics with identified model
python3 sim/pinocchio_sim_clean.py
```

---

## System identification scripts

All three scripts implement the same pipeline:

1. Load CSV → zero-phase Butterworth LPF (10 Hz cutoff)
2. Finite-difference velocity and acceleration estimation
3. Build stacked regressor `Φ` via Newton-Euler inverse dynamics
4. Per-joint torque normalization
5. QR-based base parameter reduction
6. Constrained SLSQP optimization enforcing physical feasibility
7. REL error metric + torque prediction plots

| Script | Speed strategy | DH d_6 bug fixed | Default stride | Best for |
|--------|---------------|------------------|----------------|----------|
| `sysid_paper.py` | None (78 NE calls/sample) | No | 1 | Reference / small datasets |
| `sysid_subsample.py` | Skip every N-th sample | Yes | 10 | Quick runs, RAM-limited machines |
| `sysid_fast.py` | Vectorized NE (1 forward pass/sample) | Yes | 1 | Full dataset, best accuracy |

### Physical feasibility constraints enforced

- Mass `m_i > 0` for each link
- Pseudo-inertia matrix `Σ_i ⪰ 0` (encodes positive-definite inertia and triangle inequalities)
- Viscous friction `F_v_i ≥ 0` and Coulomb friction `F_c_i ≥ 0`

### Parameter vector layout (78 floats, 13 per link)

```
φ = [m, m·cx, m·cy, m·cz, Jxx, Jxy, Jxz, Jyy, Jyz, Jzz, Fv, Fc, F0]  × 6 links
```

---

## DH parameters (Table 1 from the paper)

| Joint | α_{i-1} | a_{i-1} | d_i | θ_i offset |
|-------|---------|---------|-----|------------|
| 1 | 0 | 0 | L1 = 0.12675 m | 0 |
| 2 | 3π/2 | 0 | 0 | −0.437π |
| 3 | 0 | L2 = 0.30594 m | 0 | −0.063π |
| 4 | 3π/2 | 0 | L3+L4 = 0.30002 m | 0 |
| 5 | π/2 | 0 | 0 | 0 |
| 6 | 3π/2 | 0 | 0 | 0 |

---

## Joint limits

| Joint | Min (rad) | Max (rad) |
|-------|-----------|-----------|
| waist | −π | π |
| shoulder | −0.56π | 0.56π |
| elbow | −0.56π | 0.51π |
| forearm_roll | −π | π |
| wrist_angle | −0.59π | 0.72π |
| wrist_rotate | −π | π |
