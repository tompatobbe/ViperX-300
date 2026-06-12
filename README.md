# ViperX-300 — Dynamic Model Identification & Control

Master's-thesis codebase: physically-feasible dynamic model identification of
the Trossen Robotics **ViperX-300 6-DoF** arm, following Momani & Hosseinzadeh,
*Mechatronics* 112 (2025) 103419, with model-based control (robust law + ERG)
as the next phase.

```
τ = M(q)q̈ + C(q,q̇)q̇ + G(q) + τ_friction
```

**Start here:** `docs/HANDOVER.md` (current state & next steps) ·
`CLAUDE.md` (conventions, units, working agreement) ·
`docs/CHANGELOG.md` (engineering log) · `docs/THESIS_NOTES.md` (discussion).

## The pipeline

```
1. COLLECT    collect_200hz.sh            gated 200 Hz collection (smoke + real run)
              ├─ run_trajectories.py        seed-42 excitation trajectory mover
              ├─ record_joint_states_200hz.py  recorder (every msg, header-stamp time)
              ├─ check_topic_rate.py         pre-flight: topic rate ≥ 150 Hz
              └─ check_collection.py         post-flight: CSV verdict (rate/dropouts)
                       │
                       ▼   data/<run>.csv
2. IDENTIFY   sysid_feasible.py           feasible SDP identification (no ROS needed)
                       │
                       ▼   outputs/npy/<run>__<cfg>.npy  (+ .json sidecar)
3. EXPORT     phi_to_urdf.py              phi → standalone URDF
                       │
                       ▼   outputs/urdf/<...>.urdf
4. VALIDATE   compare_urdf_performance.py RNEA torque prediction vs measured (needs ROS)
```

`pipeline_artifacts.py` caches stage outputs by config hash — results never
overwrite. `identify_200hz.sh` chains steps 2–4; `sweep_gamma.sh` runs the
γ-regularisation sweep.

## Quick start

```bash
# Identify (numpy/scipy/cvxpy only — no ROS required)
python3 sysid_feasible.py data/<run>.csv --no-plot --stride 1 \
    --method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL

# Export the URDF
python3 phi_to_urdf.py outputs/npy/<the npy it wrote>.npy

# Validate (needs real pinocchio 3.9 → source ROS first!)
source /opt/ros/humble/setup.bash
python3 compare_urdf_performance.py --friction --drop-glitches \
    --csv data/<run>.csv --urdf-b outputs/urdf/<the urdf>.urdf
```

Without ROS sourced, `import pinocchio` finds a bogus PyPI name-squat (0.4.3)
and validation fails — see "Pinocchio / ROS gotcha" in `CLAUDE.md`. Do **not**
`pip install pinocchio`.

## Layout

| Path | Contents |
|---|---|
| *(root)* | The active pipeline scripts shown above |
| `data/` | Recorded runs (CSV) + `data/logs/` (collection/sweep logs) |
| `outputs/` | Identified artifacts: `npy/` (phi + sidecars), `urdf/`, `legacy/` |
| `docs/` | HANDOVER, CHANGELOG, THESIS_NOTES, PAPER_SUMMARY, Paper.txt, runbooks |
| `urdf/` | Factory `vx300s.urdf` (validation baseline / kinematics only) |
| `control/` | Hardware control snippets (`trq.py` = torque→current) |
| `sim/` | Pinocchio simulation |
| `tests/` | pytest suite (`pipeline_artifacts` round-trips) |
| `tools/` | Working utilities off the critical path: `volt_watch.py` (supply-dip watcher), `diagnose_*`, `monitor_servos.py`, plotting/visualisation |
| `figures/` | Saved plots |
| `archive/` | Superseded code, kept for traceability: `identification/` (pre-SDP sysid variants), `collection/` (May-era 47 Hz collection flow), `npy/` (pre-artifact phi files), `scratch/` |

The delivered, cross-validated model (May data, `cfg-640cb8ef`) is listed in
`docs/HANDOVER.md` §TL;DR.

## Hardware facts (never violate silently)

- Joint order: `waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate`.
- CSV `*_effort` is Dynamixel present current in **mA**;
  `τ = (mA/1000)·2.409·n_motors`, shoulder & elbow are dual-motor (`n=2`).
- SI units everywhere; arm is fixed-base; match joints by **name**, not index.
