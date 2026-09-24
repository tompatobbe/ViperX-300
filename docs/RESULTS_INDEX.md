# RESULTS INDEX — which artifact backs which thesis claim

Stable map from thesis claims to the exact files that support them. Numbers are
deliberately NOT duplicated here — follow the pointer to the named CHANGELOG /
THESIS_NOTES entry so there is one source of truth. Update this file whenever a
champion changes or a new benchmark result lands.

## The deliverable (validated champion, 2026-06-24)

| What | File |
|---|---|
| URDF (convenience copy, provenance header) | `urdf/champion.urdf` |
| URDF (canonical artifact) | `outputs/urdf/traj_run_200hz_20260624_124955__sysid_feasible-v1-5__cfg-27904c2e__phi_to_urdf-v1-1__cfg-451881cf.urdf` |
| phi vector (+ `.json` sidecar with config/recipe) | `outputs/npy/traj_run_200hz_20260624_124955__sysid_feasible-v1-5__cfg-27904c2e.npy` |

Recipe: `champion.urdf` header and the phi `.json` sidecar; numbers in CHANGELOG
2026-06-24 "Validated identified model".

**Previous champion (2026-06-13, `cfg-a92e984c`)** — identified on
`traj_run_200hz_20260612_131613.csv`. Superseded as the deliverable but *not*
invalid: it is the model behind the gravity-benchmark claim below. Recipe &
reproduce commands: `docs/HANDOVER.md` §CURRENT STATE.

## Datasets

| Dataset | Role |
|---|---|
| `data/traj_run_200hz_20260624_124955.csv` | 200 Hz, operating-point tour (900 s) — the **current** champion was identified on this |
| `data/traj_run_200hz_20260623_145333.csv` | 200 Hz — held-out dynamics validation of the current champion |
| `data/traj_run_200hz_20260612_131613.csv` | 200 Hz primary — the 2026-06-13 champion (`cfg-a92e984c`) was identified on this |
| `data/traj_run_200hz_20260612_161025.csv` | 200 Hz replicate (same seed-42 traj) — held-out validation of the 2026-06-13 champion |
| `data/traj_run_20260518_143818.csv` | May 47 Hz — independent held-out set. ⚠ noisy q̈ at 47 Hz makes inertia-sensitive metrics misleading (see CHANGELOG "Champion validated on full dynamics") |
| `data/static_gravity_20260613_183554.csv` | Static gravity benchmark (holding currents, raw mA) |
| `data/logs/` | γ / γ-Ia sweep logs (model-selection evidence) |

## Claims → evidence

| Thesis claim | Script that produced it | Numbers & discussion |
|---|---|---|
| Champion beats factory CAD by 55 % on held-out rigid-body torque and matches the paper's validation REL (0.426 vs 0.392) | `compare_urdf_performance.py` | CHANGELOG 2026-06-24 "Validated identified model" |
| w₂ is a model-fidelity knob (default under-couples; w₂=10 sweet spot, w₂=100 overfits masses) | `sysid_feasible.py --w2` sweep | THESIS_NOTES "The coupling weight w₂ sets *model fidelity*" |
| Workspace coverage (operating-point tour) fixes the conditioning of the excitation | coverage report | CHANGELOG 2026-06-24 "Workspace coverage"; THESIS_NOTES "Workspace coverage vs. conditioning" |
| 2026-06-13 champion reproduces the paper's published gravity (shoulder 95 %, elbow 90 %), beats factory on every gravity joint | `compare_gravity.py` | CHANGELOG "Gravity deficit CLOSED on v1.5"; THESIS_NOTES "Resolution (2026-06-13, later)"; figure `figures/gravity_compare_static_gravity_20260613_183554.png` |
| 2026-06-13 champion beats the no-model baseline and factory URDF on held-out torque | `compare_urdf_performance.py` | CHANGELOG "Champion validated on full dynamics" |
| Kinematics bug root-cause (standard vs modified DH) and its fix | `tools/test_fk_equivalence.py`, `tools/test_phi_urdf_consistency.py` | CHANGELOG 2026-06-13 "ROOT CAUSE FOUND & FIXED" |
| Effort units are Dynamixel mA; τ = (mA/1000)·2.409·motors | — | CHANGELOG 2026-06-10; CLAUDE.md "Conventions" |
| Reflected motor inertia Ia is identifiable (~0.8 kg·m² shoulder) but doesn't dethrone (pre-fix analysis) | `sweep_gamma_ia.sh` | CHANGELOG "γ retune"; THESIS_NOTES "Reflected motor inertia" |
| Encoder velocity register is lagged/attenuated — differentiated position is the right input | — | CHANGELOG 2026-06-11; THESIS_NOTES "Encoder velocity vs differentiated position" |

## Baselines the champion is compared against

- **Factory URDF:** `urdf/vx300s.urdf` (validation baseline / kinematics only — never an inertial prior).
- **Paper's published model:** included by default in `compare_gravity.py` (`external/paper_model/`; disable with `--no-paper`); paper benchmark numbers in `docs/PAPER_SUMMARY.md`.
- **No-model baseline:** computed inside `compare_urdf_performance.py` (predict-zero-torque margin).

## Superseded (historical narrative only — do NOT cite as results)

- **May model `cfg-640cb8ef`** and everything identified before 2026-06-13: produced through the buggy standard-DH chain, invalid. Kept in `outputs/` + `archive/urdf/` for provenance; story in HANDOVER §Historical record.
- Pre-artifact-system URDFs: `archive/urdf/`.

## Open items that qualify the claims (for the Discussion chapter)

- ≈0.63 static-amplitude scale anomaly → standstill-stiction hypothesis, pending lab run (`collect_stiction_hysteresis.sh`); THESIS_NOTES "Standstill stiction".
- Current champion trails the paper on waist (REL 0.55) and wrist_angle (0.54) — friction-model candidates; CHANGELOG 2026-06-24 "Validated identified model" §Impact.
- Gravity benchmark (`compare_gravity.py`) not yet re-run on the 2026-06-24 champion.
- wrist_angle under-excited; forearm_roll swing is not gravity (excluded from means) — HANDOVER "Other open items".
