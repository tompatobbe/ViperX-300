# Lab runbook — 200 Hz data collection

**Status:** procedure, not yet executed (written 2026-06-11, off-site). **Goal:**
collect a new identification dataset with `/vx300s/joint_states` publishing and
recording at ~**200 Hz** (the paper's rate), instead of the ~47 Hz of
`traj_run_20260518_143818.csv`.

**Why this matters** (see CHANGELOG 2026-06-11 and THESIS_NOTES "Encoder velocity vs
differentiated position"): the ~47 Hz rate is the shared root cause of (a) noisy `q̈`
from differentiated position and (b) the unusable, lagged/attenuated velocity
register. A clean 200 Hz capture is the prerequisite for identifying the per-link
inertias (currently a regulariser blob) and pushing REL from 0.59 toward the paper's
0.43. Nothing downstream changes the model unless the *measurement* rate improves.

---

## ⭐ One-command procedure (added 2026-06-11 — use this)

The gates and steps below are now automated by **`collect_200hz.sh`** (new files
only: `record_joint_states_200hz.py`, `check_topic_rate.py`, `check_collection.py`;
nothing existing was modified). With the driver running in its own terminal
(`ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s`):

```bash
bash collect_200hz.sh --smoke     # 60 s end-to-end rehearsal through ALL gates
bash collect_200hz.sh             # the real 900 s run
```

The script refuses to move the arm unless every gate passes, in this order:
env check → FTDI latency_timer → 1 ms (sudo only if needed) → **topic-rate gate**
(≥ 150 Hz, else abort) → recorder started & confirmed receiving → excitation
trajectory (unmodified `run_trajectories.py`, **seed 42** — the delivered-model
trajectory) → recorder stopped → `check_collection.py` PASS/FAIL verdict, then it
prints the identify/validate commands. Output: `data/traj_run_200hz_<stamp>.csv`,
logs in `data/logs/`. The new recorder records **every message** with the
**header-stamp time base** (Tier 2 below, now done), which removes the Step-4
throttle caveat entirely.

Manual Steps 1–6 below remain as the reference / fallback procedure and for
debugging a failed gate.

---

## Root cause to fix

The recorder (`record_joint_states.py`) is a **passive subscriber** — it cannot
record faster than `/vx300s/joint_states` publishes. The previous run requested
200 Hz but the topic only published ~47 Hz. **47 Hz is the classic signature of the
FTDI `latency_timer` defaulting to 16 ms** on the U2D2 (caps serial round-trips to
~62 Hz; after sync-read overhead for 8 motors → ~47 Hz). So the fix is upstream of
this repo: the serial bus + driver, verified before any long run.

---

## Step 1 — FTDI latency timer → 1 ms  (the main lever)

1. Identify the U2D2 serial device. Interbotix usually symlinks it to `/dev/ttyDXL`:
   ```bash
   ls -l /dev/ttyDXL /dev/ttyUSB*      # find the real ttyUSBn behind ttyDXL
   ```
2. Check the current value (expect `16`):
   ```bash
   cat /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
   ```
3. **Runtime fix** (does not survive replug/reboot):
   ```bash
   echo 1 | sudo tee /sys/bus/usb-serial/devices/ttyUSB0/latency_timer
   cat       /sys/bus/usb-serial/devices/ttyUSB0/latency_timer   # confirm → 1
   ```
4. **Persistent fix (udev).** First check whether interbotix's own rule already sets
   it — if so, you may already be at 1 ms and the bottleneck is elsewhere (go to
   Step 2):
   ```bash
   grep -i latency /etc/udev/rules.d/*interbotix* /etc/udev/rules.d/*.rules 2>/dev/null
   ```
   If absent, add a rule (e.g. `/etc/udev/rules.d/99-ftdi-latency.rules`):
   ```
   SUBSYSTEM=="usb-serial", DRIVER=="ftdi_sio", ATTR{latency_timer}="1"
   ```
   then reload and re-plug the U2D2:
   ```bash
   sudo udevadm control --reload-rules && sudo udevadm trigger
   ```

## Step 2 — Driver publish rate / baud (headroom)

- The `interbotix_xs_sdk` publishes joint states from its serial read loop; with a
  1 ms latency timer that loop is serial-limited, not artificially capped. Still,
  inspect the xsarm control launch / `motor_configs` for any explicit publish-rate
  or read-rate parameter and confirm it is not set below 200.
- The default motor-bus baud is 1 Mbps. The XM540-W270 supports much higher; raising
  the baud (motor register + driver config) gives headroom if 200 Hz is marginal.
  **Only touch this if Step 1 alone doesn't reach 200 Hz** — verify first.

## Step 3 — Verification GATE (do not skip)

Launch the driver, then **measure the actual topic rate before collecting anything**:
```bash
ros2 topic hz /vx300s/joint_states
```
- Expect **~200 Hz** (or whatever the bus sustains; ≫47 is the point).
- If still ~50 Hz → the latency timer change didn't apply. Re-check
  `cat .../latency_timer` and that you targeted the *correct* ttyUSBn.
- A short test collection (e.g. `--duration 30`) followed by the dt check below is a
  cheap second gate.

## Step 4 — Recorder settings  ⚠ throttle caveat
> **Superseded 2026-06-11:** this caveat applies to the old `record_joint_states.py`
> only. The one-command flow uses `record_joint_states_200hz.py`, which has **no
> throttle** (records every message) — nothing below is needed unless you fall back
> to the old recorder.

`record_joint_states.py` throttles by comparing each message to the **last written
row's** time (`min_dt = 1/rate`). If you set the recorder `--rate` *equal* to the
publish rate, jitter makes messages land just inside `min_dt` and get dropped — and
because the comparison is against the last *written* row, you can end up writing only
every other message (≈ halving the rate to ~100 Hz).

**Therefore set the recorder rate well ABOVE the publish rate** (e.g. `--rate 300`
for a 200 Hz topic, `min_dt ≈ 3.3 ms < 5 ms` spacing → nothing dropped).

> Note: `run_trajectories.sh` passes a single `--rate` to **both** the recorder and
> the trajectory generator, so you cannot set them independently there. For this run,
> prefer launching them in **two terminals** (as `record_joint_states.py`'s docstring
> intends) so the recorder rate can be high while the trajectory stays at 200 Hz:
>
> ```bash
> # Terminal 1 — recorder (rate ABOVE publish so nothing is throttled)
> python3 record_joint_states.py --duration 900 --rate 300 --output data/traj_run_200hz.csv
>
> # Terminal 2 — excitation trajectory (after the recorder prints "recording started")
> python3 run_trajectories.py --duration 900 --rate 200 --seed 42 --stride 4
> ```
> (A future recorder revision — Tier 2, not yet done — would record every message and
> use the message header timestamp, removing this caveat entirely.)

## Step 5 — Post-collection sanity checks (before trusting the run)

> **Automated 2026-06-11:** `python3 check_collection.py data/<run>.csv
> --min-rate 150 --expect-duration 900` performs all of this (plus effort-column,
> empty-cell, and coverage checks) with a PASS/FAIL exit code; `collect_200hz.sh`
> runs it for you. The snippet below remains for manual inspection.

On the new CSV, confirm the rate and timing are actually clean:
```bash
python3 - <<'PY'
import numpy as np, pandas as pd
df = pd.read_csv('data/traj_run_200hz.csv')   # adjust name
t = df['time'].values; dt = np.diff(t)
print(f"samples={len(df)} dur={t[-1]-t[0]:.1f}s rate~{1/np.median(dt):.1f}Hz "
      f"dt p95={np.percentile(dt,95)*1e3:.2f}ms max={dt.max()*1e3:.1f}ms")
# dropout-row check (the all-joints=-pi sentinel)
q = df[[f'{j}_pos' for j in
        ['waist','shoulder','elbow','forearm_roll','wrist_angle','wrist_rotate']]].values
print("dropout rows (all joints ~ -pi):", int(np.all(np.abs(q+np.pi)<1e-3,axis=1).sum()))
PY
```
Targets: rate ≈ 200 Hz, `dt` jitter small and bounded, dropout rows ≈ 0.

## Step 6 — Identify & validate on the new run

Use the established recipe (no preprocessing flags needed — at 200 Hz, differentiated
position is clean and the encoder velocity is no longer the issue):
```bash
python3 sysid_feasible.py data/traj_run_200hz.csv --no-plot --stride 1 \
  --method cvxpy --entropic 0.05 --w2 100 --solver CLARABEL
python3 phi_to_urdf.py outputs/npy/<the npy it wrote>.npy
source /opt/ros/humble/setup.bash
python3 compare_urdf_performance.py --friction --urdf-b outputs/urdf/<the urdf>.urdf
```
Compare the unconstrained vs feasible REL and the per-link inertias against the
delivered 47 Hz model — the question is whether 200 Hz finally lets the inertias lift
off the regulariser blob and narrows the unconstrained↔feasible gap.

---

## Open items deferred (not part of this runbook)

- ~~**Tier 2 — recorder fidelity:** use `msg.header.stamp` for the time column,
  `qos_profile_sensor_data` (BEST_EFFORT, to match the publisher and avoid silent
  drops), and record every message (drop the throttle). Directly improves `q̈`
  quality and removes the Step-4 caveat.~~ **Done 2026-06-11** —
  `record_joint_states_200hz.py` (see THESIS_NOTES "Recorder instrumentation").
- **Tier 3 — excitation:** make the trajectory optimizer minimise `cond(Φ_b)` of the
  actual base regressor (currently it minimises the condition number of a `[vel;accel]`
  matrix), and widen waist/forearm_roll coverage (~60 % of range last time). This is
  the lever that would *identify* the per-link inertias rather than fill them from the
  blob.
