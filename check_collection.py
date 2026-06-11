#!/usr/bin/env python3
"""
Post-collection sanity check — docs/COLLECTION_200HZ.md, Step 5, as a gate.

Verifies a recorded CSV is fit for identification BEFORE any time is spent on
it: sampling rate, timing jitter/gaps, communication-dropout sentinel rows,
missing cells, dead effort columns, duration, and per-joint motion coverage.
Pure pandas/numpy — needs no ROS, runs on any machine.

Usage:
    python3 check_collection.py data/traj_run_200hz_<stamp>.csv \
        [--min-rate 150] [--expect-duration 900]

Exit codes: 0 = PASS (usable; warnings, if any, are printed) · 1 = FAIL.
"""
import argparse
import sys

import numpy as np
import pandas as pd

ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll',
              'wrist_angle', 'wrist_rotate']
DROPOUT_TOL = 1e-3      # all arm joints within this of −π ⇒ sync-read dropout
MIN_EFFORT_MA = 5.0     # a live, moving joint must exceed this at least once

_results: list[tuple[str, str]] = []   # (level, message)


def _add(level: str, msg: str) -> None:
    _results.append((level, msg))
    print(f'[{level:^4}] {msg}')


def main() -> None:
    parser = argparse.ArgumentParser(
        description='Sanity-check a collected joint-state CSV (PASS/FAIL gate)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('csv', help='Recorded CSV to check')
    parser.add_argument('--min-rate', type=float, default=150.0,
                        help='FAIL if median sampling rate is below this [Hz]')
    parser.add_argument('--expect-duration', type=float, default=None,
                        help='WARN if the recording spans < 90 %% of this [s]')
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    print(f'── check_collection: {args.csv} ──')

    # ── Required columns ──────────────────────────────────────────────────────
    needed = ['time'] + [f'{j}_{f}' for j in ARM_JOINTS
                         for f in ('pos', 'vel', 'effort')]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        _add('FAIL', f'missing columns: {missing}')
        print('VERDICT: FAIL — not a usable identification CSV')
        sys.exit(1)

    n = len(df)
    t = df['time'].values.astype(float)
    if n < 2 or np.isnan(t).any():
        _add('FAIL', f'{n} rows / NaNs in time column')
        print('VERDICT: FAIL — not a usable identification CSV')
        sys.exit(1)

    # ── Timing ────────────────────────────────────────────────────────────────
    dt = np.diff(t)
    span = t[-1] - t[0]
    med_dt = float(np.median(dt))
    rate = 1.0 / med_dt if med_dt > 0 else 0.0

    _add('OK' if rate >= args.min_rate else 'FAIL',
         f'{n} rows, span {span:.1f} s, median rate {rate:.1f} Hz '
         f'(required ≥ {args.min_rate:.0f})')

    backwards = int((dt < 0).sum())
    if backwards:
        lvl = 'FAIL' if backwards > 0.001 * n else 'WARN'
        _add(lvl, f'{backwards} non-monotonic time steps')

    p95 = float(np.percentile(dt, 95))
    dmax = float(dt.max())
    lvl = 'OK'
    if dmax > 5 * med_dt:
        lvl = 'WARN'
    _add(lvl, f'dt median {med_dt*1e3:.2f} ms, p95 {p95*1e3:.2f} ms, '
              f'max {dmax*1e3:.1f} ms')
    if lvl == 'WARN':
        worst = np.argsort(dt)[-3:][::-1]
        gaps = ', '.join(f'{dt[i]*1e3:.0f} ms at t={t[i]:.1f} s' for i in worst)
        print(f'       largest gaps: {gaps}')

    if args.expect_duration is not None and span < 0.9 * args.expect_duration:
        _add('WARN', f'span {span:.1f} s < 90 % of expected '
                     f'{args.expect_duration:.0f} s')

    # ── Communication dropouts (all arm joints ≈ −π) ──────────────────────────
    q = df[[f'{j}_pos' for j in ARM_JOINTS]].values.astype(float)
    drop_mask = np.all(np.abs(q + np.pi) < DROPOUT_TOL, axis=1)
    n_drop = int(drop_mask.sum())
    if n_drop == 0:
        _add('OK', 'dropout-sentinel rows (all arm joints ≈ −π): 0')
    else:
        frac = n_drop / n
        _add('FAIL' if frac > 0.005 else 'WARN',
             f'{n_drop} dropout-sentinel rows ({100*frac:.3f} %) — '
             f'identify with --drop-glitches')

    # ── Missing cells in arm columns ──────────────────────────────────────────
    arm_cols = [f'{j}_{f}' for j in ARM_JOINTS for f in ('pos', 'vel', 'effort')]
    n_nan = int(df[arm_cols].isna().sum().sum())
    if n_nan == 0:
        _add('OK', 'no empty cells in arm columns')
    else:
        frac = n_nan / (n * len(arm_cols))
        _add('FAIL' if frac > 0.005 else 'WARN',
             f'{n_nan} empty cells in arm columns ({100*frac:.3f} %)')

    # ── Effort columns alive (identification needs torque) ───────────────────
    dead = []
    for j in ARM_JOINTS:
        eff = df[f'{j}_effort'].values.astype(float)
        if np.nanmax(np.abs(eff)) < MIN_EFFORT_MA:
            dead.append(j)
    if dead:
        _add('FAIL', f'effort column dead (<{MIN_EFFORT_MA:.0f} mA peak): {dead}')
    else:
        peaks = ', '.join(
            f'{j} {np.nanmax(np.abs(df[f"{j}_effort"].values.astype(float))):.0f}'
            for j in ARM_JOINTS)
        _add('OK', f'effort peaks [mA]: {peaks}')

    # ── Motion coverage (informational) ───────────────────────────────────────
    print('       position range per joint [rad]:')
    for j in ARM_JOINTS:
        pos = df[f'{j}_pos'].values.astype(float)
        good = pos[~drop_mask] if n_drop else pos
        print(f'         {j:<13} {np.nanmin(good):+7.3f} … {np.nanmax(good):+7.3f}')

    # ── Header-stamp vs arrival jitter (only if recv_time present) ───────────
    if 'recv_time' in df.columns:
        rt = df['recv_time'].values.astype(float)
        adt = np.diff(rt)
        print(f'       arrival-dt (recv_time): median {np.median(adt)*1e3:.2f} ms, '
              f'p95 {np.percentile(adt, 95)*1e3:.2f} ms, '
              f'max {adt.max()*1e3:.1f} ms')

    # ── Verdict ───────────────────────────────────────────────────────────────
    fails = [m for lvl, m in _results if lvl == 'FAIL']
    warns = [m for lvl, m in _results if lvl == 'WARN']
    if fails:
        print(f'VERDICT: FAIL ({len(fails)} failure(s)) — do not identify on this run')
        sys.exit(1)
    print(f'VERDICT: PASS — usable for identification'
          + (f' ({len(warns)} warning(s) above)' if warns else ''))
    sys.exit(0)


if __name__ == '__main__':
    main()
