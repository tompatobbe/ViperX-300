#!/usr/bin/env python3
"""Workspace-coverage diagnostic for an excitation design or a collected run.

Motivation (2026-06-24): the cond(Φ_b) optimiser conditions the *identification*
regressor, but it is BLIND to coverage of dynamically-degenerate joints (above
all the waist — the dynamics are invariant to its angle, so the optimiser has no
incentive to sweep it) and a single Fourier curve only thinly samples the rest of
the workspace. Re-running a 900 s collection just to discover the model never saw
the edges is the time sink this script removes: it reports, OFFLINE, exactly which
joint angles a design (or a recorded CSV) actually visits, so a design is vetted
*before* any hardware time is spent.

Inputs:
  - a design .npz (a, b, q0) saved by run_trajectories.py --save, or
  - a collected CSV with <joint>_pos columns (data/*.csv).

Reports per joint: visited [min,max], range, and % of the operating-limit range
covered; plus a coupled shoulder–elbow occupancy grid (the joint pair whose
first-moment lumping started this whole effort). With --plot, draws per-joint
angle histograms and the shoulder–elbow scatter.

    python3 tools/coverage_report.py outputs/excitation_design.npz
    python3 tools/coverage_report.py data/traj_run_200hz_20260623_145333.csv --plot
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import run_trajectories as rt  # noqa: E402  (pure numpy; no ROS)

JOINTS = rt.JOINT_NAMES
LO, HI = rt.LIMITS_LO, rt.LIMITS_HI


def angles_from_npz(path, duration, rate, tour=False, tour_scale=rt.TOUR_MS_SCALE):
    """Evaluate the design's joint trajectory over `duration` s at `rate` Hz.
    With tour=True, reproduces run_trajectories.py --tour (scaled multisine around
    0 + slow operating-point tour) so coverage can be vetted before collecting."""
    d = np.load(path)
    a, b, q0 = d['a'], d['b'], d['q0']
    t = np.arange(int(duration * rate)) / rate
    if tour:
        q = rt.build_tour_waypoints(t, a, b, q0, ms_scale=tour_scale)
    else:
        q = rt.traj_pos(t, a, b, q0)
    np.clip(q, LO, HI, out=q)   # the executor clips to limits before commanding
    return q


def angles_from_csv(path):
    """Read the 6 arm-joint position columns from a recorded CSV."""
    import pandas as pd
    df = pd.read_csv(path)
    cols = [f'{j}_pos' for j in JOINTS]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        sys.exit(f'CSV missing columns: {missing}')
    q = df[cols].to_numpy()
    # Drop the all-joints≈−π park/dropout rows so they don't distort the range.
    keep = ~np.all(np.isclose(q, -np.pi, atol=0.05), axis=1)
    return q[keep]


def per_joint_report(q):
    print(f'\nSamples: {len(q)}')
    print(f'{"Joint":<14}{"visited [min, max]":>22}{"range":>9}{"limit range":>16}'
          f'{"coverage":>10}')
    cov = []
    for j, name in enumerate(JOINTS):
        lo, hi = q[:, j].min(), q[:, j].max()
        rng = hi - lo
        limrng = HI[j] - LO[j]
        pct = 100.0 * rng / limrng if limrng > 0 else 0.0
        cov.append(pct)
        flag = '  <-- thin' if pct < 40 else ''
        print(f'{name:<14}[{lo:>+6.2f},{hi:>+6.2f}]{"":>6}{rng:>8.2f}'
              f'  [{LO[j]:+.2f},{HI[j]:+.2f}]{pct:>8.0f}%{flag}')
    print(f'\nMean per-joint range coverage: {np.mean(cov):.0f}%  '
          f'(min {np.min(cov):.0f}% on {JOINTS[int(np.argmin(cov))]})')
    return np.array(cov)


def shoulder_elbow_occupancy(q, nbins=12):
    """2-D occupancy of the shoulder×elbow plane. Reported against the REACHABLE
    cells only: shoulder & elbow are physically coupled (self-collision), so the
    reachable region is the diagonal collision band, not the full rectangle.
    Counting only in-band cells gives the true coverage of what the arm can do."""
    sh, el = q[:, 1], q[:, 2]
    edges_s = np.linspace(LO[1], HI[1], nbins + 1)
    edges_e = np.linspace(LO[2], HI[2], nbins + 1)
    cs = 0.5 * (edges_s[:-1] + edges_s[1:])
    ce = 0.5 * (edges_e[:-1] + edges_e[1:])
    SHC, ELC = np.meshgrid(cs, ce, indexing='ij')
    in_band = ((ELC <= rt.SH_EL_BAND_HI[0] * SHC + rt.SH_EL_BAND_HI[1]) &
               (ELC >= rt.SH_EL_BAND_LO[0] * SHC + rt.SH_EL_BAND_LO[1]))
    H, _, _ = np.histogram2d(sh, el, bins=[edges_s, edges_e])
    visited = (H > 0)
    reach = int(in_band.sum())
    occ = 100.0 * int((visited & in_band).sum()) / max(reach, 1)
    print(f'\nShoulder×elbow occupancy: {occ:.0f}% of the {reach} REACHABLE '
          f'(in-band) cells of the {nbins}×{nbins} grid visited')
    return occ


def plot(q):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 4, figsize=(16, 7))
    for j, name in enumerate(JOINTS):
        ax = axes.flat[j]
        ax.hist(q[:, j], bins=40, color='steelblue')
        ax.axvline(LO[j], color='r', ls='--', lw=1)
        ax.axvline(HI[j], color='r', ls='--', lw=1)
        ax.set_title(name)
    axes.flat[6].scatter(q[:, 1], q[:, 2], s=2, alpha=0.3)
    axes.flat[6].set(xlabel='shoulder', ylabel='elbow', title='shoulder × elbow')
    axes.flat[7].axis('off')
    fig.tight_layout()
    plt.show()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='design .npz or recorded .csv')
    ap.add_argument('--duration', type=float, default=900.0,
                    help='(npz only) seconds of trajectory to evaluate')
    ap.add_argument('--rate', type=float, default=50.0,
                    help='(npz only) sample rate for the evaluation grid')
    ap.add_argument('--plot', action='store_true', help='show histograms + scatter')
    ap.add_argument('--tour', action='store_true',
                    help='(npz only) apply the operating-point tour (run_trajectories --tour)')
    args = ap.parse_args()

    if args.source.endswith('.npz'):
        q = angles_from_npz(args.source, args.duration, args.rate, tour=args.tour)
        print(f'Design: {args.source}  ({args.duration:.0f} s @ {args.rate:.0f} Hz eval'
              f'{", TOUR" if args.tour else ""})')
    else:
        q = angles_from_csv(args.source)
        print(f'Recorded run: {args.source}')

    per_joint_report(q)
    shoulder_elbow_occupancy(q)
    if args.plot:
        plot(q)


if __name__ == '__main__':
    main()
