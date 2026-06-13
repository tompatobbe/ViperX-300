#!/usr/bin/env python3
"""Analyse the static-pose gravity experiment (collect_static_gravity.sh).

Input is a normal recorder time-series CSV (record_joint_states_200hz.py:
`time, <joint>_pos/_vel/_effort, ...`). The arm is dead-stopped while it dwells
at each pose, so we segment poses by MOTION DETECTION on the velocity columns —
no clock alignment needed. For each stationary window we average the holding
current (mA, master motor) and position, then compare measured current to G(q)
from our identified model and the paper's, reusing the verified G evaluators in
tools/compare_paper_model.py.

At standstill measured current ≈ gravity (+ small stiction), with NO
friction/inertia/Coriolis confound. Per joint we fit measured ≈ slope·G + offset:
  - OUR slope ≈ 1, offset ≈ 0, r ≈ 1  ⇒ gravity model is physically correct →
    safe to start PD + gravity-compensation control (resolves the circularity:
    we'd command G(q) and the arm actually needs that current to hold station).
  - PAPER slope = per-joint amplitude factor (the ≈0.6 anomaly); mA-vs-mA so
    independent of our k_t/EFFORT_SCALE.
  - forearm_roll PAPER slope < 0 ⇒ confirms the sign/first-moment issue.

Run (no ROS needed):
    python3 tools/analyze_static_gravity.py --csv data/static_gravity_<date>.csv \
        --phi outputs/npy/<model>.npy [--sidecar data/static_gravity_<date>.poses.json]
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
import sysid_feasible as sf                       # noqa: E402
import compare_paper_model as cpm                 # noqa: E402

JN = sf.ARM_JOINTS


def segment_poses(df, vel_thresh, min_dwell_s):
    """Return list of (q_mean(6), eff_mean(6), n_samples) for each stationary
    window. Stationary = all arm-joint |vel| < vel_thresh; a window must last
    >= min_dwell_s. We average the central 60% of each window to skip settling."""
    t = df["time"].values.astype(float)
    pos = df[[f"{j}_pos" for j in JN]].values.astype(float)
    vel = df[[f"{j}_vel" for j in JN]].values.astype(float)
    eff = df[[f"{j}_effort" for j in JN]].values.astype(float)

    still = np.all(np.abs(vel) < vel_thresh, axis=1)
    fs = 1.0 / np.median(np.diff(t))
    min_n = int(min_dwell_s * fs)

    segs = []
    i, N = 0, len(still)
    while i < N:
        if not still[i]:
            i += 1
            continue
        j = i
        while j < N and still[j]:
            j += 1
        if j - i >= min_n:
            lo = i + int(0.2 * (j - i))           # drop first/last 20% (settling)
            hi = i + int(0.8 * (j - i))
            segs.append((pos[lo:hi].mean(0), eff[lo:hi].mean(0), j - i))
        i = j
    return segs


def line_fit(x, y, min_range=10.0):
    """y ≈ slope·x + offset; undefined when G barely varies (gravity-free axis)."""
    if np.ptp(x) < min_range:
        return np.nan, np.nan, np.nan
    slope, offset = np.polyfit(x, y, 1)
    return slope, offset, np.corrcoef(x, y)[0, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True, help="recorder time-series CSV")
    ap.add_argument("--phi", required=True, help="our identified φ (.npy)")
    ap.add_argument("--paper-dir", default=os.path.join(REPO, "external", "paper_model"))
    ap.add_argument("--sidecar", default=None, help="poses JSON (count validation)")
    ap.add_argument("--vel-thresh", type=float, default=0.02,
                    help="rad/s; below this on all joints = stationary")
    ap.add_argument("--min-dwell", type=float, default=2.0, help="s; min stationary window")
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    segs = segment_poses(df, args.vel_thresh, args.min_dwell)
    q = np.array([s[0] for s in segs])            # (P,6) rad
    meas = np.array([s[1] for s in segs])         # (P,6) mA

    n_expect = None
    if args.sidecar and os.path.exists(args.sidecar):
        n_expect = len(json.load(open(args.sidecar))["poses"])

    print("=" * 72)
    print("Static-pose gravity check — measured holding current vs G(q)  [mA]")
    print("=" * 72)
    print(f"  csv  : {args.csv}")
    print(f"  phi  : {args.phi}")
    msg = f"  found {len(segs)} stationary windows"
    if n_expect is not None:
        msg += f" (sidecar expects {n_expect}" + (" ✓)" if len(segs) == n_expect
                                                   else " — MISMATCH, check --vel-thresh/--min-dwell)")
    print(msg)
    if len(segs) < 3:
        print("  Too few windows — adjust --vel-thresh / --min-dwell.")
        return

    calc, *_ = cpm.load_paper(args.paper_dir)
    phi = np.load(args.phi)
    paper_G = cpm.paper_gravity_mA(calc, q)                  # (P,6) mA
    our_G = cpm.our_gravity_Nm(phi, q) / sf.EFFORT_SCALE     # Nm→mA (master)

    print("\n  Per-pose measured holding current [mA] (q = mean stationary position):")
    print(f"  {'#':>2} " + "".join(f"{j[:5]:>8}" for j in JN))
    for k in range(len(segs)):
        print(f"  {k:>2} " + "".join(f"{meas[k,i]:8.0f}" for i in range(6)))

    print("\n  Per joint:  measured ≈ slope·G + offset")
    print(f"  {'joint':<14}|{'OUR  slope':>11}{'off':>7}{'r':>7}"
          f"  |{'PAPER slope':>12}{'off':>7}{'r':>7}")
    print("  " + "-" * 68)
    for j, nm in enumerate(JN):
        so, oo, ro = line_fit(our_G[:, j], meas[:, j])
        sp, op, rp = line_fit(paper_G[:, j], meas[:, j])
        print(f"  {nm:<14}|{so:11.3f}{oo:7.0f}{ro:7.3f}"
              f"  |{sp:12.3f}{op:7.0f}{rp:7.3f}")

    res = meas - our_G
    print("\n  OUR-model gravity residual |measured − G| [mA], per joint:")
    print(f"  {'joint':<14}{'mean':>8}{'max':>8}{'RMS':>8}   (vs measured RMS)")
    for j, nm in enumerate(JN):
        a = np.abs(res[:, j])
        print(f"  {nm:<14}{a.mean():8.1f}{a.max():8.1f}"
              f"{np.sqrt((res[:,j]**2).mean()):8.1f}   ({np.sqrt((meas[:,j]**2).mean()):.0f})")

    print("\n  Readout:")
    print("   • OUR slope≈1, offset≈0, r≈1 on shoulder/elbow ⇒ gravity model correct")
    print("     → safe to start PD + gravity-compensation control.")
    print("   • PAPER slope = per-joint amplitude factor (the ≈0.6 anomaly), mA-vs-mA.")
    print("   • forearm_roll PAPER slope < 0 ⇒ confirms the sign/first-moment issue.")


if __name__ == "__main__":
    main()
