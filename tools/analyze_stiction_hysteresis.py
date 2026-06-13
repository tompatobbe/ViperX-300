#!/usr/bin/env python3
"""analyze_stiction_hysteresis.py — measure the standstill stiction band.

Companion to control/stiction_hysteresis_poses.py. Each target pose was held
twice — approached "ascending" (final motion raises the joint) and "descending"
(final motion lowers it). Static friction settles at opposite ends of its band
in the two cases, so per gravity-bearing joint:

  band   = I_hold(descending) − I_hold(ascending)   ≈ 2·tau_breakaway  [raw mA]
  midpt  = ½·(I_descending + I_ascending)            cancels stiction → ≈ true gravity

The hypothesis (THESIS_NOTES "Standstill stiction") predicts a non-trivial band
AND that the midpoint lands on the model/paper gravity, while a single-direction
hold (the original static benchmark) sits one half-band below it. Optionally pass
--phi / --urdf / paper to overlay model gravity and check midpt ≈ model.

Everything is in raw master-motor mA (the calibration-free axis); Nm shown via
EFFORT_SCALE. Runs without ROS if only --phi/paper are used; --urdf needs
Pinocchio.

Usage
-----
    python3 tools/analyze_stiction_hysteresis.py --csv data/stiction_hyst_<date>.csv \
        --phi outputs/npy/traj_run_200hz_20260612_131613__sysid_feasible-v1-5__cfg-a92e984c.npy
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sysid_feasible import ARM_JOINTS, N_JOINTS, EFFORT_SCALE
from compare_gravity import (
    detect_static_segments, predict_paper, predict_phi, predict_urdf, _PAPER_DIR,
)


def assign_trials(segments, sidecar_path):
    """Match detected dwells to the sidecar trials by temporal order, validated
    against the commanded target q. Returns list of (trial_dict, segment)."""
    meta = json.load(open(sidecar_path))
    trials = meta["trials"]
    targets = {t["target"]: np.array(t["q"]) for t in trials}

    # Keep only segments that sit at one of the target poses (drop home/sleep).
    tol = 0.15
    kept = []
    for s in segments:
        d = {k: np.linalg.norm(q - s["q"]) for k, q in targets.items()}
        knear = min(d, key=d.get)
        if d[knear] < tol:
            s = dict(s, near_target=knear, near_err=d[knear])
            kept.append(s)

    if len(kept) != len(trials):
        print(f"  ⚠ detected {len(kept)} pose-dwells but sidecar has {len(trials)} "
              f"trials — matching by order, check the table below")
    pairs = []
    for tr, seg in zip(trials, kept):
        if seg["near_target"] != tr["target"]:
            print(f"  ⚠ order/target mismatch: trial target {tr['target']} "
                  f"({tr['approach']}) vs detected pose {seg['near_target']}")
        pairs.append((tr, seg))
    return meta, pairs


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", required=True, help="stiction hysteresis CSV")
    p.add_argument("--sidecar", default=None,
                   help="trial sidecar (default <csv>.poses.json)")
    p.add_argument("--phi", action="append", default=[])
    p.add_argument("--urdf", action="append", default=[])
    p.add_argument("--no-paper", action="store_true")
    p.add_argument("--vel-thresh", type=float, default=0.02)
    p.add_argument("--min-dur", type=float, default=3.0)
    p.add_argument("--settle", type=float, default=2.0)
    args = p.parse_args()

    sidecar = args.sidecar or os.path.splitext(args.csv)[0] + ".poses.json"

    print("=" * 70)
    print("STANDSTILL-STICTION hysteresis  (units: master-motor mA)")
    print("=" * 70)
    df = pd.read_csv(args.csv)
    segs = detect_static_segments(df, args.vel_thresh, args.min_dur, args.settle)
    if not segs:
        sys.exit("No static dwells detected — loosen --vel-thresh / --min-dur.")
    meta, pairs = assign_trials(segs, sidecar)
    print(f"  matched {len(pairs)} dwells | delta={meta.get('delta')} rad | "
          f"swept={meta.get('swept')}")

    # Group by target: collect ascending/descending holding currents.
    targets = sorted({tr["target"] for tr, _ in pairs})
    by_target = {k: {} for k in targets}
    q_of = {}
    for tr, seg in pairs:
        by_target[tr["target"]][tr["approach"]] = seg["eff_mA"]
        q_of[tr["target"]] = np.array(tr["q"])

    # Per gravity joint, report band and midpoint per target.
    gj = [ARM_JOINTS.index(x) for x in ("shoulder", "elbow", "wrist_angle")]
    print("\n  Hysteresis band = I(descending) − I(ascending) ≈ 2·τ_breakaway [mA]")
    print("  midpt = ½(asc+desc) ≈ true gravity (stiction cancels)\n")
    bands = {j: [] for j in gj}
    midpts = {k: np.zeros(N_JOINTS) for k in targets}
    for k in targets:
        a = by_target[k].get("ascending")
        d = by_target[k].get("descending")
        if a is None or d is None:
            print(f"  target {k}: missing an approach — skipped")
            continue
        mid = 0.5 * (a + d)
        midpts[k] = mid
        print(f"  target {k}  q={np.round(q_of[k],2).tolist()}")
        print(f"    {'joint':<14}{'asc':>9}{'desc':>9}{'band':>9}{'±τ[mA]':>9}{'±τ[Nm]':>9}{'midpt':>9}")
        for j in gj:
            band = d[j] - a[j]
            tau = abs(band) / 2.0
            bands[j].append(band)
            print(f"    {ARM_JOINTS[j]:<14}{a[j]:>9.0f}{d[j]:>9.0f}{band:>9.0f}"
                  f"{tau:>9.0f}{tau*EFFORT_SCALE[j]:>9.3f}{mid[j]:>9.0f}")

    print("\n  Mean stiction half-band τ_breakaway per joint [mA / Nm]:")
    for j in gj:
        if bands[j]:
            tau = np.mean(np.abs(bands[j])) / 2.0
            print(f"    {ARM_JOINTS[j]:<14}{tau:>9.0f} mA   {tau*EFFORT_SCALE[j]:>7.3f} Nm")

    # Optional: midpoint vs model gravity (predict at each target q).
    preds = {}
    Q = np.array([q_of[k] for k in targets])
    if not args.no_paper and os.path.isdir(_PAPER_DIR):
        try:
            preds["paper"] = predict_paper(Q)
        except Exception as e:
            print(f"  [warn] paper skipped: {e}")
    for f in args.phi:
        try:
            preds[f"phi:{os.path.basename(f)}"] = predict_phi(Q, f)
        except Exception as e:
            print(f"  [warn] phi {f} skipped: {e}")
    for u in args.urdf:
        try:
            preds[f"urdf:{os.path.basename(u)}"] = predict_urdf(Q, u)
        except Exception as e:
            print(f"  [warn] urdf {u} skipped: {e}")

    if preds:
        print("\n  midpt − model gravity per target [mA]  (≈0 confirms stiction "
              "fully explains the static gap):")
        M = np.array([midpts[k] for k in targets])
        for name, P in preds.items():
            print(f"    {name}")
            for j in gj:
                resid = M[:, j] - P[:, j]
                print(f"      {ARM_JOINTS[j]:<14} mean |midpt−model| = "
                      f"{np.mean(np.abs(resid)):>7.0f} mA   "
                      f"(midpt RMS {np.sqrt(np.mean(M[:,j]**2)):.0f}, "
                      f"model RMS {np.sqrt(np.mean(P[:,j]**2)):.0f})")


if __name__ == "__main__":
    main()
