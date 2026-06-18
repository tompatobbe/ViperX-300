#!/usr/bin/env python3
"""
compare_gravity.py — Gravity-only model comparison on a static-pose dataset
===========================================================================

Why this script exists
----------------------
The open problem (CHANGELOG 2026-06-13) is that our identified models carry
almost no gravity: the SDP+entropic solution zeroed the first moments, so G(q)
is nearly constant over configuration while the *measured* holding currents are
strongly pose-dependent. The friction-fitted torque-prediction validation hid
this, because a per-joint offset/friction basis fitted on the test data absorbs
a flat model's error.

This script isolates **gravity alone**, which is exactly the broken term.

The physics that makes it clean
-------------------------------
In a *held* pose q̇ = q̈ = 0, so M(q)q̈ and C(q,q̇)q̇ vanish and the measured
joint current collapses to

    measured_current(pose) ≈ G(q) + offset + (small stiction).

We therefore drive the arm to a set of static poses (the
`collect_static_gravity.sh` experiment), dwell at each, and average the holding
current over each dwell. That gives one near-noise-free G(q) measurement per
pose per joint — the ideal ground truth for a gravity-only comparison.

What it compares (all as PURE gravity G(q))
-------------------------------------------
  • measured        — mean holding current over each dwell  [raw mA]
  • paper           — the paper authors' identified G(q)    (external/paper_model,
                       already in master-motor mA)
  • a URDF          — Pinocchio RNEA(q, 0, 0)                [Nm → mA]
  • a phi (.npy)    — our regressor inverse_dynamics_phi(q, 0, 0) with the
                       friction columns zeroed                [Nm → mA]

Units. Everything is reported in **raw master-motor current (mA)** — the same
units as the CSV `*_effort` column and as the paper's functions. This is the
assumption-free axis: it never touches k_t or the ×2 dual-motor factor, so it
sidesteps the open ≈0.63 effort-scale anomaly entirely. (Nm is also printed,
via EFFORT_SCALE, for interpretability.)

Metrics, per joint
------------------
  • swing       — peak-to-peak of G over the poses. The headline: a gravity-free
                  model has swing ≈ 0 while measured/paper swing is large.
  • RMSE/MAE    — raw, and after removing the best per-joint constant offset
                  (offset removal isolates the gravity *shape*; the removed
                  constant is the stiction/current bias, not gravity).
  • corr        — correlation of predicted vs measured G over the poses.

A no-model baseline (G = 0) is implicit: its offset-removed RMSE equals the
measured swing's spread, and its swing/corr are 0.

Convention check
----------------
The paper model uses standard DH with internal q2/q3 offsets; our data and the
factory URDF use the robot joint frame. If both `paper` and a URDF are present
the script prints their per-joint correlation over the poses — high correlation
(the CHANGELOG saw r≈0.95 at the shoulder) confirms the angle convention lines
up before any score is trusted.

Usage
-----
    # pure Python, no ROS needed for paper + phi:
    python3 compare_gravity.py --csv data/static_gravity_20260613_183554.csv \
        --phi outputs/npy/<model>.npy

    # add URDF models (needs real Pinocchio via ROS):
    source /opt/ros/humble/setup.bash
    python3 compare_gravity.py --csv data/static_gravity_20260613_183554.csv \
        --urdf urdf/vx300s.urdf --phi outputs/npy/<model>.npy --plot
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

# Reuse the EXACT constants / dynamics of the identification pipeline.
from sysid_feasible import (
    ARM_JOINTS, N_JOINTS, N_PARAMS, N_PARAMS_T, EFFORT_SCALE,
    inverse_dynamics_phi,
)

# Paper authors' published model (cloned into external/, which is .gitignored).
_PAPER_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "external", "paper_model")


# --------------------------------------------------------------------------- #
# 1. Static-dwell detection
# --------------------------------------------------------------------------- #
def detect_static_segments(df, vel_thresh=0.02, min_dur_s=3.0, settle_s=2.0):
    """Find held poses straight from the velocity signal — no time-base alignment.

    A sample is "static" when *every* arm joint has |q̇| < vel_thresh. We take
    each contiguous static run that lasts at least `min_dur_s`, drop the first
    `settle_s` (post-move ringing / settling), and average q and effort over the
    remainder.

    Returns a list of dicts: {q (6,), eff_mA (6,), n, t0, t1}.
    """
    t = df["time"].to_numpy()
    pos = df[[f"{j}_pos" for j in ARM_JOINTS]].to_numpy(float)
    vel = df[[f"{j}_vel" for j in ARM_JOINTS]].to_numpy(float)
    eff = df[[f"{j}_effort" for j in ARM_JOINTS]].to_numpy(float)   # raw mA

    static = np.all(np.abs(vel) < vel_thresh, axis=1)

    segments = []
    n = len(static)
    i = 0
    while i < n:
        if not static[i]:
            i += 1
            continue
        j = i
        while j < n and static[j]:
            j += 1
        # static run is [i, j)
        t0, t1 = t[i], t[j - 1]
        if (t1 - t0) >= min_dur_s:
            keep = t[i:j] >= (t0 + settle_s)
            sl = np.arange(i, j)[keep]
            if sl.size > 0:
                segments.append(dict(
                    q=pos[sl].mean(0),
                    eff_mA=eff[sl].mean(0),
                    n=sl.size,
                    t0=t0, t1=t1,
                ))
        i = j
    return segments


def match_to_commanded(segments, poses_path):
    """Label each detected segment with the nearest commanded pose (validation)."""
    if not poses_path or not os.path.exists(poses_path):
        for k, s in enumerate(segments):
            s["pose"], s["match_err"] = k, np.nan
        return None
    cmd = np.array([p["q"] for p in json.load(open(poses_path))["poses"]])
    for s in segments:
        d = np.linalg.norm(cmd - s["q"], axis=1)
        s["pose"] = int(np.argmin(d))
        s["match_err"] = float(d.min())
    return cmd


# --------------------------------------------------------------------------- #
# 2. Gravity predictors  →  G in mA, shape (n_poses, 6)
# --------------------------------------------------------------------------- #
def _nm_to_ma(tau_nm):
    """Convert per-joint torque [Nm] to master-motor current [mA] (inverse of
    EFFORT_SCALE), so model predictions land on the raw-effort axis."""
    return tau_nm / EFFORT_SCALE


def predict_paper(Q):
    """Paper authors' identified G(q). Output already in master-motor mA."""
    sys.path.insert(0, _PAPER_DIR)
    from Gravity_Compensation_Function import calculate_gravity
    G = np.array([calculate_gravity(*q) for q in Q])
    return G   # already mA


def predict_phi(Q, phi_path):
    """Our regressor's pure gravity: inverse_dynamics_phi at q̇=q̈=0, with the
    per-link friction columns (viscous, Coulomb, offset) zeroed so only the
    rigid-body gravity torque remains. Nm → mA."""
    phi = np.load(phi_path)
    phi_g = phi.copy()
    for i in range(N_JOINTS):                  # zero columns 10,11,12 of each link
        phi_g[i * N_PARAMS + 10:i * N_PARAMS + 13] = 0.0
    z = np.zeros(N_JOINTS)
    G_nm = np.array([inverse_dynamics_phi(q, z, z, phi_g) for q in Q])
    return _nm_to_ma(G_nm)


def predict_urdf(Q, urdf_path):
    """Pinocchio RNEA(q, 0, 0) = G(q). Joints matched by name. Nm → mA.

    Needs the *real* Pinocchio 3.x (source ROS). Raises a clear error otherwise.
    """
    try:
        import pinocchio as pin
    except Exception as e:                                   # pragma: no cover
        raise RuntimeError(
            "import pinocchio failed — for URDF models source ROS first:\n"
            "  source /opt/ros/humble/setup.bash\n"
            f"(underlying error: {e})")
    if not hasattr(pin, "buildModelFromUrdf"):
        raise RuntimeError(
            "pinocchio has no buildModelFromUrdf — the bogus ~/.local "
            "pinocchio 0.4.3 is shadowing the real one. "
            "source /opt/ros/humble/setup.bash (or pip uninstall pinocchio).")

    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()
    idx_v = []
    for name in ARM_JOINTS:
        jid = model.getJointId(name)
        if jid >= model.njoints:
            raise ValueError(f"Joint '{name}' not found in {urdf_path}")
        idx_v.append(model.joints[jid].idx_v)
    idx_v = np.asarray(idx_v)
    idx_q = np.asarray([model.joints[model.getJointId(n)].idx_q for n in ARM_JOINTS])

    v = np.zeros(model.nv)
    a = np.zeros(model.nv)
    G_nm = np.zeros((len(Q), N_JOINTS))
    for k, qk in enumerate(Q):
        q = pin.neutral(model)
        q[idx_q] = qk
        tau = pin.rnea(model, data, q, v, a)
        G_nm[k] = tau[idx_v]
    return _nm_to_ma(G_nm)


# --------------------------------------------------------------------------- #
# 3. Metrics
# --------------------------------------------------------------------------- #
def per_joint_metrics(meas, pred):
    """meas, pred : (n_poses, 6) in mA. Returns dict of (6,) arrays.

    Offset-removed metrics subtract, per joint, the constant c that minimises
    ||meas - (pred + c)|| — i.e. c = mean(meas - pred). This isolates the
    gravity *shape*; the removed c is the stiction / current-bias offset.
    """
    err = pred - meas
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)

    c = np.mean(meas - pred, axis=0)              # best per-joint offset
    err_o = (pred + c) - meas
    rmse_o = np.sqrt(np.mean(err_o ** 2, axis=0))
    mae_o = np.mean(np.abs(err_o), axis=0)

    swing = pred.max(0) - pred.min(0)             # peak-to-peak of G over poses

    corr = np.zeros(N_JOINTS)
    for j in range(N_JOINTS):
        sm, sp = meas[:, j].std(), pred[:, j].std()
        corr[j] = (np.corrcoef(meas[:, j], pred[:, j])[0, 1]
                   if sm > 1e-9 and sp > 1e-9 else 0.0)
    return dict(rmse=rmse, mae=mae, rmse_o=rmse_o, mae_o=mae_o,
                swing=swing, corr=corr, offset=c)


def print_table(name, m, meas_swing):
    print(f"\n  {name}")
    print(f"    {'joint':<14}{'swing':>9}{'meas_sw':>9}{'RMSE':>9}"
          f"{'RMSE_off':>10}{'MAE_off':>9}{'corr':>7}")
    for j, jn in enumerate(ARM_JOINTS):
        print(f"    {jn:<14}{m['swing'][j]:>9.1f}{meas_swing[j]:>9.1f}"
              f"{m['rmse'][j]:>9.1f}{m['rmse_o'][j]:>10.1f}"
              f"{m['mae_o'][j]:>9.1f}{m['corr'][j]:>7.2f}")
    # mean over the gravity-bearing joints only (waist/wrist_rotate carry ~none)
    gj = [ARM_JOINTS.index(x) for x in ("shoulder", "elbow", "wrist_angle")]
    print(f"    {'-- grav.joints':<14}{'':>9}{'':>9}{m['rmse'][gj].mean():>9.1f}"
          f"{m['rmse_o'][gj].mean():>10.1f}{m['mae_o'][gj].mean():>9.1f}"
          f"{m['corr'][gj].mean():>7.2f}   (mean over shoulder/elbow/wrist_angle)")


# --------------------------------------------------------------------------- #
def maybe_plot(Q, meas, preds, out_path, show=True):
    import matplotlib.pyplot as plt

    order = np.lexsort([Q[:, i] for i in range(N_JOINTS)])  # stable-ish ordering
    x = np.arange(len(meas))
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(x, meas[order, j], "ko-", lw=1.2, ms=4, label="measured")
        for label, P in preds.items():
            ax.plot(x, P[order, j], "o--", ms=3, lw=0.9, label=label)
        ax.set_ylabel(f"G {ARM_JOINTS[j]} [mA]")
        if j >= 4:
            ax.set_xlabel("pose (sorted)")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle("Gravity per static pose: measured vs models  [master-motor mA]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n  Saved plot → {out_path}")
    if show:
        try:
            plt.show()
        except Exception as e:                                   # headless / no GUI
            print(f"  [info] could not open an interactive window ({e}); "
                  "see the saved PNG.")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/static_gravity_20260613_183554.csv",
                   help="static-pose CSV (collect_static_gravity.sh output)")
    p.add_argument("--poses", default=None,
                   help="poses .json (default: <csv>.poses.json) — for labelling only")
    p.add_argument("--phi", action="append", default=[],
                   help="identified phi .npy (repeatable)")
    p.add_argument("--urdf", action="append", default=[],
                   help="URDF to score (repeatable; needs ROS/Pinocchio)")
    p.add_argument("--no-paper", action="store_true",
                   help="skip the paper authors' model")
    p.add_argument("--vel-thresh", type=float, default=0.02,
                   help="|q̇| below this (all joints) = static [rad/s]")
    p.add_argument("--min-dur", type=float, default=3.0,
                   help="minimum static-run duration to accept [s]")
    p.add_argument("--settle", type=float, default=2.0,
                   help="discard this much at the start of each dwell [s]")
    p.add_argument("--plot", action="store_true",
                   help="open an interactive plot window (also saves a PNG)")
    p.add_argument("--no-show", action="store_true",
                   help="with --plot, save the PNG but don't open a window")
    args = p.parse_args()

    poses_path = args.poses or os.path.splitext(args.csv)[0] + ".poses.json"

    print("=" * 70)
    print("GRAVITY-ONLY model comparison on static poses  (units: master-motor mA)")
    print("=" * 70)
    print(f"  data : {args.csv}")

    df = pd.read_csv(args.csv)
    segs = detect_static_segments(df, args.vel_thresh, args.min_dur, args.settle)
    if not segs:
        sys.exit("No static dwells detected — loosen --vel-thresh / --min-dur.")
    match_to_commanded(segs, poses_path)

    Q = np.array([s["q"] for s in segs])
    meas = np.array([s["eff_mA"] for s in segs])
    print(f"  detected {len(segs)} static dwells "
          f"(vel<{args.vel_thresh} rad/s, ≥{args.min_dur}s, settle {args.settle}s)")
    me = np.array([s["match_err"] for s in segs])
    if np.isfinite(me).any():
        print(f"  pose-match residual: max {np.nanmax(me):.3f} rad "
              f"(should be ≪ joint spacing)")
        if np.nanmax(me) > 0.15:
            print("  ⚠ large match residual — detected holds may not be the commanded poses")

    meas_swing = meas.max(0) - meas.min(0)

    # Build predictors.
    preds = {}
    if not args.no_paper:
        if os.path.isdir(_PAPER_DIR):
            try:
                preds["paper"] = predict_paper(Q)
            except Exception as e:
                print(f"  [warn] paper model skipped: {e}")
        else:
            print(f"  [warn] paper model not found at {_PAPER_DIR} — skipped")
    for u in args.urdf:
        try:
            preds[f"urdf:{os.path.basename(u)}"] = predict_urdf(Q, u)
        except Exception as e:
            print(f"  [warn] URDF {u} skipped: {e}")
    for f in args.phi:
        try:
            preds[f"phi:{os.path.basename(f)}"] = predict_phi(Q, f)
        except Exception as e:
            print(f"  [warn] phi {f} skipped: {e}")

    if not preds:
        sys.exit("No models to compare (need at least --phi/--urdf or the paper model).")

    # Convention check: paper vs first URDF correlation over the poses.
    urdf_keys = [k for k in preds if k.startswith("urdf:")]
    if "paper" in preds and urdf_keys:
        gj = [ARM_JOINTS.index(x) for x in ("shoulder", "elbow", "wrist_angle")]
        P, U = preds["paper"], preds[urdf_keys[0]]
        print("\n  Convention check — corr(paper G, factory G) over poses "
              "(high = angle frames agree):")
        for j in gj:
            c = np.corrcoef(P[:, j], U[:, j])[0, 1]
            print(f"    {ARM_JOINTS[j]:<14}{c:>7.2f}")

    print("\n  Measured gravity swing (peak-to-peak holding current over poses) [mA]:")
    print("    " + "  ".join(f"{jn}={meas_swing[j]:.0f}"
                             for j, jn in enumerate(ARM_JOINTS)))
    print("    (a model that captures gravity must reproduce these swings; "
          "a gravity-free model shows swing ≈ 0)")

    print("\n" + "-" * 70)
    print("PER-MODEL gravity prediction vs measured  [mA]")
    print("  swing=model peak-to-peak · meas_sw=measured peak-to-peak · "
          "RMSE_off/corr after offset removal")
    print("-" * 70)
    for name, P in preds.items():
        print_table(name, per_joint_metrics(meas, P), meas_swing)

    if args.plot:
        os.makedirs("figures", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.csv))[0]
        maybe_plot(Q, meas, preds, os.path.join("figures", f"gravity_compare_{base}.png"),
                   show=not args.no_show)


if __name__ == "__main__":
    main()
