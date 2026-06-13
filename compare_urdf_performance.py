#!/usr/bin/env python3
"""
compare_urdf_performance.py — Validate a URDF against recorded robot data
========================================================================

The honest way to ask "is my identified URDF better than the original?" is to
check how well each model *predicts the torques the real robot actually used*.

For every sample of a recorded trajectory we know (q, dq, ddq) and the measured
joint effort. Pinocchio's RNEA (recursive Newton–Euler inverse dynamics) turns
(q, dq, ddq) + a URDF's mass/inertia parameters into the predicted torque. The
URDF whose predicted torque is closest to the measured torque is the better
dynamic model.

    measured torque  ──┐
                       ├──►  per-joint RMSE / MAE / relative-error / R²
    RNEA(URDF, q,dq,ddq)┘

We reuse sysid_feasible.load_and_filter so the filtering, derivative estimation
and effort→Nm scaling are *identical* to the identification pipeline — otherwise
the comparison would not be apples-to-apples.

Note on friction: plain RNEA models rigid-body dynamics only (gravity + inertia +
Coriolis), not joint friction. Friction shows up as a residual that *neither*
URDF can explain, which inflates both errors equally. Pass --friction to also
report errors after a per-joint least-squares fit of viscous + Coulomb + offset
friction (fit independently for each model), isolating the rigid-body quality.

**The rigid-body-only, per-joint numbers are the PRIMARY criterion.** The
friction-fitted numbers are a secondary diagnostic: the nuisance basis is
fitted on the evaluation data itself and can absorb most of the measured
torque on its own (found 2026-06-13: the basis with *no* rigid-body model
scored within 0.01 Nm of the then-best model — see CHANGELOG). Every report
therefore includes a **no-model baseline** (τ_pred = 0 pushed through the
identical protocol); a model that does not beat the baseline decisively has
explained nothing, regardless of how it ranks against another model.

Usage
-----
    python3 compare_urdf_performance.py                 # defaults below
    python3 compare_urdf_performance.py --csv data/traj_run_20260518_143818.csv \
        --urdf-a urdf/vx300s.urdf \
        --urdf-b urdf/traj_run_20260518_143818__sysid_feasible-v1-1__cfg-5d6e6cae.urdf \
        --friction --plot

Lower RMSE / relative-error and higher R² = better model.
"""

import argparse
import os

import numpy as np
import pinocchio as pin

# Reuse the EXACT data pipeline used for identification.
from sysid_feasible import load_and_filter, ARM_JOINTS, N_JOINTS


# --------------------------------------------------------------------------- #
# RNEA torque prediction
# --------------------------------------------------------------------------- #
def predict_torques(urdf_path, q_arm, dq_arm, ddq_arm):
    """Run inverse dynamics for every sample and return the 6 arm-joint torques.

    Joints are matched to the model *by name*, so this works whether the URDF is
    arm-only (nq=6) or the full robot incl. gripper/fingers (nq=10). Non-arm
    joints stay at the neutral configuration with zero velocity/acceleration.

    Parameters
    ----------
    q_arm, dq_arm, ddq_arm : (N, 6) arrays for the 6 arm joints, in ARM_JOINTS order.

    Returns
    -------
    tau_pred : (N, 6) predicted torque [Nm] for the 6 arm joints.
    """
    model = pin.buildModelFromUrdf(urdf_path)
    data = model.createData()

    # Map each arm joint name -> (configuration index, velocity index) in THIS model.
    idx_q, idx_v = [], []
    for name in ARM_JOINTS:
        jid = model.getJointId(name)
        if jid >= model.njoints:
            raise ValueError(f"Joint '{name}' not found in {urdf_path}")
        joint = model.joints[jid]
        if joint.nq != 1 or joint.nv != 1:
            raise ValueError(
                f"Joint '{name}' in {urdf_path} has nq={joint.nq}, nv={joint.nv}; "
                "expected a simple 1-DOF revolute joint."
            )
        idx_q.append(joint.idx_q)
        idx_v.append(joint.idx_v)
    idx_q = np.asarray(idx_q)
    idx_v = np.asarray(idx_v)

    N = q_arm.shape[0]
    tau_pred = np.zeros((N, N_JOINTS))
    q = pin.neutral(model)
    v = np.zeros(model.nv)
    a = np.zeros(model.nv)

    for k in range(N):
        q[idx_q] = q_arm[k]
        v[idx_v] = dq_arm[k]
        a[idx_v] = ddq_arm[k]
        tau = pin.rnea(model, data, q, v, a)
        tau_pred[k] = tau[idx_v]

    return tau_pred, model.name or os.path.basename(urdf_path)


# --------------------------------------------------------------------------- #
# Optional per-joint friction fit (viscous + Coulomb + constant offset)
# --------------------------------------------------------------------------- #
def fit_friction(tau_meas, tau_pred, dq, ddq=None):
    """Least-squares fit residual = tau_meas - tau_pred to f(dq) per joint.

        f = b*dq + c*sign(dq) + d            (+ Ia*ddq  if ddq is given)

    Passing ddq adds a per-joint reflected motor-inertia column Ia·q̈ to the
    nuisance basis (--fit-ia). Like friction, it is fitted per model per
    dataset — symmetric across A and B — so the comparison still isolates the
    rigid-body (link-parameter) quality; URDFs identified with
    sysid_feasible --motor-inertia carry no Ia in their <inertial> blocks, so
    this supplies the q̈-proportional torque their rigid-body part deliberately
    excludes. The fitted Ia [kg·m²] are returned for plausibility checks
    (expect ~N²·J_rotor at ≈270:1 gearing).

    Returns (friction-compensated prediction tau_pred + f, Ia or None).
    """
    tau_comp = tau_pred.copy()
    ia_fit = np.zeros(N_JOINTS) if ddq is not None else None
    for j in range(N_JOINTS):
        residual = tau_meas[:, j] - tau_pred[:, j]
        cols = [dq[:, j], np.sign(dq[:, j]), np.ones_like(dq[:, j])]
        if ddq is not None:
            cols.append(ddq[:, j])
        A = np.column_stack(cols)
        coef, *_ = np.linalg.lstsq(A, residual, rcond=None)
        tau_comp[:, j] = tau_pred[:, j] + A @ coef
        if ddq is not None:
            ia_fit[j] = coef[3]
    return tau_comp, ia_fit


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metrics(tau_meas, tau_pred):
    """Per-joint RMSE [Nm], MAE [Nm], relative error [-], R² [-]."""
    err = tau_pred - tau_meas
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)

    denom = np.maximum(np.abs(tau_meas), np.abs(tau_pred))
    denom[denom < 1e-9] = 1e-9
    rel = np.mean(np.abs(err) / denom, axis=0)

    ss_res = np.sum(err ** 2, axis=0)
    ss_tot = np.sum((tau_meas - tau_meas.mean(axis=0)) ** 2, axis=0)
    ss_tot[ss_tot < 1e-12] = 1e-12
    r2 = 1.0 - ss_res / ss_tot
    return rmse, mae, rel, r2


def print_report(label, tau_meas, tau_pred):
    rmse, mae, rel, r2 = metrics(tau_meas, tau_pred)
    print(f"\n  {label}")
    print(f"    {'joint':<14}{'RMSE[Nm]':>10}{'MAE[Nm]':>10}{'rel.err':>10}{'R²':>8}")
    for j, name in enumerate(ARM_JOINTS):
        print(f"    {name:<14}{rmse[j]:>10.3f}{mae[j]:>10.3f}{rel[j]:>10.3f}{r2[j]:>8.3f}")
    print(f"    {'-- mean --':<14}{rmse.mean():>10.3f}{mae.mean():>10.3f}"
          f"{rel.mean():>10.3f}{r2.mean():>8.3f}")
    return rmse, rel


def baseline_margin(rmse_0, rmse_a, rmse_b):
    """How much each model beats the no-model baseline (τ_pred = 0 through the
    identical protocol). A model that is not clearly below the baseline has
    explained nothing — any A-vs-B ranking below is then meaningless."""
    print(f"\n  Margin over no-model baseline (mean RMSE {rmse_0.mean():.3f} Nm; "
          f"positive = model explains signal):")
    for label, rmse in (("A", rmse_a), ("B", rmse_b)):
        margin = rmse_0.mean() - rmse.mean()
        pct = 100.0 * margin / rmse_0.mean()
        flag = "" if pct > 10.0 else "   ← NOT meaningfully better than no model"
        print(f"    {label}: {margin:+.3f} Nm ({pct:+.1f}%){flag}")


def winner_summary(name_a, rmse_a, name_b, rmse_b):
    print("\n" + "=" * 64)
    print("VERDICT (lower total RMSE = better dynamic model)")
    print("=" * 64)
    tot_a, tot_b = rmse_a.mean(), rmse_b.mean()
    print(f"  A  {name_a:<45}{tot_a:>8.3f} Nm")
    print(f"  B  {name_b:<45}{tot_b:>8.3f} Nm")
    if tot_b < tot_a:
        improve = 100.0 * (tot_a - tot_b) / tot_a
        print(f"\n  → B is better by {improve:.1f}% mean RMSE.")
    elif tot_a < tot_b:
        worse = 100.0 * (tot_b - tot_a) / tot_a
        print(f"\n  → A is better; B is {worse:.1f}% worse mean RMSE.")
    else:
        print("\n  → Tie.")
    print("\n  Per-joint RMSE improvement of B over A [Nm] (positive = B better):")
    for j, name in enumerate(ARM_JOINTS):
        print(f"    {name:<14}{rmse_a[j] - rmse_b[j]:>+8.3f}")


# --------------------------------------------------------------------------- #
def maybe_plot(t, tau_meas, tau_pred_a, tau_pred_b, name_a, name_b, out_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 2, figsize=(13, 9), sharex=True)
    for j, ax in enumerate(axes.flat):
        ax.plot(t, tau_meas[:, j], lw=1.0, color="k", label="measured")
        ax.plot(t, tau_pred_a[:, j], "--", lw=0.9, label=f"A: {name_a}")
        ax.plot(t, tau_pred_b[:, j], "--", lw=0.9, label=f"B: {name_b}")
        ax.set_ylabel(f"τ {ARM_JOINTS[j]} [Nm]")
        if j >= 4:
            ax.set_xlabel("time [s]")
    axes.flat[0].legend(fontsize=8, loc="best")
    fig.suptitle("Measured vs RNEA-predicted joint torque")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f"\n  Saved plot → {out_path}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default="data/traj_run_20260518_143818.csv",
                   help="recorded trajectory CSV")
    p.add_argument("--urdf-a", default="urdf/vx300s.urdf",
                   help="baseline URDF (original)")
    p.add_argument("--urdf-b",
                   default="urdf/traj_run_20260518_143818__sysid_feasible-v1-1__cfg-5d6e6cae.urdf",
                   help="your identified URDF")
    p.add_argument("--fc", type=float, default=10.0, help="low-pass cutoff [Hz]")
    p.add_argument("--stride", type=int, default=1, help="subsample stride")
    p.add_argument("--friction", action="store_true",
                   help="also report errors with per-joint friction fitted")
    p.add_argument("--fit-ia", action="store_true",
                   help="add a per-joint reflected motor-inertia column Ia·q̈ "
                        "to the friction fit (both models; implies a different "
                        "protocol — numbers are NOT comparable to --friction-"
                        "only runs). Needed to validate URDFs identified with "
                        "sysid_feasible --motor-inertia. Requires --friction.")
    p.add_argument("--plot", action="store_true", help="save a torque comparison plot")
    p.add_argument("--drop-glitches", action="store_true",
                   help="remove all-joints=−π sync-read dropout rows before "
                        "filtering (same flag as sysid_feasible)")
    args = p.parse_args()

    print("=" * 64)
    print("URDF dynamic-model comparison via inverse-dynamics torque prediction")
    print("=" * 64)
    print(f"  data : {args.csv}")
    print(f"  A    : {args.urdf_a}")
    print(f"  B    : {args.urdf_b}")

    # Identical pipeline to identification: filter, derivatives, effort→Nm.
    t, q, dq, ddq, tau_meas = load_and_filter(args.csv, fc=args.fc, stride=args.stride,
                                              drop_glitches=args.drop_glitches)
    print(f"  samples: {len(t)}   duration: {t[-1]:.1f} s")

    tau_a, name_a = predict_torques(args.urdf_a, q, dq, ddq)
    tau_b, name_b = predict_torques(args.urdf_b, q, dq, ddq)

    tau_zero = np.zeros_like(tau_meas)

    print("\n" + "-" * 64)
    print("RIGID-BODY ONLY (gravity + inertia + Coriolis, no friction)")
    print("— PRIMARY CRITERION —")
    print("-" * 64)
    rmse_0, _ = print_report("0: no-model baseline (τ_pred = 0)", tau_meas, tau_zero)
    rmse_a, _ = print_report(f"A: {os.path.basename(args.urdf_a)}", tau_meas, tau_a)
    rmse_b, _ = print_report(f"B: {os.path.basename(args.urdf_b)}", tau_meas, tau_b)
    baseline_margin(rmse_0, rmse_a, rmse_b)
    winner_summary(os.path.basename(args.urdf_a), rmse_a,
                   os.path.basename(args.urdf_b), rmse_b)

    if args.fit_ia and not args.friction:
        p.error("--fit-ia requires --friction")

    if args.friction:
        ddq_fit = ddq if args.fit_ia else None
        print("\n" + "-" * 64)
        print("WITH PER-JOINT FRICTION FITTED (viscous + Coulomb + offset"
              + (" + motor inertia Ia·q̈" if args.fit_ia else "") + ")")
        print("— secondary diagnostic; nuisance basis is fitted on this data —")
        print("-" * 64)
        tau_0_f, _    = fit_friction(tau_meas, tau_zero, dq, ddq_fit)
        tau_a_f, ia_a = fit_friction(tau_meas, tau_a, dq, ddq_fit)
        tau_b_f, ia_b = fit_friction(tau_meas, tau_b, dq, ddq_fit)
        rmse_0_f, _ = print_report("0: no-model baseline (τ_pred = 0)", tau_meas, tau_0_f)
        rmse_a_f, _ = print_report(f"A: {os.path.basename(args.urdf_a)}", tau_meas, tau_a_f)
        rmse_b_f, _ = print_report(f"B: {os.path.basename(args.urdf_b)}", tau_meas, tau_b_f)
        baseline_margin(rmse_0_f, rmse_a_f, rmse_b_f)
        if args.fit_ia:
            print(f"\n  Fitted reflected motor inertia Ia [kg·m²] "
                  f"(plausible scale: ~N²·J_rotor):")
            print(f"    {'joint':<14}{'A':>12}{'B':>12}")
            for j, name in enumerate(ARM_JOINTS):
                print(f"    {name:<14}{ia_a[j]:>12.5f}{ia_b[j]:>12.5f}")
        winner_summary(os.path.basename(args.urdf_a), rmse_a_f,
                       os.path.basename(args.urdf_b), rmse_b_f)

    if args.plot:
        os.makedirs("figures", exist_ok=True)
        base = os.path.splitext(os.path.basename(args.csv))[0]
        maybe_plot(t, tau_meas, tau_a, tau_b,
                   os.path.basename(args.urdf_a), os.path.basename(args.urdf_b),
                   os.path.join("figures", f"urdf_compare_{base}.png"))


if __name__ == "__main__":
    main()
