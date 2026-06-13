#!/usr/bin/env python3
"""Benchmark an identified model against the paper authors' PUBLISHED model.

Source paper authors' repo (vendored at external/paper_model):
  github.com/MomaniMutaz/ViperX-300-6DoF-Robotic-Arm-Dynamical-Model
Their model ships as symbolic M/V/G/friction functions with the identified
*base* parameters baked in — NOT a URDF, and base parameters cannot be split
back into per-link inertials, so a URDF-to-URDF diff is impossible. The only
faithful comparison is at the level of predicted physics.

WHAT THIS COMPARES
  Predicted joint current  τ̂ = G(q) + F(q̇)   [milliAmps]
  for BOTH the paper model and our identified φ, against measured current.

  - Gravity G(q): paper's `calculate_gravity` (their Gravity_Compensation_
    Function.py) vs ours from sysid_feasible's own Newton-Euler regressor
    (verified == Pinocchio RNEA to 3e-7 Nm, tools/test_phi_urdf_consistency.py).
  - Friction F(q̇): each model's OWN identified Fv·q̇ + Fc·sign(q̇) + F0
    (paper base-params 37..54; ours from φ). NOT fitted on the test data — so
    this avoids the friction-fit masking documented in THESIS_NOTES (2026-06-13).
  - M(q)·q̈ + V(q,q̇)·q̇ are OMITTED FOR BOTH (symmetric): the paper's M/V are
    ~80k lines of MATLAB, they need noisy q̈, and they are small at this slow
    trajectory's speeds. The omitted inertia/Coriolis is a common residual.

UNITS — everything in mA (the robot's native control variable and the paper's
native identification unit). Ours and measured are converted from Nm by the
SAME EFFORT_SCALE used in identification; the paper model stays pristine, so we
impose no k_t / motor-count assumption on it.

THE 0.63 ANOMALY — measured gravity-correlated current is consistently ~0.63×
the CAD/paper gravity (CHANGELOG 2026-06-13). That is a dimensionless amplitude
gap, identical in mA or Nm, and it would penalise the paper model on raw RMSE
for a pure calibration offset. So we ALSO report a scale-allowed view: the
per-joint optimal gain α (meas ≈ α·pred) and the residual RMSE after that gain,
which isolates SHAPE (does the model capture gravity's geometry) from amplitude.

Run (no ROS needed):
  python3 tools/compare_paper_model.py --csv data/<run>.csv \
      --phi outputs/npy/<model>.npy --drop-glitches
"""
import argparse, os, re, sys, warnings

import numpy as np

warnings.filterwarnings("ignore", message="Mean of empty slice")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import sysid_feasible as sf  # noqa: E402


# --------------------------------------------------------------------------- #
#  Paper model loaders
# --------------------------------------------------------------------------- #
def load_paper(paper_dir):
    """Return (gravity_fn, Fv, Fc, F0) from the vendored paper repo, all mA."""
    sys.path.insert(0, paper_dir)
    from Gravity_Compensation_Function import calculate_gravity  # noqa

    # The 54-element base-parameter vector is local to calculate_gravity; parse
    # it straight from the source so the friction map can't drift out of sync.
    src = open(os.path.join(paper_dir, "Gravity_Compensation_Function.py")).read()
    block = re.search(r"coefficients\s*=\s*np\.array\(\[(.*?)\]\)", src, re.S).group(1)
    c = np.array([float(x) for x in
                  re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", block)])
    assert c.size == 54, f"expected 54 base coefficients, parsed {c.size}"

    # Frictions.m, 1-indexed:  Fi = c(37+2(i-1))·q̇ + c(38+2(i-1))·sign(q̇) + c(49+(i-1))
    Fv = c[[36, 38, 40, 42, 44, 46]]
    Fc = c[[37, 39, 41, 43, 45, 47]]
    F0 = c[[48, 49, 50, 51, 52, 53]]
    return calculate_gravity, Fv, Fc, F0


def paper_gravity_mA(calc, q):
    """(N,6) joint angles -> (N,6) gravity current [mA]. Per-sample because the
    paper function hardcodes G1=0.0 (scalar), which breaks array broadcasting."""
    return np.array([calc(*q[n]) for n in range(q.shape[0])])


# --------------------------------------------------------------------------- #
#  Our model
# --------------------------------------------------------------------------- #
def our_gravity_Nm(phi, q):
    """(N,6) -> (N,6) gravity torque [Nm] from our identified φ, friction zeroed,
    evaluated at q̇=q̈=0 via sysid_feasible's Newton-Euler inverse dynamics."""
    phi_g = phi.copy()
    for i in range(sf.N_JOINTS):
        phi_g[i * sf.N_PARAMS + 10:i * sf.N_PARAMS + 13] = 0.0   # Fv, Fc, F0
    z = np.zeros(sf.N_JOINTS)
    return np.array([sf.inverse_dynamics_phi(q[n], z, z, phi_g) for n in range(q.shape[0])])


def our_friction(phi):
    Fv = np.array([phi[i * sf.N_PARAMS + 10] for i in range(sf.N_JOINTS)])
    Fc = np.array([phi[i * sf.N_PARAMS + 11] for i in range(sf.N_JOINTS)])
    F0 = np.array([phi[i * sf.N_PARAMS + 12] for i in range(sf.N_JOINTS)])
    return Fv, Fc, F0


# --------------------------------------------------------------------------- #
#  Metrics
# --------------------------------------------------------------------------- #
def per_joint_stats(pred, meas):
    """pred, meas: (N,6). Returns dict of (6,) arrays."""
    err = pred - meas
    rmse = np.sqrt(np.mean(err ** 2, axis=0))
    mae = np.mean(np.abs(err), axis=0)
    ss_tot = np.sum((meas - meas.mean(0)) ** 2, axis=0)
    ss_res = np.sum(err ** 2, axis=0)
    r2 = 1.0 - ss_res / np.where(ss_tot == 0, np.nan, ss_tot)
    # Pearson r and amplitude-allowed gain α (meas ≈ α·pred) + post-gain RMSE
    r = np.zeros(6); alpha = np.zeros(6); rmse_g = np.zeros(6)
    for j in range(6):
        p, m = pred[:, j], meas[:, j]
        denom = np.dot(p, p)
        alpha[j] = np.dot(p, m) / denom if denom > 0 else np.nan
        rmse_g[j] = np.sqrt(np.mean((m - alpha[j] * p) ** 2))
        sp, sm = p.std(), m.std()
        r[j] = np.mean((p - p.mean()) * (m - m.mean())) / (sp * sm) if sp > 0 and sm > 0 else np.nan
    return dict(rmse=rmse, mae=mae, r2=r2, r=r, alpha=alpha, rmse_gain=rmse_g)


JN = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def print_block(title, stats):
    print(f"\n  {title}")
    print(f"    {'joint':<14}{'RMSE':>9}{'MAE':>9}{'R²':>8}{'corr r':>8}"
          f"{'gain α':>9}{'RMSE|α':>9}")
    for j, nm in enumerate(JN):
        print(f"    {nm:<14}{stats['rmse'][j]:9.1f}{stats['mae'][j]:9.1f}"
              f"{stats['r2'][j]:8.3f}{stats['r'][j]:8.3f}"
              f"{stats['alpha'][j]:9.3f}{stats['rmse_gain'][j]:9.1f}")
    print(f"    {'-- mean --':<14}{stats['rmse'].mean():9.1f}{stats['mae'].mean():9.1f}"
          f"{np.nanmean(stats['r2']):8.3f}{np.nanmean(stats['r']):8.3f}"
          f"{np.nanmean(stats['alpha']):9.3f}{stats['rmse_gain'].mean():9.1f}")


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--phi", required=True, help="our identified φ (.npy)")
    ap.add_argument("--paper-dir", default=os.path.join(REPO, "external", "paper_model"))
    ap.add_argument("--stride", type=int, default=20,
                    help="subsample for our per-sample NE eval (default 20)")
    ap.add_argument("--fs", type=float, default=50.0)
    ap.add_argument("--fc", type=float, default=10.0)
    ap.add_argument("--drop-glitches", action="store_true")
    args = ap.parse_args()

    print("=" * 70)
    print("Paper-model benchmark — predicted current G(q)+F(q̇) [mA] vs measured")
    print("=" * 70)
    print(f"  data  : {args.csv}")
    print(f"  ours  : {args.phi}")
    print(f"  paper : {args.paper_dir}")

    calc, pFv, pFc, pF0 = load_paper(args.paper_dir)
    phi = np.load(args.phi)

    t, q, qd, qdd, tau_Nm = sf.load_and_filter(
        args.csv, fs=args.fs, fc=args.fc, stride=args.stride,
        drop_glitches=args.drop_glitches)
    print(f"  samples: {q.shape[0]}  (stride {args.stride})   duration: {t[-1]:.1f} s")

    # Everything to mA.  measured & ours via the identification EFFORT_SCALE.
    es = sf.EFFORT_SCALE                       # (6,) Nm per mA
    meas = tau_Nm / es

    paper_pred = paper_gravity_mA(calc, q) + (qd * pFv + np.sign(qd) * pFc + pF0)

    oFv, oFc, oF0 = our_friction(phi)
    our_G_mA = our_gravity_Nm(phi, q) / es
    our_F_mA = (qd * oFv + np.sign(qd) * oFc + oF0) / es
    our_pred = our_G_mA + our_F_mA

    zero = np.zeros_like(meas)

    print("\n" + "-" * 70)
    print("FULL PREDICTION  τ̂ = G(q) + F(q̇)   [mA]   (vs measured current)")
    print("  α = optimal gain (meas ≈ α·pred); RMSE|α = residual after gain")
    print("-" * 70)
    s0 = per_joint_stats(zero, meas)
    sP = per_joint_stats(paper_pred, meas)
    sB = per_joint_stats(our_pred, meas)
    print_block("0: no-model baseline (τ̂ = 0)", s0)
    print_block("A: PAPER published model", sP)
    print_block("B: OUR identified model", sB)

    base = s0["rmse"].mean()
    print(f"\n  Margin over no-model baseline (mean RMSE {base:.1f} mA; positive = explains signal):")
    for tag, s in (("A paper", sP), ("B ours", sB)):
        d = base - s["rmse"].mean()
        flag = "" if d > 0 else "   ← NOT better than no model"
        print(f"    {tag}: {d:+.1f} mA ({100*d/base:+.1f}%){flag}")

    # Gravity-only, model-vs-model: does our identified gravity agree with paper's?
    print("\n" + "-" * 70)
    print("GRAVITY ONLY — paper G(q) vs our G(q)  [mA]  (model-vs-model, no measured)")
    print("  r = shape agreement; α = our/paper amplitude ratio")
    print("-" * 70)
    paper_G = paper_gravity_mA(calc, q)
    sG = per_joint_stats(our_G_mA, paper_G)   # 'meas'=paper G, 'pred'=our G
    print(f"    {'joint':<14}{'corr r':>9}{'α(our/paper)':>14}{'RMSE':>10}")
    for j, nm in enumerate(JN):
        print(f"    {nm:<14}{sG['r'][j]:9.3f}{1.0/sG['alpha'][j] if sG['alpha'][j] else np.nan:14.3f}"
              f"{sG['rmse'][j]:10.1f}")

    print("\n" + "=" * 70)
    print("VERDICT (full G+F, lower mean RMSE vs measured = better)")
    print("=" * 70)
    print(f"  A  paper model   {sP['rmse'].mean():8.1f} mA")
    print(f"  B  our model     {sB['rmse'].mean():8.1f} mA")
    print(f"  (no-model        {s0['rmse'].mean():8.1f} mA)")


if __name__ == "__main__":
    main()
