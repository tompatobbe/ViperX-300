#!/usr/bin/env python3
"""Verify the explicit M(q), C(q,q̇), G(q) of the identified model (control/dynamics.py).

WHY: the controller so far used only G(q). Before using M and C (computed-torque
/ inverse-dynamics feed-forward) — and to report them in the thesis — they must
be shown to be (a) the same model that was identified and validated, and
(b) physically sound. Checks, over random states inside the controller's joint
limits:

  1. Independent M:  M from Pinocchio CRBA (+Ia armature) vs M built column by
     column from sysid_feasible's OWN Newton–Euler code on φ
     (M[:,j] = τ(q, 0, e_j) − τ(q, 0, 0), friction zeroed). Different code,
     different frames (DH vs URDF) → agreement means M *is* the identified model.
  2. Symmetry of the independently built M (a property, not a construction).
  3. Positive definiteness: eigenvalues of M over the workspace (min/max, cond).
  4. Skew-symmetry: N = Ṁ − 2C must satisfy N + Nᵀ = 0 (passivity property used
     in every PD+G / computed-torque stability proof).
  5. Reconstruction: M q̈ + C q̇ + G == RNEA == sysid inverse_dynamics_phi.
  6. Paper comparison (external/paper_model, translated by tools/paper_dynamics.py):
     6a translator check: paper G from the .m vs the authors' own Python port;
     6b paper self-consistency: their V(q,q̇) vs the Christoffel C q̇ of their M;
     6c ours vs paper, in mA: per-entry correlation over the states (scale-free)
        and diagonal magnitude ratio.
  7. (--csv) Magnitude budget on a recorded trajectory: RMS of M q̈, C q̇, G per
     joint — quantifies how much of the torque PD+G leaves to feedback — and a
     residual test: does adding the model's M q̈ + C q̇ shrink the measured-torque
     residual, and with what best-fit scale k (≈1 ⇒ right size)?
     --paper-budget runs the same test on the paper's M, V (slow).

Read-only analysis, no hardware. Needs real Pinocchio:
    source /opt/ros/humble/setup.bash
    python3 tools/check_dynamics_matrices.py
    python3 tools/check_dynamics_matrices.py --csv data/traj_run_200hz_20260623_145333.csv
    python3 tools/check_dynamics_matrices.py --csv data/traj_run_200hz_20260623_145333.csv --paper-budget
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "tools"))
import sysid_feasible as sf  # noqa: E402
from control.dynamics import ArmDynamics, DEFAULT_PHI, DEFAULT_URDF  # noqa: E402

J = sf.ARM_JOINTS
# Controller joint limits (control/pd_grav_control.py LIMITS_LO/HI; duplicated
# here because that module needs rclpy at import time).
LIMITS_LO = np.array([-2.80, -1.50, -0.20, -1.50, -1.50, -2.80])
LIMITS_HI = np.array([2.80, 0.80, 1.00, 1.50, 1.50, 2.80])


def phi_rigid(phi_path):
    """φ with friction (Fv, Fc, F0) zeroed — rigid body + Ia, like RNEA."""
    phi = np.load(phi_path).copy()
    for i in range(sf.N_JOINTS):
        phi[i * sf.N_PARAMS + 10:i * sf.N_PARAMS + 13] = 0.0
    return phi


def M_from_phi(phi, q):
    z = np.zeros(6)
    g = sf.inverse_dynamics_phi(q, z, z, phi)
    return np.column_stack([sf.inverse_dynamics_phi(q, z, e, phi) - g for e in np.eye(6)])


def christoffel_V(M_fn, q, dq, eps=1e-5):
    """V = C(q,q̇)q̇ from any M(q) via Christoffel symbols (finite differences):
    V_i = Σ_jk (∂M_ij/∂q_k − ½ ∂M_jk/∂q_i) q̇_j q̇_k."""
    dM = np.array([(M_fn(q + eps * e) - M_fn(q - eps * e)) / (2 * eps)
                   for e in np.eye(6)])                      # dM[k] = ∂M/∂q_k
    Mdot = np.einsum("kij,k->ij", dM, dq)                    # Σ_k ∂M_ij/∂q_k q̇_k
    grad = np.einsum("ijk,j,k->i", dM, dq, dq)               # ∂(q̇ᵀMq̇)/∂q_i
    return Mdot @ dq - 0.5 * grad


def rel(a, b):
    return np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--urdf", default=DEFAULT_URDF)
    ap.add_argument("--phi", default=DEFAULT_PHI)
    ap.add_argument("--n", type=int, default=200, help="random states (default 200)")
    ap.add_argument("--vmax", type=float, default=1.5, help="|q̇| bound [rad/s]")
    ap.add_argument("--amax", type=float, default=3.0, help="|q̈| bound [rad/s²]")
    ap.add_argument("--no-paper", action="store_true", help="skip the paper comparison")
    ap.add_argument("--csv", help="recorded trajectory for the magnitude budget (step 7)")
    ap.add_argument("--paper-budget", action="store_true",
                    help="also run step 7's residual test on the paper's M, V (slow, ~5 min)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    np.set_printoptions(precision=4, suppress=True, linewidth=130)
    rng = np.random.default_rng(args.seed)
    dyn = ArmDynamics(args.urdf, args.phi)
    phi = phi_rigid(args.phi)
    Q = rng.uniform(LIMITS_LO, LIMITS_HI, (args.n, 6))
    V = rng.uniform(-args.vmax, args.vmax, (args.n, 6))
    A = rng.uniform(-args.amax, args.amax, (args.n, 6))

    print(f"URDF: {os.path.relpath(args.urdf, REPO)}")
    print(f"phi:  {os.path.relpath(args.phi, REPO)}")
    print(f"Ia (armature) [kg·m²]: {dyn.ia}")
    print(f"{args.n} random states, q in controller limits, |q̇|≤{args.vmax}, |q̈|≤{args.amax}\n")

    e_M, e_sym, e_skew, e_rec, e_rec_phi, eigs = [], [], [], [], [], []
    for q, v, a in zip(Q, V, A):
        M, C, G = dyn.M(q), dyn.C(q, v), dyn.G(q)
        Mphi = M_from_phi(phi, q)
        e_M.append(rel(M, Mphi))
        e_sym.append(np.max(np.abs(Mphi - Mphi.T)))
        eigs.append(np.linalg.eigvalsh(M))
        Nm = dyn.M_dot(q, v) - 2.0 * C
        e_skew.append(np.max(np.abs(Nm + Nm.T)) / max(np.max(np.abs(dyn.M_dot(q, v))), 1e-12))
        tau = M @ a + C @ v + G
        e_rec.append(np.max(np.abs(tau - dyn.inverse_dynamics(q, v, a))))
        e_rec_phi.append(np.max(np.abs(tau - sf.inverse_dynamics_phi(q, v, a, phi))))
    eigs = np.array(eigs)

    results = []

    def ok(b):
        results.append(bool(b))
        return "PASS" if b else "FAIL"
    print("1. M (Pinocchio CRBA+Ia) vs M from sysid Newton–Euler on φ: "
          f"max rel. err {max(e_M):.1e}  {ok(max(e_M) < 1e-6)}")
    print(f"2. Symmetry of the φ-built M: max |M − Mᵀ| = {max(e_sym):.1e} kg·m²  "
          f"{ok(max(e_sym) < 1e-9)}")
    print(f"3. Positive definiteness: λ_min ∈ [{eigs[:, 0].min():.2e}, {eigs[:, 0].max():.2e}], "
          f"λ_max ∈ [{eigs[:, -1].min():.3f}, {eigs[:, -1].max():.3f}] kg·m², "
          f"cond(M) ≤ {(eigs[:, -1] / eigs[:, 0]).max():.0f}  {ok(eigs[:, 0].min() > 0)}")
    print(f"4. Skew-symmetry of Ṁ − 2C: max |N + Nᵀ| / max|Ṁ| = {max(e_skew):.1e}  "
          f"{ok(max(e_skew) < 1e-5)}")
    print(f"5. M q̈ + C q̇ + G vs RNEA: {max(e_rec):.1e} Nm;  vs sysid φ: "
          f"{max(e_rec_phi):.1e} Nm  {ok(max(e_rec) < 1e-9 and max(e_rec_phi) < 1e-6)}")

    passed = all(results)
    q0 = np.array([0.0, -0.6, 0.5, 0.0, 0.4, 0.0])
    print(f"\n   M(q) at the controller test pose q = {q0} [kg·m²]:\n{dyn.M(q0)}")

    if not args.no_paper:
        paper_section(dyn, Q, V, rng)
    if args.csv:
        budget_section(dyn, args.csv, args.phi, paper=args.paper_budget and not args.no_paper)
    return 0 if passed else 1


def paper_section(dyn, Q, V, rng, n=30):
    import paper_dynamics as pdyn
    if not os.path.isdir(pdyn.PAPER_DIR):
        print(f"\n6. [skip] paper model not found at {pdyn.PAPER_DIR}")
        return
    print("\n6. Paper model (translated from MATLAB; native unit mA) — "
          f"{n} of the states")
    M_p, V_p, G_p = pdyn.load_paper_M(), pdyn.load_paper_V(), pdyn.load_paper_G()
    sys.path.insert(0, pdyn.PAPER_DIR)
    from Gravity_Compensation_Function import calculate_gravity
    idx = rng.choice(len(Q), n, replace=False)

    eg = max(rel(G_p(Q[k]), np.array(calculate_gravity(*Q[k]), float)) for k in idx)
    print(f"   6a translator: G(.m translated) vs authors' Python port: max rel. err {eg:.1e}")

    ev = max(rel(christoffel_V(M_p, Q[k], V[k]), V_p(Q[k], V[k])) for k in idx)
    print(f"   6b paper self-consistency: V_paper vs Christoffel(M_paper)·q̇: max rel. err {ev:.1e}")

    Mo = np.array([dyn.to_mA(dyn.M(Q[k])) for k in idx])           # (n,6,6) mA·s²/rad
    Mp = np.array([M_p(Q[k]) for k in idx])
    Vo = np.array([dyn.to_mA(dyn.C(Q[k], V[k]) @ V[k]) for k in idx])
    Vp = np.array([V_p(Q[k], V[k]) for k in idx])
    print("   6c ours vs paper (ours converted Nm → mA by EFFORT_SCALE, row-wise):")
    print(f"      {'joint':<13}{'M_jj ours':>10}{'M_jj paper':>11}{'ratio p/o':>10}"
          f"{'corr M_jj':>10}{'corr V_j':>9}{'RMS V ours':>11}{'RMS V paper':>12}")
    for j in range(6):
        mo, mp = Mo[:, j, j], Mp[:, j, j]
        c_m = np.corrcoef(mo, mp)[0, 1] if mo.std() > 1e-9 and mp.std() > 1e-9 else np.nan
        c_v = np.corrcoef(Vo[:, j], Vp[:, j])[0, 1]
        print(f"      {J[j]:<13}{mo.mean():>10.1f}{mp.mean():>11.1f}{mp.mean() / mo.mean():>10.2f}"
              f"{c_m:>10.2f}{c_v:>9.2f}{np.sqrt(np.mean(Vo[:, j]**2)):>11.1f}"
              f"{np.sqrt(np.mean(Vp[:, j]**2)):>12.1f}")
    print("      (M_jj = mean diagonal over the states [mA·s²/rad]; corr = Pearson over the "
          "states, scale-free; nan = entry ~constant)")


def budget_section(dyn, csv, phi_path, paper=False):
    print(f"\n7. Torque budget on {os.path.relpath(csv, REPO)} (RMS per joint, Nm)")
    _, q, dq, ddq, tau = sf.load_and_filter(csv, fs=200.0, stride=40, drop_glitches=True)
    Mq = np.array([dyn.M(a) @ b for a, b in zip(q, ddq)])
    Cq = np.array([dyn.C(a, b) @ b for a, b in zip(q, dq)])
    Gq = np.array([dyn.G(a) for a in q])
    rms = lambda x: np.sqrt(np.mean(x**2, axis=0))  # noqa: E731
    print(f"   {'joint':<13}{'M q̈':>8}{'C q̇':>8}{'G':>8}{'measured':>10}{'(Mq̈+Cq̇)/G':>13}")
    for j in range(6):
        g = rms(Gq)[j]
        r = f"{rms(Mq + Cq)[j] / g:>13.2f}" if g > 0.05 else f"{'— (G≈0)':>13}"
        print(f"   {J[j]:<13}{rms(Mq)[j]:>8.3f}{rms(Cq)[j]:>8.3f}{g:>8.3f}{rms(tau)[j]:>10.3f}{r}")
    print(f"   |q̇| RMS {np.round(rms(dq), 2)} rad/s;  |q̈| RMS {np.round(rms(ddq), 2)} rad/s²")

    # Does the model's inertial part EXPLAIN measured torque? Residual after each
    # model's own G + identified friction, without vs with its Mq̈ + Cq̇, and the
    # least-squares scale k on that term (k ≈ 1 ⇒ M, C have the right size).
    # All in mA so ours and the paper share one axis. Nothing is fitted except k.
    meas = tau / sf.EFFORT_SCALE
    phi = np.load(phi_path)
    fr = np.array([phi[i * sf.N_PARAMS + 10:i * sf.N_PARAMS + 13] for i in range(6)])
    F = fr[:, 0] * dq + fr[:, 1] * np.sign(dq) + fr[:, 2]
    rows = [("ours", dyn.to_mA(Gq.T).T, F / sf.EFFORT_SCALE, dyn.to_mA((Mq + Cq).T).T)]
    if paper:
        import paper_dynamics as pdyn
        import compare_paper_model as cpm
        print(f"   (evaluating the paper's 5 MB Coriolis vector on {len(q)} samples — minutes)")
        calc, pFv, pFc, pF0 = cpm.load_paper(pdyn.PAPER_DIR)
        M_p, V_p = pdyn.load_paper_M(), pdyn.load_paper_V()
        rows.append(("paper", np.array([calc(*a) for a in q], float),
                     pFv * dq + pFc * np.sign(dq) + pF0,
                     np.array([M_p(a) @ c + V_p(a, b) for a, b, c in zip(q, dq, ddq)])))
    print(f"\n   Held-out residual |meas − G − F| vs |meas − G − F − (Mq̈+Cq̇)| (RMS, mA; {len(q)} samples)")
    print(f"   {'model':<7}{'':<9}" + "".join(f"{j[:9]:>10}" for j in J))
    for name, G, Fm, MC in rows:
        r0 = meas - G - Fm
        k = (r0 * MC).sum(0) / (MC * MC).sum(0)
        print(f"   {name:<7}{'no M,C':<9}" + "".join(f"{x:>10.0f}" for x in rms(r0)))
        print(f"   {'':<7}{'+ M,C':<9}" + "".join(f"{x:>10.0f}" for x in rms(r0 - MC)))
        print(f"   {'':<7}{'best k':<9}" + "".join(f"{x:>10.2f}" for x in k))


if __name__ == "__main__":
    sys.exit(main())
