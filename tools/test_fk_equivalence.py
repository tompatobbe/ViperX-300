#!/usr/bin/env python3
"""
test_fk_equivalence.py — does sysid_feasible's DH model match the real robot?
=============================================================================

`sysid_feasible.forward_kinematics` (the DH chain the whole regressor is built
on) must describe the SAME physical kinematics as the genuine Interbotix
`urdf/vx300s.urdf`. If it does not, every identified parameter is fit through
wrong joint axes and no regularisation can recover the true dynamics — this is
the gravity-deficit failure (see CHANGELOG 2026-06-13 "MAJOR (CORRECTED): the
DH kinematic model ... does not match the real ViperX-300").

The check must be FAIR: a mere difference in *conventions* between the DH chain
and the URDF (base frame, per-joint zero offset, tool frame, axis sign) must
NOT cause a failure — only a genuine kinematic difference should. So we compare
**intrinsic, convention-free signatures** of the two chains.

CHECK 1 (headline) — consecutive joint-axis angles.
  The angle between joint i's and joint i+1's rotation axes is independent of
  base frame, joint zero offsets, tool frame, and axis sign. For a 6R serial
  arm it is the chain's geometric fingerprint. The real ViperX is
  ⊥,∥,⊥,⊥,⊥ (waist⊥shoulder, shoulder∥elbow [the parallel pitch pair],
  then the wrist). The DH model must reproduce this within tolerance.

CHECK 2 (diagnostic) — per-joint axis directions in the base frame, after the
  single best-fit base rotation that aligns the two axis sets (Kabsch on the
  axis directions). Shows *which* joints are mis-oriented.

NB this validates the rotational STRUCTURE (axes). Link lengths (a, d) and the
absolute reach are NOT fully checked here — once the axes are correct, add a
pose-level check that calibrates the constant base/tool transforms and per-joint
zero offsets and asserts end-effector poses match over random q.

Run:  source /opt/ros/humble/setup.bash && python3 tools/test_fk_equivalence.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pinocchio as pin

from sysid_feasible import forward_kinematics

ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
URDF_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "urdf", "vx300s.urdf")
ANGLE_TOL = 2.0      # deg, consecutive-axis-angle agreement
AXIS_TOL = 2.0       # deg, per-joint axis agreement after base alignment
N_CONFIGS = 5        # axis directions are config-independent; average for robustness
SEED = 0


def _omega_axis(R0, R1, eps):
    """Unit world rotation axis from R0→R1 (a small rotation of magnitude eps)."""
    W = ((R1 - R0) / eps) @ R0.T
    a = np.array([W[2, 1], W[0, 2], W[1, 0]])
    n = np.linalg.norm(a)
    return a / n if n > 1e-9 else a


def dh_joint_axes(q, eps=1e-6):
    R0 = forward_kinematics(q)[6][:3, :3]
    out = []
    for j in range(6):
        dq = np.zeros(6); dq[j] = eps
        out.append(_omega_axis(R0, forward_kinematics(q + dq)[6][:3, :3], eps))
    return np.array(out)


def urdf_joint_axes(q, model, data, iq, jid, eps=1e-6):
    def R(qq):
        qf = pin.neutral(model)
        for i, v in zip(iq, qq):
            qf[i] = v
        pin.forwardKinematics(model, data, qf)
        return data.oMi[jid].rotation.copy()
    R0 = R(q)
    out = []
    for j in range(6):
        dq = np.zeros(6); dq[j] = eps
        out.append(_omega_axis(R0, R(q + dq), eps))
    return np.array(out)


def consec_angles(axes):
    return np.array([np.degrees(np.arccos(abs(np.clip(axes[i] @ axes[i + 1], -1, 1))))
                     for i in range(5)])


def kabsch_rot(P, Q):
    """Best-fit proper rotation mapping directions P→Q (rows are unit vectors)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))])
    return Vt.T @ D @ U.T


def main():
    rng = np.random.default_rng(SEED)
    model = pin.buildModelFromUrdf(URDF_PATH)
    data = model.createData()
    iq = [model.joints[model.getJointId(j)].idx_q for j in ARM_JOINTS]
    jid = model.getJointId("wrist_rotate")

    # A joint's world axis direction is configuration-DEPENDENT, so both models'
    # axes must be evaluated at the SAME q (never average axis vectors across
    # configs — that corrupts the directions). The consecutive-axis ANGLE is
    # config-independent; we average that scalar over several configs for
    # robustness, evaluating DH and URDF at identical q each time.
    qs = [rng.uniform(-0.5, 0.5, 6) for _ in range(N_CONFIGS)]
    dh_ax_per = [dh_joint_axes(q) for q in qs]
    ur_ax_per = [urdf_joint_axes(q, model, data, iq, jid) for q in qs]

    print("=" * 70)
    print("FK equivalence: sysid_feasible DH model  vs  real urdf/vx300s.urdf")
    print("=" * 70)

    # CHECK 1 — intrinsic consecutive-axis-angle signature (scalar, per-config).
    a_dh = np.mean([consec_angles(ax) for ax in dh_ax_per], 0)
    a_ur = np.mean([consec_angles(ax) for ax in ur_ax_per], 0)
    # CHECK 2 uses a single shared config (axis directions matched at same q).
    dh_ax, ur_ax = dh_ax_per[0], ur_ax_per[0]
    print("\n  CHECK 1 — consecutive joint-axis angle [deg] (convention-free):")
    print(f"    {'pair':<30}{'DH':>8}{'URDF':>8}{'Δ':>8}")
    ok1 = True
    for i in range(5):
        d = abs(a_dh[i] - a_ur[i])
        if d > ANGLE_TOL:
            ok1 = False
        pair = f"{ARM_JOINTS[i]}->{ARM_JOINTS[i+1]}"
        print(f"    {pair:<30}{a_dh[i]:>8.1f}{a_ur[i]:>8.1f}{d:>8.1f}"
              f"{'   MISMATCH' if d > ANGLE_TOL else ''}")

    # CHECK 2 — per-joint axis directions after the single best-fit base rotation.
    # Joint axes are undirected lines, so resolve each axis's sign jointly with
    # the base rotation (alternate: fit R, flip each DH axis to agree with R·dh
    # vs ur, refit) — otherwise sign flips corrupt the Kabsch fit.
    signs = np.ones(6)
    for _ in range(10):
        R = kabsch_rot(dh_ax * signs[:, None], ur_ax)
        new = np.sign([(R @ dh_ax[j]) @ ur_ax[j] for j in range(6)])
        new[new == 0] = 1
        if np.array_equal(new, signs):
            break
        signs = new
    dh_ax = dh_ax * signs[:, None]
    print("\n  CHECK 2 — per-joint axis in base frame (DH aligned to URDF):")
    print(f"    {'joint':<14}{'DH→base':>22}{'URDF':>22}{'angle°':>9}")
    ok2 = True
    for j, name in enumerate(ARM_JOINTS):
        a = R @ dh_ax[j]
        ang = np.degrees(np.arccos(abs(np.clip(a @ ur_ax[j], -1, 1))))  # undirected
        if ang > AXIS_TOL:
            ok2 = False
        print(f"    {name:<14}{str(np.round(a,2)):>22}{str(np.round(ur_ax[j],2)):>22}{ang:>9.2f}")

    print("\n" + "-" * 70)
    print(f"  CHECK 1 axis-angle signature : {'PASS' if ok1 else 'FAIL'}  (authoritative)")
    print(f"  CHECK 2 per-joint axes       : diagnostic only — residual angles are")
    print( "           confounded by the two models' differing joint-zero offsets")
    print( "           (same numeric q ⇒ physically different poses), so small")
    print( "           non-zero values here are expected and do NOT indicate a bug.")
    print("  (link lengths a,d are NOT checked here — add a pose-level calibration "
          "check\n   that solves base+tool+joint-zero offsets to validate reach.)")
    print("=" * 70)
    if ok1:
        print("PASS — DH joint-axis structure matches the real robot.")
        sys.exit(0)
    print("FAIL — DH joint-axis structure does NOT match the real robot.")
    sys.exit(1)


if __name__ == "__main__":
    main()
