#!/usr/bin/env python3
"""
test_phi_urdf_consistency.py — phi→URDF export must preserve the dynamics
==========================================================================

The identified phi lives in the DH frames of sysid_feasible's Newton–Euler
regressor. phi_to_urdf (standalone mode) re-expresses it as a URDF. If the
two kinematic descriptions disagree, the exported URDF predicts different
torques than the phi it was built from — a silent export bug (observed
2026-06-13: same phi, zero pose, shoulder gravity −1.49 Nm via the regressor
vs −0.62 Nm via the URDF; see CHANGELOG).

This test draws random phi vectors and random states and requires

    regressor_fast(q, q̇, q̈) @ phi  ==  Pinocchio RNEA(URDF(phi), q, q̇, q̈)

to machine precision (friction entries zeroed — RNEA is rigid-body only).
Needs the real Pinocchio: `source /opt/ros/humble/setup.bash` first.

Run:  python3 tools/test_phi_urdf_consistency.py
"""

import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pinocchio as pin

import phi_to_urdf
from sysid_feasible import regressor_fast, N_JOINTS, N_PARAMS

JOINT_NAMES = phi_to_urdf.JOINT_NAMES
TOL = 1e-6          # Nm — exact up to float round-off accumulated over the RNEA chain
N_STATES = 25
SEED = 0


def random_phi(rng):
    """Random physically-plausible-scale phi (78,), friction terms zeroed."""
    phi = np.zeros(N_JOINTS * N_PARAMS)
    for i in range(N_JOINTS):
        b = i * N_PARAMS
        m = rng.uniform(0.2, 2.0)
        c = rng.uniform(-0.15, 0.15, 3)            # CoM within ±15 cm
        phi[b + 0] = m
        phi[b + 1:b + 4] = m * c
        # J_O = J_com + parallel axis, J_com diagonal-dominant SPD
        A = rng.uniform(-1.0, 1.0, (3, 3))
        J_com = 1e-3 * (A @ A.T + 3.0 * np.eye(3))
        J_O = J_com + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
        phi[b + 4] = J_O[0, 0]; phi[b + 5] = J_O[0, 1]; phi[b + 6] = J_O[0, 2]
        phi[b + 7] = J_O[1, 1]; phi[b + 8] = J_O[1, 2]; phi[b + 9] = J_O[2, 2]
        # Fv, Fc, F0 stay 0: RNEA models rigid-body dynamics only.
    return phi


def urdf_torque_fn(phi):
    """Export phi → standalone URDF → Pinocchio model; return tau(q,dq,ddq)."""
    root = phi_to_urdf.generate_standalone(phi)
    phi_to_urdf._indent(root)
    import xml.etree.ElementTree as ET
    with tempfile.NamedTemporaryFile("w", suffix=".urdf", delete=False) as f:
        f.write('<?xml version="1.0" ?>\n')
        ET.ElementTree(root).write(f, encoding="unicode")
        path = f.name
    try:
        model = pin.buildModelFromUrdf(path)
    finally:
        os.unlink(path)
    data = model.createData()
    iq = [model.joints[model.getJointId(j)].idx_q for j in JOINT_NAMES]
    iv = [model.joints[model.getJointId(j)].idx_v for j in JOINT_NAMES]

    def tau(q, dq, ddq):
        qf = np.zeros(model.nq); vf = np.zeros(model.nv); af = np.zeros(model.nv)
        qf[iq] = q; vf[iv] = dq; af[iv] = ddq
        return np.asarray(pin.rnea(model, data, qf, vf, af))[iv]
    return tau


def main():
    rng = np.random.default_rng(SEED)
    worst = 0.0
    for trial in range(3):
        phi = random_phi(rng)
        tau_urdf = urdf_torque_fn(phi)
        for _ in range(N_STATES):
            q   = rng.uniform(-1.5, 1.5, N_JOINTS)
            dq  = rng.uniform(-2.0, 2.0, N_JOINTS)
            ddq = rng.uniform(-5.0, 5.0, N_JOINTS)
            t_reg  = regressor_fast(q, dq, ddq) @ phi
            t_urdf = tau_urdf(q, dq, ddq)
            worst = max(worst, float(np.max(np.abs(t_reg - t_urdf))))
    print(f"max |tau_regressor − tau_URDF| over {3 * N_STATES} random states: "
          f"{worst:.3e} Nm  (tolerance {TOL:g})")
    if worst > TOL:
        print("FAIL — exported URDF does not reproduce the phi dynamics")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    main()
