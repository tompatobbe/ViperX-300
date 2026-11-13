#!/usr/bin/env python3
"""Explicit rigid-body dynamics matrices of the identified ViperX-300 model.

    τ = M(q)q̈ + C(q,q̇)q̇ + G(q)            (+ friction, not included here)

The controller so far used only G(q) (PD + gravity compensation). This module
exposes the FULL identified model as explicit matrices, for analysis (thesis:
symmetry / positive definiteness / skew-symmetry of Ṁ − 2C, comparison with
the paper's M and V) and for model-based control (computed-torque / inverse-
dynamics feed-forward).

Source of truth is the identified URDF (via Pinocchio), plus the per-joint
reflected motor inertia Ia from the (84,) φ vector:
  - M(q)   = CRBA(URDF, q) + diag(Ia)       Ia goes into `model.armature`, so
                                             Pinocchio's CRBA/RNEA include it.
  - C(q,q̇) = Pinocchio's computeCoriolisMatrix — the Christoffel-symbol
             factorisation, i.e. the (unique-up-to-skew) choice for which
             Ṁ − 2C is skew-symmetric. Ia is constant → contributes nothing.
  - G(q)   = computeGeneralizedGravity.

A URDF cannot carry Ia (it is local to the joint axis, not a link inertia —
see phi_to_urdf.py), which is why it is read from φ and set as armature.

Units: SI (Nm, kg·m²). `to_mA(...)` converts torque rows to Dynamixel master-
motor mA with the SAME EFFORT_SCALE used in identification (dual-motor
shoulder/elbow), for comparison with measured current and the paper's model.

Needs the real Pinocchio: `source /opt/ros/humble/setup.bash`.

Usage:
    from control.dynamics import ArmDynamics
    dyn = ArmDynamics()                 # champion URDF + champion φ (Ia)
    M = dyn.M(q); C = dyn.C(q, dq); G = dyn.G(q)
    tau = dyn.inverse_dynamics(q, dq, ddq)    # == M ddq + C dq + G
"""
import os
import sys

import numpy as np
import pinocchio as pin

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import sysid_feasible as sf  # noqa: E402  (constants only; pure numpy)

ARM_JOINTS = sf.ARM_JOINTS
# Champion (2026-06-24) — same defaults as control/pd_grav_control.py.
DEFAULT_PHI = os.path.join(
    REPO, "outputs/npy/traj_run_200hz_20260624_124955__sysid_feasible-v1-5__cfg-27904c2e.npy")
DEFAULT_URDF = os.path.join(
    REPO, "outputs/urdf/traj_run_200hz_20260624_124955__sysid_feasible-v1-5__"
          "cfg-27904c2e__phi_to_urdf-v1-1__cfg-451881cf.urdf")


def load_ia(phi_path):
    """Reflected motor inertia Ia (6,) [kg·m²] from an (84,) φ; zeros for (78,)."""
    phi = np.load(phi_path)
    if len(phi) == sf.N_PARAMS_T + sf.N_JOINTS:
        return phi[sf.N_PARAMS_T:].copy()
    if len(phi) == sf.N_PARAMS_T:
        return np.zeros(sf.N_JOINTS)
    raise ValueError(f"unexpected φ length {len(phi)} in {phi_path}")


class ArmDynamics:
    """M(q), C(q,q̇), G(q) of the identified arm (joint order = ARM_JOINTS)."""

    def __init__(self, urdf=DEFAULT_URDF, phi=DEFAULT_PHI, ia=None):
        self.model = pin.buildModelFromUrdf(urdf)
        # Match joints BY NAME (CLAUDE.md); the identified URDF is arm-only nq=6.
        names = [self.model.names[i] for i in range(1, self.model.njoints)]
        if self.model.nq != 6 or names != ARM_JOINTS:
            raise ValueError(f"expected arm-only URDF with joints {ARM_JOINTS}, "
                             f"got nq={self.model.nq} {names}")
        self.ia = np.asarray(ia, float) if ia is not None else (
            load_ia(phi) if phi else np.zeros(6))
        self.model.armature = self.ia.copy()
        self.data = self.model.createData()

    # --- the three matrices ----------------------------------------------------
    def M(self, q):
        """Joint-space inertia matrix (6,6) [kg·m²], incl. diag(Ia). Symmetric PD."""
        Mu = pin.crba(self.model, self.data, np.asarray(q, float))
        Mu = np.triu(Mu)                      # CRBA is defined on the upper triangle
        return Mu + np.triu(Mu, 1).T

    def C(self, q, dq):
        """Coriolis/centrifugal matrix (6,6) [kg·m²/s]; C(q,q̇)q̇ is the velocity torque."""
        return pin.computeCoriolisMatrix(self.model, self.data,
                                         np.asarray(q, float), np.asarray(dq, float)).copy()

    def G(self, q):
        """Gravity torque (6,) [Nm]."""
        return pin.computeGeneralizedGravity(self.model, self.data, np.asarray(q, float)).copy()

    # --- convenience -------------------------------------------------------------
    def inverse_dynamics(self, q, dq, ddq):
        """RNEA τ (6,) [Nm] — rigid body + Ia, no friction. == M q̈ + C q̇ + G."""
        return pin.rnea(self.model, self.data, np.asarray(q, float),
                        np.asarray(dq, float), np.asarray(ddq, float)).copy()

    def M_dot(self, q, dq, eps=1e-6):
        """Ṁ(q,q̇) = dM/dt along q̇, by central difference [kg·m²/s]."""
        q, dq = np.asarray(q, float), np.asarray(dq, float)
        return (self.M(q + eps * dq) - self.M(q - eps * dq)) / (2.0 * eps)

    @staticmethod
    def to_mA(x):
        """Convert torque-row quantities (vector or matrix rows) Nm → master-motor mA."""
        x = np.asarray(x, float)
        return x / (sf.EFFORT_SCALE if x.ndim == 1 else sf.EFFORT_SCALE[:, None])
