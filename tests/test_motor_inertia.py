"""
tests/test_motor_inertia.py
Run with:  pytest tests/test_motor_inertia.py -v

Correctness of the per-joint reflected motor-inertia extension
(sysid_feasible --motor-inertia): the linear regressor and the scalar
Newton–Euler inverse dynamics must agree, W(q,q̇,q̈) @ phi == τ(q,q̇,q̈,phi),
both for the plain 78-parameter model and the 84-parameter (+Ia) model.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from sysid_feasible import (
    N_JOINTS, N_PARAMS_T, regressor_fast, inverse_dynamics_phi,
)

RNG = np.random.default_rng(42)


def _random_state():
    q   = RNG.uniform(-1.5, 1.5, N_JOINTS)
    dq  = RNG.uniform(-2.0, 2.0, N_JOINTS)
    ddq = RNG.uniform(-5.0, 5.0, N_JOINTS)
    return q, dq, ddq


def _random_phi(n):
    return RNG.uniform(-0.5, 0.5, n)


def test_regressor_matches_newton_euler_78():
    for _ in range(5):
        q, dq, ddq = _random_state()
        phi = _random_phi(N_PARAMS_T)
        W = regressor_fast(q, dq, ddq)
        assert W.shape == (N_JOINTS, N_PARAMS_T)
        np.testing.assert_allclose(
            W @ phi, inverse_dynamics_phi(q, dq, ddq, phi),
            rtol=1e-9, atol=1e-12)


def test_regressor_matches_newton_euler_84_with_ia():
    for _ in range(5):
        q, dq, ddq = _random_state()
        phi = _random_phi(N_PARAMS_T + N_JOINTS)
        phi[N_PARAMS_T:] = np.abs(phi[N_PARAMS_T:])   # Ia >= 0 like the SDP
        W = regressor_fast(q, dq, ddq, motor_inertia=True)
        assert W.shape == (N_JOINTS, N_PARAMS_T + N_JOINTS)
        np.testing.assert_allclose(
            W @ phi, inverse_dynamics_phi(q, dq, ddq, phi),
            rtol=1e-9, atol=1e-12)


def test_ia_block_is_diagonal_qdd():
    q, dq, ddq = _random_state()
    W = regressor_fast(q, dq, ddq, motor_inertia=True)
    np.testing.assert_allclose(W[:, N_PARAMS_T:], np.diag(ddq))


def test_ia_columns_do_not_change_link_columns():
    q, dq, ddq = _random_state()
    W_plain = regressor_fast(q, dq, ddq)
    W_ia = regressor_fast(q, dq, ddq, motor_inertia=True)
    np.testing.assert_array_equal(W_ia[:, :N_PARAMS_T], W_plain)
