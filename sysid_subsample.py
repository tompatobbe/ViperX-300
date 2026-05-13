#!/usr/bin/env python3
"""
sysid_subsample.py — System Identification for ViperX-300 6-DOF Arm
=====================================================================
Same methodology as sysid_paper.py with two fixes applied:
  1. DH bug fixed: d_6 = 0.0 (was L6 = 0.13658)
  2. NE backward pass index corrected: f_ext[i+1] instead of f_ext[i]

Speed/RAM fix: --stride N subsamples the data after filtering.
  --stride 10  → uses every 10th sample (~10x faster, ~10x less RAM)
  --stride 1   → original behaviour (all samples)

pip install numpy scipy matplotlib pandas
"""

import numpy as np
import scipy.linalg as la
import scipy.signal as sig
import scipy.optimize as opt
import matplotlib.pyplot as plt
import datetime
import os
import pandas as pd
from typing import Tuple, List, Optional

# =============================================================================
# 1. DH kinematics
# =============================================================================

L1 = 0.12675
L2 = 0.30594
L3 = 0.21981
L4 = 0.08021
L5 = 0.07000
L6 = 0.13658   # end-effector length (NOT used as d_6 — see fix below)

DH_PARAMS = np.array([
    # alpha_prev   a_prev    d_i          theta_offset
    [0.0,          0.0,      L1,           0.0          ],   # joint 1
    [3*np.pi/2,    0.0,      0.0,         -0.437*np.pi  ],   # joint 2
    [0.0,          L2,       0.0,         -0.063*np.pi  ],   # joint 3
    [3*np.pi/2,    0.0,      L3+L4,        0.0          ],   # joint 4
    [  np.pi/2,    0.0,      0.0,          0.0          ],   # joint 5
    [3*np.pi/2,    0.0,      0.0,          0.0          ],   # joint 6  FIX: d_6=0 per Table 1
], dtype=float)

N_JOINTS   = 6
N_PARAMS   = 13
N_PARAMS_T = N_JOINTS * N_PARAMS   # 78


def _dh_transform(alpha, a, d, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [ 0,     sa,     ca,    d],
        [ 0,      0,      0,    1],
    ])


def forward_kinematics(q):
    T = [np.eye(4)]
    for i in range(N_JOINTS):
        alpha, a, d, theta_off = DH_PARAMS[i]
        T.append(T[-1] @ _dh_transform(alpha, a, d, q[i] + theta_off))
    return T


def _skew(v):
    return np.array([
        [ 0,    -v[2],  v[1]],
        [ v[2],  0,    -v[0]],
        [-v[1],  v[0],  0   ],
    ])


# =============================================================================
# 2. Newton-Euler inverse dynamics (corrected backward pass)
# =============================================================================

def _inertia_at_origin(phi_link):
    Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = phi_link[4:10]
    return np.array([
        [Jxx, Jxy, Jxz],
        [Jxy, Jyy, Jyz],
        [Jxz, Jyz, Jzz],
    ])


def inverse_dynamics_phi(q, dq, ddq, phi):
    T = forward_kinematics(q)
    R = [T[i][:3, :3] for i in range(N_JOINTS + 1)]

    g0 = np.array([0, 0, 9.81])
    omega  = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    domega = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    ddp    = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    ddp[0] = g0

    for i in range(1, N_JOINTS + 1):
        Rp   = R[i-1]; Ri = R[i]
        Rrel = Rp.T @ Ri
        zi_1_loc   = Rrel.T @ np.array([0, 0, 1])
        omega_loc  = Rrel.T @ omega[i-1]
        domega_loc = Rrel.T @ domega[i-1]
        omega[i]   = omega_loc + dq[i-1] * zi_1_loc
        domega[i]  = domega_loc + ddq[i-1] * zi_1_loc + np.cross(omega_loc, dq[i-1]*zi_1_loc)
        r_loc = Ri.T @ (T[i][:3, 3] - T[i-1][:3, 3])
        ddp[i] = Ri.T @ (Rp @ ddp[i-1]) + np.cross(domega[i], r_loc) + np.cross(omega[i], np.cross(omega[i], r_loc))

    f_ext = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    n_ext = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    tau   = np.zeros(N_JOINTS)

    for i in range(N_JOINTS, 0, -1):
        idx   = i - 1
        phi_i = phi[idx * N_PARAMS : (idx + 1) * N_PARAMS]
        mi    = phi_i[0]
        mci   = phi_i[1:4]
        Ji    = _inertia_at_origin(phi_i)
        Fv_i  = phi_i[10]; Fc_i = phi_i[11]; F0_i = phi_i[12]

        fi = mi * ddp[i] + np.cross(domega[i], mci) + np.cross(omega[i], np.cross(omega[i], mci))
        ni = Ji @ domega[i] + np.cross(omega[i], Ji @ omega[i]) + np.cross(mci, ddp[i])

        if i < N_JOINTS:
            Rrel_next = R[i].T @ R[i+1]
            r_next    = R[i].T @ (T[i+1][:3, 3] - T[i][:3, 3])
            # FIX: use f_ext[i+1] / n_ext[i+1] (set in previous iteration)
            f_next = Rrel_next @ f_ext[i+1]
            n_next = Rrel_next @ n_ext[i+1] + np.cross(r_next, f_next)
        else:
            f_next = np.zeros(3)
            n_next = np.zeros(3)

        f_ext[i] = fi + f_next
        n_ext[i] = ni + n_next

        tau[idx] = n_ext[i][2] + Fv_i * dq[idx] + Fc_i * np.sign(dq[idx]) + F0_i

    return tau


def regressor(q, dq, ddq):
    W = np.zeros((N_JOINTS, N_PARAMS_T))
    e = np.zeros(N_PARAMS_T)
    for k in range(N_PARAMS_T):
        e[:] = 0.0; e[k] = 1.0
        W[:, k] = inverse_dynamics_phi(q, dq, ddq, e)
    return W


# =============================================================================
# 3. Base parameter reduction
# =============================================================================

def find_base_parameters(W_stacked, tol=1e-10):
    Q, R_mat, P = la.qr(W_stacked, pivoting=True, mode='economic')
    rank = np.sum(np.abs(np.diag(R_mat)) > tol * np.abs(R_mat[0, 0]))
    base_cols = np.sort(P[:rank])
    W_base = W_stacked[:, base_cols]
    L = np.linalg.lstsq(W_base, W_stacked, rcond=None)[0].T
    return base_cols, L


# =============================================================================
# 4. Data loading + zero-phase Butterworth LPF
# =============================================================================

STALL_TORQUE = np.array([4.1, 10.6, 10.6, 4.1, 4.1, 4.1])
EFFORT_SCALE = STALL_TORQUE / 100.0
ARM_JOINTS   = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def load_and_filter(csv_path, fs=50.0, fc=10.0, stride=1):
    df = pd.read_csv(csv_path)
    t  = df["time"].values.astype(float); t -= t[0]

    pos_cols    = [f"{j}_pos"    for j in ARM_JOINTS]
    effort_cols = [f"{j}_effort" for j in ARM_JOINTS]
    q_raw   = df[pos_cols].values.astype(float)
    tau_raw = df[effort_cols].values.astype(float) * EFFORT_SCALE

    fs_actual = 1.0 / float(np.median(np.diff(t)))
    if abs(fs_actual - fs) > 2.0:
        print(f"    [warn] --fs={fs:.1f} Hz but CSV is {fs_actual:.1f} Hz")
    fs = fs_actual

    nyq  = fs / 2.0
    Wn   = min(fc / nyq, 0.99)
    b, a = sig.butter(4, Wn, btype='low')

    q_filt   = np.vstack([sig.filtfilt(b, a, q_raw[:,j])   for j in range(N_JOINTS)]).T
    tau_filt = np.vstack([sig.filtfilt(b, a, tau_raw[:,j]) for j in range(N_JOINTS)]).T

    dt  = np.gradient(t)
    dq  = np.gradient(q_filt, axis=0) / dt[:, np.newaxis]
    dq  = np.vstack([sig.filtfilt(b, a, dq[:,j]) for j in range(N_JOINTS)]).T
    ddq = np.gradient(dq, axis=0) / dt[:, np.newaxis]
    ddq = np.vstack([sig.filtfilt(b, a, ddq[:,j]) for j in range(N_JOINTS)]).T

    # Subsample after filtering (stride=1 → no change)
    if stride > 1:
        t, q_filt, dq, ddq, tau_filt = (
            t[::stride], q_filt[::stride], dq[::stride], ddq[::stride], tau_filt[::stride]
        )

    return t, q_filt, dq, ddq, tau_filt


# =============================================================================
# 5. Normalisation
# =============================================================================

def normalize_rows(W, tau, tau_max=None):
    if tau_max is None:
        tau_max = np.maximum(np.max(np.abs(tau), axis=0), 1e-6)
    scale = 1.0 / tau_max
    return W * scale[np.newaxis, :], tau * scale[np.newaxis, :], tau_max


# =============================================================================
# 6. Feasibility constraints
# =============================================================================

def pseudo_inertia(phi_link):
    m, mcx, mcy, mcz, Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = phi_link[:10]
    return np.array([
        [(Jyy+Jzz-Jxx)/2,  -Jxy,             -Jxz,            mcx],
        [-Jxy,              (Jxx+Jzz-Jyy)/2,  -Jyz,            mcy],
        [-Jxz,              -Jyz,              (Jxx+Jyy-Jzz)/2, mcz],
        [mcx,                mcy,               mcz,             m  ],
    ])


def feasibility_constraints(phi):
    constraints = []
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        constraints.append({'type': 'ineq', 'fun': lambda phi, j=idx: phi[j] - 1e-4})

        def min_eig(phi, idx=idx):
            return np.min(np.linalg.eigvalsh(pseudo_inertia(phi[idx:idx+N_PARAMS]))) - 1e-6
        constraints.append({'type': 'ineq', 'fun': min_eig})
        constraints.append({'type': 'ineq', 'fun': lambda phi, j=idx+10: phi[j]})
        constraints.append({'type': 'ineq', 'fun': lambda phi, j=idx+11: phi[j]})
    return constraints


# =============================================================================
# 7. Optimisation
# =============================================================================

def initial_phi_guess():
    Fv = [0.126, 0.101, 0.0403, 0.0234, 0.0166, 0.00158]
    Fc = [0.239, 0.412, 0.261,  0.0303, 0.0608, 0.00295]
    F0 = [-0.0529, -0.203, -0.766, 0.0119, -0.0127, -0.00469]
    masses = [0.8, 0.8, 0.5, 0.3, 0.2, 0.1]
    phi0 = np.zeros(N_PARAMS_T)
    for i in range(N_JOINTS):
        idx = i * N_PARAMS; m = masses[i]
        phi0[idx+0]  = m
        phi0[idx+4]  = m * 0.01
        phi0[idx+7]  = m * 0.01
        phi0[idx+9]  = m * 0.01
        phi0[idx+10] = Fv[i]; phi0[idx+11] = Fc[i]; phi0[idx+12] = F0[i]
    return phi0


def identify(W_stacked, tau_stacked, phi0=None, w1=1.0, w2=1e-3, verbose=True):
    if phi0 is None:
        phi0 = initial_phi_guess()

    def objective(phi):
        r = W_stacked @ phi - tau_stacked
        return w1 * r @ r + w2 * (phi - phi0) @ (phi - phi0)

    def gradient(phi):
        r = W_stacked @ phi - tau_stacked
        return 2*w1 * W_stacked.T @ r + 2*w2 * (phi - phi0)

    if verbose:
        print(f"Starting SLSQP — {N_PARAMS_T} params, {W_stacked.shape[0]} equations...")

    result = opt.minimize(objective, phi0, jac=gradient, method='SLSQP',
                          constraints=feasibility_constraints(phi0),
                          options={'maxiter': 2000, 'ftol': 1e-9, 'disp': verbose})
    if verbose:
        print(f"Status: {result.message}  cost={result.fun:.6f}")
    return result.x


# =============================================================================
# 8. REL metric
# =============================================================================

def rel_metric(tau_true, tau_pred):
    denom = np.maximum(np.abs(tau_true), np.abs(tau_pred))
    denom = np.where(denom < 1e-6, 1e-6, denom)
    return np.mean(np.abs(tau_true - tau_pred) / denom, axis=0)


# =============================================================================
# 9. Full pipeline
# =============================================================================

def run_identification(csv_path, fs=50.0, fc_lpf=10.0, stride=10,
                       verbose=True, plot=True):
    print("=" * 60)
    print(f"ViperX-300 SysID — Subsampled (stride={stride})")
    print("=" * 60)

    print("\n[1] Loading and filtering data...")
    t, q, dq, ddq, tau_meas = load_and_filter(csv_path, fs=fs, fc=fc_lpf, stride=stride)
    N = len(t)
    print(f"    Samples after stride: {N},  duration: {t[-1]-t[0]:.2f} s")

    print("\n[2] Building stacked regressor W...")
    W_list = []; tau_list = []
    for n in range(N):
        if verbose and n % max(1, N//10) == 0:
            print(f"    sample {n}/{N}", end="\r")
        W_list.append(regressor(q[n], dq[n], ddq[n]))
        tau_list.append(tau_meas[n])
    print()

    W_full   = np.stack(W_list).reshape(N * N_JOINTS, N_PARAMS_T)
    tau_full = np.stack(tau_list).reshape(N * N_JOINTS)
    print(f"    W shape: {W_full.shape}")

    print("\n[3] Normalising...")
    tau_max  = np.maximum(np.max(np.abs(tau_meas), axis=0), 1e-3)
    scale_vec = np.tile(1.0 / tau_max, N)
    W_norm   = W_full   * scale_vec[:, np.newaxis]
    tau_norm = tau_full * scale_vec
    print(f"    τ_max per joint [Nm]: {tau_max.round(3)}")

    print("\n[4] Finding base parameters via QR...")
    base_cols, L_mat = find_base_parameters(W_norm)
    print(f"    Full: {N_PARAMS_T},  Base: {len(base_cols)}")

    phi_ls, *_ = np.linalg.lstsq(W_norm, tau_norm, rcond=None)
    rel_ls = rel_metric(tau_meas, (W_full @ phi_ls).reshape(N, N_JOINTS))
    print(f"\n    Unconstrained lstsq REL: {rel_ls.round(4)}  mean={rel_ls.mean():.4f}")

    print("\n[5] Constrained identification (SLSQP)...")
    phi_id = identify(W_norm, tau_norm, phi0=initial_phi_guess(), verbose=verbose)

    tau_pred_id = (W_full @ phi_id).reshape(N, N_JOINTS)
    rel_id = rel_metric(tau_meas, tau_pred_id)
    print(f"\n[6] Constrained REL: {rel_id.round(4)}  mean={rel_id.mean():.4f}")

    print("\n[7] Identified friction parameters:")
    print(f"{'Joint':<12} {'Fv':>12} {'Fc':>10} {'F0':>10}")
    for i in range(N_JOINTS):
        idx = i * N_PARAMS
        print(f"  {ARM_JOINTS[i]:<10} {phi_id[idx+10]:>12.4f} {phi_id[idx+11]:>10.4f} {phi_id[idx+12]:>10.4f}")

    print("\n[8] Feasibility check:")
    for i in range(N_JOINTS):
        idx = i * N_PARAMS; pl = phi_id[idx:idx+N_PARAMS]
        eigs = np.linalg.eigvalsh(pseudo_inertia(pl))
        ok = pl[0] > 0 and np.all(eigs >= -1e-4) and pl[10] >= 0 and pl[11] >= 0
        print(f"  Link {i+1} ({ARM_JOINTS[i]}): m={pl[0]:.4f} min_eig={eigs.min():.4f} "
              f"Fv={pl[10]:.4f} Fc={pl[11]:.4f}  [{'OK' if ok else 'FAIL'}]")

    if plot:
        _plot_results(t, tau_meas, tau_pred_id, rel_id)
    return phi_id


def _plot_results(t, tau_meas, tau_pred, rel):
    fig, axes = plt.subplots(N_JOINTS, 1, figsize=(12, 10), sharex=True)
    for j, ax in enumerate(axes):
        ax.plot(t, tau_meas[:, j], label='Measured', lw=0.8)
        ax.plot(t, tau_pred[:, j], '--', label='Predicted', lw=0.8)
        ax.set_ylabel(f"τ_{j+1} [Nm]")
        ax.set_title(f"{ARM_JOINTS[j]}  REL={rel[j]:.3f}", fontsize=9)
        ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Time [s]")
    plt.suptitle("ViperX-300 SysID — Subsampled", fontsize=11)
    plt.tight_layout(); plt.show()


# =============================================================================
# Entry point
# =============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ViperX-300 sysid — subsample version")
    parser.add_argument("csv", nargs="?", default="data/sysid_run1.csv")
    parser.add_argument("--fs",     type=float, default=50.0,  help="Sample rate Hz")
    parser.add_argument("--fc",     type=float, default=10.0,  help="LPF cutoff Hz")
    parser.add_argument("--stride", type=int,   default=10,    help="Keep every N-th sample after filtering (default 10)")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    phi = run_identification(args.csv, fs=args.fs, fc_lpf=args.fc,
                             stride=args.stride, plot=not args.no_plot)
    os.makedirs("npy", exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"npy/phi_subsample_{ts}.npy"
    np.save(out_path, phi)
    print(f"\nSaved to {out_path}")
