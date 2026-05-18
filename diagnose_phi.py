#!/usr/bin/env python3
"""
diagnose_phi.py — Diagnostic analysis of an identified phi.npy
===============================================================
Loads a phi vector and reports:
  • Per-link parameter table with physical sanity flags
  • Bar charts: mass, CoM position, inertia eigenvalues
  • (optional) Torque fit: predicted vs measured per joint
  • Data quality stats: excitation range, velocity RMS

Usage:
  python diagnose_phi.py npy/phi_feasible_20260517_023032.npy
  python diagnose_phi.py npy/phi_feasible_20260517_023032.npy --csv data/sysid_run1.csv
  python diagnose_phi.py npy/phi_*.npy --compare          # overlay multiple npy files
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Copy of DH/regressor from sysid_feasible.py ──────────────────────────────

L1 = 0.12675
L2 = 0.30594
L3 = 0.21981
L4 = 0.08021
L6 = 0.13658

DH_PARAMS = np.array([
    [0.0,          0.0,      L1,           0.0          ],
    [3*np.pi/2,    0.0,      0.0,         -0.437*np.pi  ],
    [0.0,          L2,       0.0,         -0.063*np.pi  ],
    [3*np.pi/2,    0.0,      L3+L4,        0.0          ],
    [  np.pi/2,    0.0,      0.0,          0.0          ],
    [3*np.pi/2,    0.0,      0.0,          0.0          ],
], dtype=float)

N_JOINTS   = 6
N_PARAMS   = 13

JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
LINK_NAMES  = ["waist_link", "upper_arm_link", "forearm_link",
               "wrist_link", "gripper_link", "ee_arm"]
COLORS      = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]

# Rough expected mass ranges [kg] for each link (from datasheet / prior work)
MASS_EXPECTED = [(0.4, 2.0), (0.5, 2.0), (0.3, 1.5),
                 (0.05, 0.8), (0.05, 0.5), (0.02, 0.3)]

STALL_TORQUE = np.array([4.1, 10.6, 10.6, 4.1, 4.1, 4.1])
EFFORT_SCALE = STALL_TORQUE / 100.0
ARM_JOINTS   = JOINT_NAMES

# URDF links driven by each joint (in phi order)
_URDF_LINKS = [
    "/shoulder_link",       # waist
    "/upper_arm_link",      # shoulder
    "/upper_forearm_link",  # elbow
    "/lower_forearm_link",  # forearm_roll
    "/wrist_link",          # wrist_angle
    "/gripper_link",        # wrist_rotate
]


# ── URDF reference loader ─────────────────────────────────────────────────────

def _rpy_to_R(r, p, y):
    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr,  cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def load_urdf_phi(urdf_path):
    import xml.etree.ElementTree as ET
    root = ET.parse(urdf_path).getroot()

    link_data = {}
    for lel in root.findall("link"):
        name = lel.get("name")
        inertial = lel.find("inertial")
        if inertial is None:
            continue
        mass_el    = inertial.find("mass")
        inertia_el = inertial.find("inertia")
        if mass_el is None or inertia_el is None:
            continue
        origin = inertial.find("origin")
        if origin is not None:
            xyz = [float(v) for v in origin.get("xyz", "0 0 0").split()]
            rpy = [float(v) for v in origin.get("rpy", "0 0 0").split()]
        else:
            xyz, rpy = [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
        link_data[name] = dict(
            m   = float(mass_el.get("value")),
            c   = np.array(xyz),
            R   = _rpy_to_R(*rpy),
            ixx = float(inertia_el.get("ixx", 0)),
            ixy = float(inertia_el.get("ixy", 0)),
            ixz = float(inertia_el.get("ixz", 0)),
            iyy = float(inertia_el.get("iyy", 0)),
            iyz = float(inertia_el.get("iyz", 0)),
            izz = float(inertia_el.get("izz", 0)),
        )

    phi = np.zeros(N_JOINTS * N_PARAMS)
    for i, lname in enumerate(_URDF_LINKS):
        if lname not in link_data:
            print(f"[warn] URDF link {lname!r} not found", file=sys.stderr)
            continue
        d = link_data[lname]
        m, c, R = d["m"], d["c"], d["R"]
        # inertia at CoM expressed in the link frame (undo inertial-frame rotation)
        J_ci = np.array([[d["ixx"], d["ixy"], d["ixz"]],
                         [d["ixy"], d["iyy"], d["iyz"]],
                         [d["ixz"], d["iyz"], d["izz"]]])
        J_c = R @ J_ci @ R.T
        # parallel-axis shift to link origin
        J_O = J_c + m * (np.dot(c, c) * np.eye(3) - np.outer(c, c))
        base = i * N_PARAMS
        phi[base]          = m
        phi[base+1:base+4] = m * c
        phi[base+4] = J_O[0, 0]; phi[base+5] = J_O[0, 1]; phi[base+6] = J_O[0, 2]
        phi[base+7] = J_O[1, 1]; phi[base+8] = J_O[1, 2]; phi[base+9] = J_O[2, 2]
    return phi


# ── Kinematics ────────────────────────────────────────────────────────────────

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


def _ne_forward_pass(q, dq, ddq, T, R):
    g0 = np.array([0.0, 0.0, 9.81])
    omega  = [np.zeros(3)] * (N_JOINTS + 1)
    domega = [np.zeros(3)] * (N_JOINTS + 1)
    ddp    = [np.zeros(3)] * (N_JOINTS + 1)
    omega  = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    domega = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    ddp    = [np.zeros(3) for _ in range(N_JOINTS + 1)]
    ddp[0] = g0
    for i in range(1, N_JOINTS + 1):
        Rp = R[i-1]; Ri = R[i]
        Rrel = Rp.T @ Ri
        zi_1_loc   = Rrel.T @ np.array([0.0, 0.0, 1.0])
        omega_loc  = Rrel.T @ omega[i-1]
        domega_loc = Rrel.T @ domega[i-1]
        omega[i]   = omega_loc + dq[i-1] * zi_1_loc
        domega[i]  = domega_loc + ddq[i-1] * zi_1_loc + np.cross(omega_loc, dq[i-1] * zi_1_loc)
        r_loc = Ri.T @ (T[i][:3, 3] - T[i-1][:3, 3])
        ddp[i] = (Ri.T @ (Rp @ ddp[i-1])
                  + np.cross(domega[i], r_loc)
                  + np.cross(omega[i], np.cross(omega[i], r_loc)))
    return omega, domega, ddp


def regressor_fast(q, dq, ddq):
    T = forward_kinematics(q)
    R = [T[i][:3, :3] for i in range(N_JOINTS + 1)]
    omega, domega, ddp = _ne_forward_pass(q, dq, ddq, T, R)
    P = N_JOINTS * N_PARAMS
    f_mat = np.zeros((3, P)); n_mat = np.zeros((3, P)); W = np.zeros((N_JOINTS, P))
    EX = np.array([1., 0., 0.]); EY = np.array([0., 1., 0.]); EZ = np.array([0., 0., 1.])
    for i in range(N_JOINTS, 0, -1):
        li = i - 1; base = li * N_PARAMS
        ω = omega[i]; dω = domega[i]; d = ddp[i]
        fi_cols = np.zeros((3, N_PARAMS))
        fi_cols[:, 0] = d
        for k, e in enumerate((EX, EY, EZ)):
            fi_cols[:, 1+k] = np.cross(dω, e) + np.cross(ω, np.cross(ω, e))
        ni_cols = np.zeros((3, N_PARAMS))
        for k, e in enumerate((EX, EY, EZ)):
            ni_cols[:, 1+k] = np.cross(e, d)
        Jdw = np.array([dω[0], 0.,    0.   ]); Jw = np.array([ω[0], 0.,   0.  ])
        ni_cols[:, 4] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([dω[1], dω[0], 0.   ]); Jw = np.array([ω[1], ω[0], 0.  ])
        ni_cols[:, 5] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([dω[2], 0.,    dω[0]]); Jw = np.array([ω[2], 0.,   ω[0]])
        ni_cols[:, 6] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([0.,    dω[1], 0.   ]); Jw = np.array([0.,   ω[1], 0.  ])
        ni_cols[:, 7] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([0.,    dω[2], dω[1]]); Jw = np.array([0.,   ω[2], ω[1]])
        ni_cols[:, 8] = Jdw + np.cross(ω, Jw)
        Jdw = np.array([0.,    0.,    dω[2]]); Jw = np.array([0.,   0.,   ω[2]])
        ni_cols[:, 9] = Jdw + np.cross(ω, Jw)
        fi_full = np.zeros((3, P)); fi_full[:, base:base+N_PARAMS] = fi_cols
        ni_full = np.zeros((3, P)); ni_full[:, base:base+N_PARAMS] = ni_cols
        if i < N_JOINTS:
            Rrel_next = R[i].T @ R[i+1]
            r_next    = R[i].T @ (T[i+1][:3, 3] - T[i][:3, 3])
            f_from_next = Rrel_next @ f_mat
            n_from_next = Rrel_next @ n_mat + np.cross(r_next, f_from_next.T).T
        else:
            f_from_next = np.zeros((3, P)); n_from_next = np.zeros((3, P))
        f_mat = fi_full + f_from_next
        n_mat = ni_full + n_from_next
        W[li, :] = n_mat[2, :]
        W[li, base+10] += dq[li]
        W[li, base+11] += np.sign(dq[li])
        W[li, base+12] += 1.0
    return W


# ── Parameter parsing / feasibility ──────────────────────────────────────────

def parse_link(phi, i):
    base = i * N_PARAMS
    p = phi[base:base+N_PARAMS]
    m   = float(p[0])
    mc  = p[1:4].astype(float)
    J_O = np.array([[p[4], p[5], p[6]],
                    [p[5], p[7], p[8]],
                    [p[6], p[8], p[9]]], dtype=float)
    Fv, Fc, F0 = float(p[10]), float(p[11]), float(p[12])
    c_com = mc / m if abs(m) > 1e-12 else mc * 0
    J_com = J_O - m * (np.dot(c_com, c_com) * np.eye(3) - np.outer(c_com, c_com))
    return m, mc, c_com, J_O, J_com, Fv, Fc, F0


def pseudo_inertia(phi_link):
    m, mcx, mcy, mcz, Jxx, Jxy, Jxz, Jyy, Jyz, Jzz = phi_link[:10]
    return np.array([
        [(Jyy+Jzz-Jxx)/2, -Jxy,            -Jxz,            mcx],
        [-Jxy,             (Jxx+Jzz-Jyy)/2, -Jyz,            mcy],
        [-Jxz,             -Jyz,             (Jxx+Jyy-Jzz)/2, mcz],
        [mcx,               mcy,              mcz,             m  ],
    ])


# ── Data loading (mirrors sysid_feasible.py) ──────────────────────────────────

def load_and_filter(csv_path, fs=50.0, fc=10.0, stride=1):
    import pandas as pd
    import scipy.signal as sig

    df  = pd.read_csv(csv_path)
    t   = df["time"].values.astype(float); t -= t[0]
    q_raw   = df[[f"{j}_pos"    for j in ARM_JOINTS]].values.astype(float)
    tau_raw = df[[f"{j}_effort" for j in ARM_JOINTS]].values.astype(float) * EFFORT_SCALE

    fs_actual = 1.0 / float(np.median(np.diff(t)))
    nyq  = fs_actual / 2.0
    Wn   = min(fc / nyq, 0.99)
    b, a = sig.butter(4, Wn, btype='low')

    q_filt   = np.vstack([sig.filtfilt(b, a, q_raw[:,j])   for j in range(N_JOINTS)]).T
    tau_filt = np.vstack([sig.filtfilt(b, a, tau_raw[:,j]) for j in range(N_JOINTS)]).T
    dt  = np.gradient(t)
    dq  = np.gradient(q_filt, axis=0) / dt[:, np.newaxis]
    dq  = np.vstack([sig.filtfilt(b, a, dq[:,j]) for j in range(N_JOINTS)]).T
    ddq = np.gradient(dq, axis=0) / dt[:, np.newaxis]
    ddq = np.vstack([sig.filtfilt(b, a, ddq[:,j]) for j in range(N_JOINTS)]).T

    if stride > 1:
        t, q_filt, dq, ddq, tau_filt = (
            t[::stride], q_filt[::stride], dq[::stride],
            ddq[::stride], tau_filt[::stride])
    return t, q_filt, dq, ddq, tau_filt


# ── Console report ────────────────────────────────────────────────────────────

def print_report(phi, label=""):
    hdr = f" {label}" if label else ""
    print(f"\n{'='*70}")
    print(f"PHI DIAGNOSTIC REPORT{hdr}")
    print(f"{'='*70}")
    print(f"\n{'Link':<16} {'m[kg]':>8} {'cx[m]':>8} {'cy[m]':>8} {'cz[m]':>8}"
          f" {'Fv':>8} {'Fc':>8} {'F0':>8}  flags")
    print("-" * 95)
    for i in range(N_JOINTS):
        m, mc, c, J_O, J_com, Fv, Fc, F0 = parse_link(phi, i)
        flags = []
        lo, hi = MASS_EXPECTED[i]
        if m < lo:  flags.append(f"MASS_LOW({m:.3f}<{lo})")
        if m > hi:  flags.append(f"MASS_HIGH({m:.3f}>{hi})")
        eigs_com = np.linalg.eigvalsh(J_com)
        if np.any(eigs_com < -1e-6): flags.append(f"JCOM_NEG({eigs_com.min():.2e})")
        eigs_pi = np.linalg.eigvalsh(pseudo_inertia(phi[i*N_PARAMS:(i+1)*N_PARAMS]))
        if np.any(eigs_pi < -1e-6):  flags.append(f"PSI_NEG({eigs_pi.min():.2e})")
        if Fv < 0: flags.append("Fv_NEG")
        if Fc < 0: flags.append("Fc_NEG")
        flag_str = "  ".join(flags) if flags else "OK"
        print(f"  {LINK_NAMES[i]:<14} {m:>8.4f} {c[0]:>8.4f} {c[1]:>8.4f} {c[2]:>8.4f}"
              f" {Fv:>8.4f} {Fc:>8.4f} {F0:>8.4f}  {flag_str}")

    print(f"\n  Inertia at CoM — eigenvalues [kg·m²]:")
    print(f"  {'Link':<14} {'λ₁':>12} {'λ₂':>12} {'λ₃':>12}  triangle slack")
    print("  " + "-" * 70)
    for i in range(N_JOINTS):
        m, mc, c, J_O, J_com, *_ = parse_link(phi, i)
        lam = np.sort(np.linalg.eigvalsh(J_com))
        l1, l2, l3 = lam
        slack = [l1+l2-l3, l2+l3-l1, l1+l3-l2]
        slack_str = f"[{slack[0]:+.2e}, {slack[1]:+.2e}, {slack[2]:+.2e}]"
        print(f"  {LINK_NAMES[i]:<14} {l1:>12.6f} {l2:>12.6f} {l3:>12.6f}  {slack_str}")
    print()


# ── Figure 1: Parameter bars ──────────────────────────────────────────────────

def plot_parameters(phi_list, labels):
    """Bar charts of mass, CoM norm, inertia eigenvalue spread for each .npy."""
    n_files = len(phi_list)
    x = np.arange(N_JOINTS)
    width = 0.8 / n_files

    fig, axes = plt.subplots(3, 1, figsize=(13, 10))
    fig.suptitle("Identified Parameters — Physical Sanity Check", fontsize=12, fontweight='bold')

    ax_m, ax_c, ax_j = axes

    # Expected mass range shading
    lo_arr = [r[0] for r in MASS_EXPECTED]
    hi_arr = [r[1] for r in MASS_EXPECTED]
    for xi, (lo, hi) in enumerate(MASS_EXPECTED):
        ax_m.axvspan(xi - 0.45, xi + 0.45, ymin=lo/max(hi_arr)/1.1,
                     ymax=hi/max(hi_arr)/1.1, alpha=0.08, color='lime', zorder=0)

    for fi, (phi, label) in enumerate(zip(phi_list, labels)):
        offset = (fi - (n_files-1)/2) * width
        masses, com_norms, jmin, jmax = [], [], [], []

        for i in range(N_JOINTS):
            m, mc, c, J_O, J_com, *_ = parse_link(phi, i)
            masses.append(m)
            com_norms.append(np.linalg.norm(c))
            eigs = np.sort(np.linalg.eigvalsh(J_com))
            jmin.append(eigs[0])
            jmax.append(eigs[-1])

        col = COLORS[fi % len(COLORS)]
        ax_m.bar(x + offset, masses,     width, label=label, color=col, alpha=0.8)
        ax_c.bar(x + offset, com_norms,  width, label=label, color=col, alpha=0.8)
        ax_j.bar(x + offset, jmax,       width, label=label, color=col, alpha=0.8)
        ax_j.bar(x + offset, [max(v, 0) for v in jmin], width, color='red',
                 alpha=0.5, label=(f"{label} λ_min" if fi == 0 else None))

    for ax in axes:
        ax.set_xticks(x); ax.set_xticklabels(LINK_NAMES, rotation=15, ha='right', fontsize=9)
        ax.grid(True, axis='y', alpha=0.3); ax.legend(fontsize=8)
        ax.axhline(0, color='white', linewidth=0.5)

    ax_m.set_ylabel("Mass [kg]"); ax_m.set_title("Link mass  (green band = expected range)")
    ax_c.set_ylabel("|CoM| [m]"); ax_c.set_title("Distance from link frame origin to CoM")
    ax_j.set_ylabel("[kg·m²]");   ax_j.set_title("I^CoM eigenvalues (red = λ_min, should be ≥ 0)")

    # Flag near-floor masses
    for fi, (phi, label) in enumerate(zip(phi_list, labels)):
        for i in range(N_JOINTS):
            m = phi[i * N_PARAMS]
            if m < MASS_EXPECTED[i][0]:
                ax_m.annotate("!", xy=(i + (fi - (n_files-1)/2)*width, m),
                               ha='center', va='bottom', color='red',
                               fontsize=11, fontweight='bold')

    plt.tight_layout()
    return fig


# ── Figure 2: Torque fit ──────────────────────────────────────────────────────

def plot_torque_fit(phi_list, labels, csv_path, fc=10.0, stride=1):
    print(f"\nBuilding regressor from {csv_path} ...")
    t, q, dq, ddq, tau_meas = load_and_filter(csv_path, fc=fc, stride=stride)
    N = len(t)

    print(f"  {N} samples, computing W ...")
    W_rows = np.empty((N * N_JOINTS, N_JOINTS * N_PARAMS))
    for n in range(N):
        if n % max(1, N//5) == 0:
            print(f"  {n}/{N}", end="\r", flush=True)
        W_rows[n*N_JOINTS:(n+1)*N_JOINTS, :] = regressor_fast(q[n], dq[n], ddq[n])
    print(f"  Done.        ")

    fig, axes = plt.subplots(N_JOINTS, 1, figsize=(14, 11), sharex=True)
    fig.suptitle(f"Torque Fit — {csv_path}", fontsize=11, fontweight='bold')

    rmse_table = {}
    for fi, (phi, label) in enumerate(zip(phi_list, labels)):
        tau_pred = (W_rows @ phi).reshape(N, N_JOINTS)
        rmse_table[label] = []
        for j in range(N_JOINTS):
            rmse = np.sqrt(np.mean((tau_meas[:, j] - tau_pred[:, j])**2))
            rmse_table[label].append(rmse)

    for j, ax in enumerate(axes):
        ax.plot(t, tau_meas[:, j], color='white', lw=0.9, alpha=0.7, label='measured')
        for fi, (phi, label) in enumerate(zip(phi_list, labels)):
            tau_pred = (W_rows @ phi).reshape(N, N_JOINTS)
            rmse = rmse_table[label][j]
            ax.plot(t, tau_pred[:, j], color=COLORS[fi % len(COLORS)],
                    lw=0.9, ls='--', label=f"{label} (RMSE={rmse:.3f}Nm)")
        ax.set_ylabel(f"τ [Nm]", fontsize=8)
        ax.set_title(f"{JOINT_NAMES[j]}", fontsize=8, loc='left')
        ax.grid(True, alpha=0.25); ax.legend(fontsize=7, loc='upper right')

    axes[-1].set_xlabel("Time [s]")

    # RMSE summary
    print(f"\n{'Joint':<16}", end="")
    for label in labels:
        print(f"  {label[:14]:>14}", end="")
    print()
    print("-" * (16 + 16*len(labels)))
    for j in range(N_JOINTS):
        print(f"  {JOINT_NAMES[j]:<14}", end="")
        for label in labels:
            print(f"  {rmse_table[label][j]:>14.4f}", end="")
        print()

    plt.tight_layout()
    return fig


# ── Figure 3: Data excitation quality ────────────────────────────────────────

def plot_data_quality(csv_path, fc=10.0, stride=1):
    print(f"\nPlotting data quality for {csv_path} ...")
    t, q, dq, ddq, tau_meas = load_and_filter(csv_path, fc=fc, stride=stride)

    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f"Data Quality — {csv_path}", fontsize=11, fontweight='bold')

    datasets = [
        (q,       "Position [rad]",   "pos"),
        (dq,      "Velocity [rad/s]", "vel"),
        (ddq,     "Accel [rad/s²]",   "acc"),
        (tau_meas,"Torque [Nm]",      "τ"),
    ]
    for ax, (data, ylabel, key) in zip(axes, datasets):
        for j in range(N_JOINTS):
            ax.plot(t, data[:, j], color=COLORS[j], lw=0.8, label=JOINT_NAMES[j])
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.25)

    axes[0].legend(fontsize=7, loc='upper right', ncol=3)
    axes[-1].set_xlabel("Time [s]")

    # Print excitation stats
    print(f"\n  Excitation statistics:")
    print(f"  {'Joint':<16} {'q_range[°]':>12} {'dq_rms[r/s]':>13} {'τ_rms[Nm]':>11}")
    print("  " + "-" * 56)
    for j in range(N_JOINTS):
        q_range = np.degrees(q[:, j].max() - q[:, j].min())
        dq_rms  = np.sqrt(np.mean(dq[:, j]**2))
        tau_rms = np.sqrt(np.mean(tau_meas[:, j]**2))
        flag = "  <-- LOW EXCITATION" if q_range < 20 else ""
        print(f"  {JOINT_NAMES[j]:<16} {q_range:>12.1f} {dq_rms:>13.4f} {tau_rms:>11.4f}{flag}")

    plt.tight_layout()
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Diagnose identified phi.npy — physical sanity + torque fit",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__)
    parser.add_argument("phi", nargs="+",
                        help="One or more .npy files to analyse (use glob, e.g. npy/phi_*.npy)")
    parser.add_argument("--csv",
                        help="CSV data file for torque-fit plots (optional)")
    parser.add_argument("--fc", type=float, default=10.0,
                        help="Low-pass filter cutoff [Hz] for CSV data (default 10)")
    parser.add_argument("--stride", type=int, default=1,
                        help="Subsample stride for CSV data (default 1)")
    parser.add_argument("--no-torque", action="store_true",
                        help="Skip torque-fit figure even if --csv is given")
    parser.add_argument("--save", action="store_true",
                        help="Save figures as PNG instead of showing interactively")
    parser.add_argument("--urdf", metavar="PATH",
                        help="URDF file used as reference overlay (default: urdf/vx300s.urdf)")
    parser.add_argument("--no-urdf", action="store_true",
                        help="Disable automatic URDF reference overlay")
    args = parser.parse_args()

    # Load all phi files
    phi_list, labels = [], []
    for path in args.phi:
        phi = np.load(path)
        if phi.shape != (78,):
            print(f"[skip] {path}: expected shape (78,), got {phi.shape}", file=sys.stderr)
            continue
        import os
        label = os.path.splitext(os.path.basename(path))[0]
        phi_list.append(phi)
        labels.append(label)
        print_report(phi, label=label)

    if not phi_list:
        sys.exit("No valid phi files loaded.")

    # Prepend URDF reference unless suppressed
    if not args.no_urdf:
        import os as _os
        _default_urdf = _os.path.join(
            _os.path.dirname(_os.path.abspath(__file__)), "urdf", "vx300s.urdf")
        urdf_path = args.urdf or _default_urdf
        if _os.path.exists(urdf_path):
            phi_urdf = load_urdf_phi(urdf_path)
            phi_list.insert(0, phi_urdf)
            labels.insert(0, "vx300s")
            print_report(phi_urdf, label=f"vx300s (URDF: {urdf_path})")
        elif args.urdf:
            print(f"[warn] URDF not found: {urdf_path}", file=sys.stderr)

    plt.style.use("dark_background")

    figs = []
    figs.append(plot_parameters(phi_list, labels))

    if args.csv and not args.no_torque:
        figs.append(plot_torque_fit(phi_list, labels, args.csv,
                                    fc=args.fc, stride=args.stride))
        figs.append(plot_data_quality(args.csv, fc=args.fc, stride=args.stride))

    if args.save:
        import datetime
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        import os; os.makedirs("figures", exist_ok=True)
        for i, fig in enumerate(figs):
            path = f"figures/diagnose_{ts}_{i+1}.png"
            fig.savefig(path, dpi=150, bbox_inches='tight')
            print(f"Saved {path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
