#!/usr/bin/env python3
"""Summarise a PD+gravity-comp α-sweep (control/pd_grav_control.py logs).

For each data/pdg_a<α>_<stamp>.npy it reports, per joint, the STEADY-STATE
(default: last 5 s) tracking droop (q_d − q), command current, and jitter. Then,
per joint, it tabulates droop vs α and linearly interpolates the α at which the
droop crosses zero — the closed-loop estimate of that joint's gravity scale
(α*·G_model ≈ true gravity). Read-only; no ROS.

    python3 tools/analyze_pdg_sweep.py                 # all data/pdg_a*.npy
    python3 tools/analyze_pdg_sweep.py --tail 3 --glob 'data/pdg_a*_1554*.npy'
"""
import argparse
import glob
import re

import numpy as np

JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']
# Log columns (all logs): 0=t, 1:7=q, 7:13=qd, 13:19=q_d, 19:25=u, 25=r.


def summarise(path, tail):
    a = np.load(path)
    t = a[:, 0] - a[0, 0]
    if t[-1] <= 0:
        return None
    m = t > (t[-1] - tail)
    q, qd_set, u = a[:, 1:7], a[:, 13:19], a[:, 19:25]
    droop = (qd_set[m] - q[m]).mean(axis=0)
    jit = q[m].std(axis=0)
    usettle = u[m].mean(axis=0)
    return dict(dur=t[-1], n=len(t), droop=droop, jit=jit, u=usettle)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--glob', default='data/pdg_a*.npy')
    ap.add_argument('--tail', type=float, default=5.0, help='steady-state window (s)')
    args = ap.parse_args()

    runs = []
    for p in sorted(glob.glob(args.glob)):
        mobj = re.search(r'pdg_a([0-9.]+)_', p)
        if not mobj:
            continue
        s = summarise(p, args.tail)
        if s:
            runs.append((float(mobj.group(1)), p, s))
    if not runs:
        print(f'no logs matched {args.glob}')
        return

    print(f'\nSteady-state over last {args.tail:.0f}s   (droop = q_d − q, rad; '
          f'u = settled current, mA; jit = std, rad)\n')
    for alpha, p, s in runs:
        flag = '' if s['dur'] > args.tail + 1 else '  [SHORT run — killed early]'
        print(f'α={alpha:<4} {p.split("/")[-1]}  ({s["dur"]:.1f}s){flag}')
        for j in (1, 2, 4):  # shoulder, elbow, wrist_angle (the gravity joints)
            print(f'    {JOINTS[j]:12} droop={s["droop"][j]:+.4f}  '
                  f'u={s["u"][j]:+7.0f}  jit={s["jit"][j]:.4f}')

    # droop vs α per joint, and zero-droop α (gravity scale) via interpolation
    print('\nDroop vs α  (sign convention: + = sagged below setpoint)\n')
    alphas = np.array([r[0] for r in runs])
    for j in (1, 2, 4):
        order = np.argsort(alphas)
        aa = alphas[order]
        dd = np.array([runs[k][2]['droop'][j] for k in order])
        cells = '  '.join(f'α{a:.1f}:{d:+.3f}' for a, d in zip(aa, dd))
        line = f'  {JOINTS[j]:12} {cells}'
        # zero crossing (need droop to change sign across the swept αs)
        if dd.min() < 0 < dd.max():
            k = np.where(np.diff(np.sign(dd)))[0][0]
            a0 = aa[k] + (aa[k + 1] - aa[k]) * (0 - dd[k]) / (dd[k + 1] - dd[k])
            line += f'   → zero-droop α* ≈ {a0:.2f}'
        print(line)
    print('\nα* is the closed-loop gravity scale: α*·G_model ≈ true gravity.')


if __name__ == '__main__':
    main()
