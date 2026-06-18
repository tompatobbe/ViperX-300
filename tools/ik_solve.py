#!/usr/bin/env python3
"""Inverse kinematics for the ViperX-300 from the identified URDF (Pinocchio).

Step 3 of docs/CONTROL_ROADMAP.md: desired end-effector pose -> joint angles, to
feed to the PD+gravity-comp controller as its setpoint. Uses the SAME URDF/
Pinocchio model as control/pd_grav_control.py, so FK/IK/gravity are all one model.

Method: damped least-squares (Levenberg-Marquardt) on the EE frame Jacobian.
Position-only by default (the 6-DoF arm leaves orientation free); pass --rpy for a
full 6-DoF pose target. Joint angles are clamped to the URDF limits each step, and
the result is checked against the controller's conservative software limits.

Needs ROS sourced for the real Pinocchio (see CLAUDE.md).

    source /opt/ros/humble/setup.bash
    python3 tools/ik_solve.py --xyz 0.30 0.0 0.40
    python3 tools/ik_solve.py --xyz 0.30 0.0 0.40 --rpy 0 1.57 0 --q-init 0 -0.6 0.5 0 0 0
"""
import argparse
import os
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_URDF = os.path.join(
    REPO, "outputs/urdf/traj_run_200hz_20260612_131613__sysid_feasible-v1-5__"
          "cfg-9ef2c992__phi_to_urdf-v1-1__cfg-3ef0a00c.urdf")
EE_FRAME = "ee_link"
ARM_JOINTS = ['waist', 'shoulder', 'elbow', 'forearm_roll', 'wrist_angle', 'wrist_rotate']

# Controller's conservative software limits (control/pd_grav_control.py) — the IK
# result is checked against these so we never hand the controller an out-of-range
# setpoint. (URDF kinematic limits are wider; these are the safe operating box.)
SOFT_LO = np.array([-2.80, -1.50, -0.20, -1.50, -1.50, -2.80])
SOFT_HI = np.array([ 2.80,  0.80,  1.00,  1.50,  1.50,  2.80])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--xyz', type=float, nargs=3, required=True, help='target EE position [m]')
    ap.add_argument('--rpy', type=float, nargs=3, default=None,
                    help='target EE orientation [rad]; omit for position-only IK')
    ap.add_argument('--q-init', type=float, nargs=6, default=[0, -0.6, 0.5, 0, 0, 0],
                    help='initial guess (rad); default a safe folded pose')
    ap.add_argument('--urdf', default=DEFAULT_URDF)
    ap.add_argument('--frame', default=EE_FRAME)
    ap.add_argument('--max-iter', type=int, default=2000)
    ap.add_argument('--tol', type=float, default=1e-4, help='convergence tol (m, or 6D norm)')
    ap.add_argument('--damp', type=float, default=1e-4, help='LM damping')
    ap.add_argument('--step', type=float, default=0.5, help='integration step')
    args = ap.parse_args()

    try:
        import pinocchio as pin
        assert hasattr(pin, 'buildModelFromUrdf')
    except Exception as e:
        sys.exit(f'real Pinocchio not available ({e}) — source ROS: '
                 f'source /opt/ros/humble/setup.bash')

    m = pin.buildModelFromUrdf(args.urdf)
    d = m.createData()
    fid = m.getFrameId(args.frame)
    lo = np.maximum(m.lowerPositionLimit, -np.pi)
    hi = np.minimum(m.upperPositionLimit, np.pi)

    # Desired pose. Position-only uses a 3-row task; full pose uses 6D log error.
    full = args.rpy is not None
    R_des = pin.rpy.rpyToMatrix(*args.rpy) if full else np.eye(3)
    oMdes = pin.SE3(R_des, np.array(args.xyz))

    q = np.clip(np.array(args.q_init, float), lo, hi)
    converged = False
    for _ in range(args.max_iter):
        pin.framesForwardKinematics(m, d, q)
        oMf = d.oMf[fid]
        if full:
            err = pin.log6(oMf.inverse() * oMdes).vector            # 6, local frame
            J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL)   # 6x6
        else:
            err = oMdes.translation - oMf.translation               # 3, world
            J = pin.computeFrameJacobian(m, d, q, fid, pin.LOCAL_WORLD_ALIGNED)[:3]
        if np.linalg.norm(err) < args.tol:
            converged = True
            break
        JJt = J @ J.T
        dq = J.T @ np.linalg.solve(JJt + args.damp * np.eye(JJt.shape[0]), err)
        q = np.clip(pin.integrate(m, q, args.step * dq), lo, hi)

    # Final error report
    pin.framesForwardKinematics(m, d, q)
    oMf = d.oMf[fid]
    pos_err = np.linalg.norm(oMdes.translation - oMf.translation)

    print(f'\ntarget xyz = {np.round(args.xyz, 4).tolist()} m'
          + (f'  rpy = {np.round(args.rpy, 3).tolist()} rad' if full else '  (position-only)'))
    print(f'converged  = {converged}   position error = {1000*pos_err:.2f} mm')
    print(f'reached xyz= {np.round(oMf.translation, 4).tolist()} m')
    print(f'\njoint solution (rad):')
    for j, name in enumerate(ARM_JOINTS):
        warn = '  <-- OUTSIDE controller soft limit!' if (q[j] < SOFT_LO[j] - 1e-6
                                                          or q[j] > SOFT_HI[j] + 1e-6) else ''
        print(f'  {name:13} {q[j]:+.4f}{warn}')
    qs = ' '.join(f'{v:.4f}' for v in q)

    ok = converged and np.all(q >= SOFT_LO - 1e-6) and np.all(q <= SOFT_HI + 1e-6)
    print('\n' + ('READY — feed to the controller:' if ok
                  else 'NOT READY (did not converge or outside soft limits) — do not run blindly:'))
    print(f'  python3 control/set_pos.py {qs}        # optional: move there in position mode first')
    print(f'  python3 control/pd_grav_control.py --alpha 1.0 --hold-pose {qs}')


if __name__ == '__main__':
    main()
