#!/usr/bin/env python3
"""PD + gravity-compensation regulation — first model-based controller.

WHY (phase transition): with a DH-fixed, statically-validated gravity model
(CHANGELOG 2026-06-13: shoulder gravity SHAPE r=0.997 in the real world), we can
start the thesis' control half. The static-gravity verdict cleared PD+G as
"safe to start" — feedback is robust to the ~0.58 scale gap, the arm won't fall.

CONTROL LAW (per joint, in mA, regulation to a fixed setpoint q_d):
    u = Kp·(q_d − q) + Ki·∫(q_d − q)dt − Kd·q̇ + α·G_mA(q)
  - Ki·∫: optional INTEGRAL term (PID), off by default (--ki-scale 0). It drives
    the steady-state droop → 0 despite the imperfect gravity model (the shoulder
    over-prediction / forearm_roll residual that left ~46 mm of EE error under
    pure PD+G). Gated until after the setpoint ramp + grace and anti-windup
    clamped (I_CAP) so a wound-up integral can't overpower the arm.
  - G_mA(q): the IDENTIFIED gravity, evaluated exactly as the static experiment
    validated it — Newton-Euler inverse dynamics on our φ at q̇=q̈=0, friction
    zeroed, divided by sysid_feasible.EFFORT_SCALE → master-motor mA (k_t and the
    dual-motor ×2 are baked into EFFORT_SCALE).
  - α: gravity feedforward gain. **Sweeping α and watching the steady-state droop
    (q_d − q) is the clean closed-loop resolution of the 0.58 anomaly:** the α
    that zeroes droop is the TRUE gravity scale. α≈1 ⇒ identified gravity is
    correct (0.58 was a position-mode stiction artifact) ⇒ precision CT is on the
    table. α≈0.58 ⇒ a real gravity scale error remains.

SAFETY (improves on control/trq.py, which ramped from zero current — momentarily
unsupporting the arm at the mode switch — and ran hot gains + a trajectory):
  - BUMP-FREE HANDOFF: before switching to current mode we read each joint's
    actual position-mode holding current and blend the command from there to the
    full law over --ramp-in seconds, so the arm is gravity-supported the whole
    time and there is no torque step.
  - Pure SETPOINT regulation (holds the pose it starts at) — no trajectory yet.
  - Per-joint soft position limits (current toward a limit ramps to zero) and
    hard current caps.
  - Kill switches: |q̇| or |q_d − q| over a bound ⇒ emergency stop (zero current).
  - SIGINT ⇒ zero current, back to position mode.

USAGE — position the arm at a test pose FIRST (position mode), e.g.
    python3 control/set_pos.py 0 -0.6 0.5 0 0 0
then launch the regulator (it captures that pose as the setpoint):
    python3 control/pd_grav_control.py --alpha 0.0          # PD only: see the droop
    python3 control/pd_grav_control.py --alpha 0.6          # ~static op point
    python3 control/pd_grav_control.py --alpha 1.0          # full identified gravity
Start gentle: --gain-scale 0.5. Ctrl-C stops safely.

Prerequisite (own terminal): arm driver running:
    ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s
"""
import argparse
import os
import signal
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.srv import OperatingModes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import sysid_feasible as sf  # noqa: E402  (pure numpy; no ROS/Pinocchio)

# Real Pinocchio (3.9, needs ROS sourced) gives gravity + FK + IK straight from
# the URDF. Without ROS, `import pinocchio` resolves to a bogus PyPI 0.4.3 that
# lacks buildModelFromUrdf — so gate on that attribute, not just the import.
try:
    import pinocchio as pin  # noqa: E402
    PIN_OK = hasattr(pin, 'buildModelFromUrdf')
except Exception:
    pin, PIN_OK = None, False

ARM_JOINTS = sf.ARM_JOINTS
DEFAULT_MODEL = os.path.join(
    REPO, "outputs/npy/traj_run_200hz_20260612_131613__sysid_feasible-v1-5__cfg-9ef2c992.npy")
# Identified URDF (same model as DEFAULT_MODEL φ; gravity matches to ~1e-7 Nm).
DEFAULT_URDF = os.path.join(
    REPO, "outputs/urdf/traj_run_200hz_20260612_131613__sysid_feasible-v1-5__"
          "cfg-9ef2c992__phi_to_urdf-v1-1__cfg-3ef0a00c.urdf")
EE_FRAME = "ee_link"   # end-effector frame in the identified URDF

# Per-joint base gains (mA/rad, mA/(rad/s)). Conservative starting point; scale
# globally with --gain-scale and tune per joint later. Bigger on the proximal
# gravity joints (shoulder/elbow), small on the distal/wrist joints.
KP_BASE = np.array([2500.0, 3500.0, 3000.0, 1200.0, 1200.0, 600.0])
KD_BASE = np.array([ 120.0,  160.0,  150.0,   60.0,   60.0,  30.0])

# Per-joint integral gains (mA/(rad·s)) and the max integral CONTRIBUTION each
# joint may accumulate (mA, anti-windup clamp). The integral term drives the
# steady-state droop (from the imperfect shoulder/forearm_roll gravity model) to
# zero. Off by default (--ki-scale 0); sweep it up like α. Proximal gravity
# joints get more authority; the I-cap stays well under the gravity load so a
# wound-up integral can never by itself overpower the arm.
KI_BASE = np.array([300.0, 400.0, 350.0, 250.0, 250.0, 120.0])
I_CAP   = np.array([200.0, 400.0, 400.0, 350.0, 350.0, 150.0])

# Hard per-joint current caps (mA). Shoulder/elbow carry gravity → higher, with
# headroom for the --dual-gain diagnostic (still well under the XM540 stall ~2.3 A).
CURRENT_CAP = np.array([700.0, 1400.0, 1400.0, 400.0, 400.0, 300.0])

# Dual-motor joints (shoulder, elbow): a shadow motor mirrors the master, so
# current control may deliver less than the commanded joint torque. --dual-gain
# multiplies the command to these joints to test/compensate for that.
DUAL_MASK = sf.MOTORS_PER_JOINT == 2

# Joint position limits (rad) — from run_trajectories / static_gravity_poses,
# EXCEPT shoulder: bumped 0.30 → 0.80 (2026-06-18) to reach untrained, more
# gravity-loaded poses for the model-extrapolation test. The hardware/URDF
# shoulder range is much wider (URDF upper = +1.76); 0.30 was a conservative
# data-collection envelope. 0.80 is still well inside the URDF limit but enters
# the extended, higher-tip-moment region — drive there only under control.
LIMITS_LO = np.array([-2.80, -1.50, -0.20, -1.50, -1.50, -2.80])
LIMITS_HI = np.array([ 2.80,  0.80,  1.00,  1.50,  1.50,  2.80])
SOFT_BUFFER = 0.20  # rad: current toward a limit ramps to zero across this band


def soft_limit_scale(u, q):
    """Zero out only the component of u that pushes a joint past its soft limit."""
    out = u.copy()
    for i in range(6):
        if q[i] <= LIMITS_LO[i] and u[i] < 0:
            out[i] = 0.0
        elif q[i] >= LIMITS_HI[i] and u[i] > 0:
            out[i] = 0.0
        elif q[i] < LIMITS_LO[i] + SOFT_BUFFER and u[i] < 0:
            out[i] = u[i] * (q[i] - LIMITS_LO[i]) / SOFT_BUFFER
        elif q[i] > LIMITS_HI[i] - SOFT_BUFFER and u[i] > 0:
            out[i] = u[i] * (LIMITS_HI[i] - q[i]) / SOFT_BUFFER
    return out


class PDGravNode(Node):
    def __init__(self, args):
        super().__init__('pd_grav_control')
        self.args = args
        self.robot = args.robot_model

        # Identified model → gravity-only φ (friction params zeroed). Kept even in
        # URDF mode as a fallback and for the startup equivalence self-check.
        phi = np.load(args.model)
        self.phi_g = phi.copy()
        for i in range(sf.N_JOINTS):
            self.phi_g[i * sf.N_PARAMS + 10:i * sf.N_PARAMS + 13] = 0.0
        self._z = np.zeros(6)

        # Gravity source: Pinocchio on the identified URDF (so the controller
        # literally runs on your URDF, and the same model gives FK/IK), with a
        # graceful fallback to the φ vector (numerically identical, ~1e-7 Nm).
        self.use_pin = (args.gravity_source == 'urdf' and PIN_OK)
        self.pin_m = self.pin_d = self.ee_id = None
        if args.gravity_source == 'urdf' and not PIN_OK:
            self.get_logger().warn('URDF/Pinocchio requested but real pinocchio not '
                                   'available (source ROS) — falling back to φ vector')
        if self.use_pin:
            self.pin_m = pin.buildModelFromUrdf(args.urdf)
            self.pin_d = self.pin_m.createData()
            self.ee_id = self.pin_m.getFrameId(EE_FRAME)
            # Pinocchio joint order matches ARM_JOINTS for this URDF (verified).

        self.kp = KP_BASE * args.gain_scale
        self.kd = KD_BASE * args.gain_scale
        self.ki = KI_BASE * args.ki_scale
        self.i_err = np.zeros(6)     # integral accumulator (rad·s)
        self.t_int_prev = None       # wall time of last integral update
        self.alpha = args.alpha

        self.pos = self.vel = self.eff = None
        self.q_d = np.array(args.hold_pose) if args.hold_pose else None
        self.u_hold = None           # measured holding current at handoff
        self.t_switch = None         # wall time current mode engaged
        self.estopped = False
        self.stop_requested = False  # graceful-exit flag (set by SIGINT)
        self._vel_trip = 0           # consecutive velocity-kill samples (debounce)
        self._pos_trip = 0           # consecutive position-kill samples (debounce)
        self.q_ref_start = None      # arm position captured at current-mode engage
        self.t_prev = None           # for filtered finite-difference velocity
        self.q_prev = None
        self.v_filt = np.zeros(6)    # low-pass filtered velocity (for damping/kill)
        self.log = []

        # Joints whose model gravity is untrustworthy → feed forward their
        # measured holding current instead (default: forearm_roll, the joint-4
        # defect — CHANGELOG 2026-06-13). Mask resolved against ARM_JOINTS.
        self.ff_hold_mask = np.array([j in args.hold_ff for j in ARM_JOINTS])

        self.cmd_pub = self.create_publisher(
            JointGroupCommand, f'/{self.robot}/commands/joint_group', 10)
        self.create_subscription(
            JointState, f'/{self.robot}/joint_states', self._cb, 10)

    # --- feedback ---------------------------------------------------------
    def _cb(self, msg):
        idx = {n: i for i, n in enumerate(msg.name)}
        self.pos = np.array([msg.position[idx[j]] for j in ARM_JOINTS])
        self.vel = np.array([msg.velocity[idx[j]] for j in ARM_JOINTS])
        self.eff = np.array([msg.effort[idx[j]] for j in ARM_JOINTS])

    def gravity_Nm(self, q):
        """Gravity torque (Nm), ARM_JOINTS order — from the URDF (Pinocchio) or φ."""
        if self.use_pin:
            return np.asarray(pin.computeGeneralizedGravity(self.pin_m, self.pin_d, q))
        return sf.inverse_dynamics_phi(q, self._z, self._z, self.phi_g)

    def gravity_mA(self, q):
        return self.gravity_Nm(q) / sf.EFFORT_SCALE

    def ee_pose(self, q):
        """Forward kinematics: end-effector (xyz [m], rpy [rad]) from joint angles.
        Requires the URDF/Pinocchio model; returns None in φ-only mode."""
        if not self.use_pin:
            return None
        pin.framesForwardKinematics(self.pin_m, self.pin_d, np.asarray(q))
        M = self.pin_d.oMf[self.ee_id]
        return np.asarray(M.translation), pin.rpy.matrixToRpy(M.rotation)

    # --- operating modes --------------------------------------------------
    def _set_mode(self, mode):
        cli = self.create_client(OperatingModes, f'/{self.robot}/set_operating_modes')
        while not cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('waiting for set_operating_modes …')
        req = OperatingModes.Request()
        req.cmd_type, req.name, req.mode = 'group', 'arm', mode
        req.profile_type, req.profile_velocity, req.profile_acceleration = 'time', 0, 0
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        self.get_logger().info(f'arm → {mode} mode')

    def publish(self, u):
        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = [float(x) for x in u]
        self.cmd_pub.publish(msg)

    def estop(self, why=''):
        # NEVER command zero current on a gravity-loaded arm — it would go limp
        # and fall. Just flag; main() parks the arm in position mode (its PID
        # catches and holds), keeping the last current applied through the switch.
        if not self.estopped:
            self.estopped = True
            self.get_logger().error(f'KILL: {why} — parking in position mode')

    # --- control loop -----------------------------------------------------
    def step(self):
        if self.pos is None or self.estopped:
            return
        q = self.pos
        now = time.time()
        elapsed = now - self.t_switch

        # Filtered velocity for damping & the velocity kill: the raw Dynamixel
        # velocity register is noisy and underreports ~50-60% on the distal joints
        # (THESIS_NOTES), which cripples the Kd damping term. Use a low-pass
        # filtered finite difference of position instead (τ≈30 ms, like trq.py).
        # GLITCH REJECTION: a timer burst gives a near-zero dt and a corrupt/late
        # joint_states gives a position jump — either makes Δq/dt explode on all
        # joints at once and false-trip the velocity kill. Reject samples with too
        # small a dt or a non-physical estimate (arm tops out ~3-4 rad/s): reuse
        # the last filtered velocity and DON'T advance the filter state, so the
        # next good sample integrates over the correct interval.
        if self.t_prev is None:
            self.t_prev, self.q_prev = now, q.copy()
            qd = self.v_filt
        else:
            dt = now - self.t_prev
            v_est = (q - self.q_prev) / dt if dt > 1e-9 else self.v_filt
            if dt < 0.004 or np.max(np.abs(v_est)) > 8.0:
                qd = self.v_filt                       # glitch — reuse, don't advance
            else:
                a_lp = dt / (0.03 + dt)
                self.v_filt = a_lp * v_est + (1.0 - a_lp) * self.v_filt
                self.t_prev, self.q_prev = now, q.copy()
                qd = self.v_filt

        # RAMPED SETPOINT: the mode switch leaves the (heavy, gravity-loaded)
        # joints displaced — yanking them back to q_d from a large error makes the
        # PD respond violently (overshoot → oscillation → coupling into other
        # joints). Instead, capture where the arm actually is at engage and ramp
        # the reference q_d_ramp from there to q_d over --recover-time, so the
        # tracking error stays small and the recovery is gentle.
        if self.q_ref_start is None:
            self.q_ref_start = q.copy()
        frac = min(1.0, elapsed / max(self.args.recover_time, 1e-3))
        q_ref = self.q_ref_start + frac * (self.q_d - self.q_ref_start)
        err = q_ref - q

        # Kills now use the TRACKING error (q_ref − q): a real runaway grows it,
        # but the transient/ramp does not. Position kill debounced (rides the
        # 1-sample mode-switch position glitch); velocity kill graced + debounced.
        if np.any(np.abs(err) > self.args.pos_error_limit):
            self._pos_trip += 1
            if self._pos_trip >= 4:
                self.estop(f'|q_ref−q|>{self.args.pos_error_limit} ({np.round(err,2)})')
                return
        else:
            self._pos_trip = 0
        if elapsed > self.args.grace and np.any(np.abs(qd) > self.args.vel_limit):
            self._vel_trip += 1
            if self._vel_trip >= 3:
                self.estop(f'|q̇|>{self.args.vel_limit} ({np.round(qd,2)})')
                return
        else:
            self._vel_trip = 0

        # Feedforward: model gravity scaled by α, EXCEPT joints in the hold-FF
        # mask (known-bad model, e.g. forearm_roll) which get their measured
        # position-mode holding current as a constant FF.
        ff = self.alpha * self.gravity_mA(q)
        ff[self.ff_hold_mask] = self.u_hold[self.ff_hold_mask]

        # INTEGRAL term: accumulate the tracking error to drive steady-state droop
        # → 0 despite the imperfect gravity model (shoulder over-prediction,
        # forearm_roll constant-FF residual). Only integrate AFTER the setpoint
        # ramp has finished (frac≥1) and the grace window has passed — integrating
        # during the engage transient/ramp would just wind up on a deliberately
        # large, shrinking error. Anti-windup: clamp each joint's accumulator so
        # its contribution ki·i_err never exceeds I_CAP.
        if self.ki.any():
            if self.t_int_prev is None:
                self.t_int_prev = now
            dt_i = now - self.t_int_prev
            self.t_int_prev = now
            if frac >= 1.0 and elapsed > self.args.grace:
                self.i_err += err * dt_i
                with np.errstate(divide='ignore', invalid='ignore'):
                    cap = np.where(self.ki > 0, I_CAP / np.maximum(self.ki, 1e-9), 0.0)
                self.i_err = np.clip(self.i_err, -cap, cap)

        # PD (on the ramped reference) + integral + gravity FF (mA).
        u = self.kp * err + self.ki * self.i_err - self.kd * qd + ff

        # Bump-free handoff: blend from the measured holding current to the law.
        r = min(1.0, elapsed / max(self.args.ramp_in, 1e-3))
        u = (1.0 - r) * self.u_hold + r * u

        # Dual-motor compensation diagnostic: scale shoulder/elbow command.
        u[DUAL_MASK] *= self.args.dual_gain

        u = soft_limit_scale(u, q)
        u = np.clip(u, -CURRENT_CAP, CURRENT_CAP)
        self.publish(u)
        # log cols: 0=t, 1:7=q, 7:13=qd, 13:19=q_d, 19:25=u, 25=r, 26:32=q_ref,
        #           32:38=integral contribution ki·i_err (mA)
        self.log.append(np.r_[time.time(), q, qd, self.q_d, u, r, q_ref,
                              self.ki * self.i_err])


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default=DEFAULT_MODEL, help='identified φ .npy (fallback / check)')
    ap.add_argument('--gravity-source', choices=['urdf', 'phi'], default='urdf',
                    help='compute gravity from the URDF via Pinocchio (also gives '
                         'FK/IK) or from the φ vector; urdf falls back to φ if '
                         'real pinocchio is unavailable')
    ap.add_argument('--urdf', default=DEFAULT_URDF, help='identified URDF for Pinocchio')
    ap.add_argument('--alpha', type=float, default=1.0, help='gravity FF gain')
    ap.add_argument('--gain-scale', type=float, default=0.5, help='global PD scale')
    ap.add_argument('--ki-scale', type=float, default=0.0,
                    help='global INTEGRAL scale (PID). 0 = off (verified PD+G '
                         'behavior). Sweep up (e.g. 0.5, 1.0) to drive the '
                         'steady-state droop → 0; integration is gated until '
                         'after the setpoint ramp + grace, with per-joint '
                         'anti-windup (I_CAP).')
    ap.add_argument('--grace', type=float, default=0.25,
                    help='seconds the VELOCITY kill is suppressed after the mode '
                         'switch, to ride the entry transient (the position/'
                         'tracking-error kill is always active as the backstop)')
    ap.add_argument('--dual-gain', type=float, default=1.0,
                    help='multiply commanded current to dual-motor joints '
                         '(shoulder/elbow); 2.0 tests the half-torque hypothesis')
    ap.add_argument('--hold-ff', nargs='*', default=['forearm_roll'],
                    help='joints whose FF = measured holding current (untrusted model)')
    ap.add_argument('--hold-pose', type=float, nargs=6, default=None,
                    help='setpoint (rad); default = pose at launch')
    ap.add_argument('--rate', type=float, default=100.0, help='control Hz')
    ap.add_argument('--ramp-in', type=float, default=0.0,
                    help='current-blend handoff (s); 0 = off. The ramped setpoint '
                         'already smooths engage, and a blend muzzles control '
                         'authority during the transient, so default off.')
    ap.add_argument('--recover-time', type=float, default=3.0,
                    help='seconds to ramp the setpoint from the post-switch '
                         'position back to q_d (gentle recovery from the transient)')
    ap.add_argument('--duration', type=float, default=0.0, help='auto-stop (s); 0=until Ctrl-C')
    ap.add_argument('--vel-limit', type=float, default=2.5,
                    help='kill if |q̇| exceeds (rad/s). 2.5 rides the benign elbow '
                         'entry-transient recovery (~1.7); the debounced tracking-'
                         'error kill is the real runaway backstop.')
    ap.add_argument('--pos-error-limit', type=float, default=0.5, help='kill if |q_d−q| exceeds (rad)')
    ap.add_argument('--robot-model', default='vx300s')
    ap.add_argument('--output', default=None, help='log .npy path')
    args = ap.parse_args()

    rclpy.init()
    node = PDGravNode(args)

    # Wait for feedback, then SETTLE: the driver's first joint_states message can
    # carry placeholder values (e.g. all −π), so collect ~1.5 s of fresh samples
    # and use the medians for both the setpoint and the holding current.
    print('[pdg] waiting for joint_states …')
    while rclpy.ok() and node.pos is None:
        rclpy.spin_once(node, timeout_sec=0.1)
    print('[pdg] settling / measuring holding current (1.5 s) …')
    pos_s, eff_s = [], []
    t0 = time.time()
    while time.time() - t0 < 1.5:
        rclpy.spin_once(node, timeout_sec=0.02)
        if node.pos is not None:
            pos_s.append(node.pos.copy())
            eff_s.append(node.eff.copy())
    pos_med = np.median(np.vstack(pos_s), axis=0)
    node.u_hold = np.median(np.vstack(eff_s), axis=0)
    if node.q_d is None:
        node.q_d = pos_med

    # Sanity gates before any torque is commanded.
    if np.any(node.q_d < LIMITS_LO - 1e-3) or np.any(node.q_d > LIMITS_HI + 1e-3):
        print(f'[pdg] ABORT: setpoint {np.round(node.q_d, 3).tolist()} is outside the '
              f'joint limits — likely a bad read or the arm is not holding a pose.\n'
              f'      Move to a test pose first, e.g.  python3 control/set_pos.py 0 -0.6 0.5 0 0 0')
        rclpy.shutdown()
        return
    print(f'[pdg] setpoint q_d  = {np.round(node.q_d, 3).tolist()}')
    print(f'[pdg] hold current  ≈ {np.round(node.u_hold).tolist()} mA')
    src = f'URDF/Pinocchio ({os.path.basename(args.urdf)})' if node.use_pin else 'φ vector'
    print(f'[pdg] gravity source: {src}')
    print(f'[pdg] gravity_mA(q_d)≈ {np.round(node.gravity_mA(node.q_d)).tolist()} mA  '
          f'(α={args.alpha}, gain×{args.gain_scale}, ki×{args.ki_scale}'
          f'{" — INTEGRAL ON" if args.ki_scale > 0 else ""})')
    if node.use_pin:
        # Self-check: URDF gravity vs φ vector (should be ~1e-7 Nm), and FK.
        diff = np.max(np.abs(node.gravity_Nm(node.q_d)
                             - sf.inverse_dynamics_phi(node.q_d, node._z, node._z, node.phi_g)))
        print(f'[pdg] URDF↔φ gravity check at q_d: max diff {diff:.2e} Nm (≈0 ⇒ same model)')
        xyz, rpy = node.ee_pose(node.q_d)
        print(f'[pdg] FK: end-effector @ q_d  xyz=[{xyz[0]:+.3f}, {xyz[1]:+.3f}, '
              f'{xyz[2]:+.3f}] m  rpy=[{rpy[0]:+.2f}, {rpy[1]:+.2f}, {rpy[2]:+.2f}] rad')
    if np.any(node.ff_hold_mask):
        held = [ARM_JOINTS[i] for i in range(6) if node.ff_hold_mask[i]]
        print(f'[pdg] FF = measured holding current for: {held} (model untrusted there)')
    if args.dual_gain != 1.0:
        print(f'[pdg] DUAL-MOTOR GAIN = {args.dual_gain}× on shoulder/elbow '
              f'(half-torque hypothesis test)')
    if np.allclose(node.u_hold, 0.0, atol=2.0):
        print('[pdg] WARNING: holding current ≈ 0 — the arm appears LIMP (torque off, '
              'resting on its stops). Move to a pose with set_pos.py so it is actively\n'
              '      holding before running, otherwise the handoff and droop test are meaningless.')

    # Engage current mode. Publish the measured holding current FIRST so the
    # servo has a gravity-supporting goal the instant it leaves position mode
    # (no zero-current gap → no fall through the switch).
    node._set_mode('current')
    node.t_switch = time.time()
    for _ in range(5):
        node.publish(node.u_hold)
        time.sleep(0.01)
    node.timer = node.create_timer(1.0 / args.rate, node.step)

    # SIGINT only flags a graceful exit; cleanup (park in position mode) runs
    # below while the rclpy context is still alive — never leave the arm limp.
    def on_sigint(sig, frm):
        node.stop_requested = True
    signal.signal(signal.SIGINT, on_sigint)

    print(f'[pdg] running @ {args.rate:.0f} Hz — Ctrl-C to stop. '
          f'Ramp-in {args.ramp_in}s.')
    t_end = time.time() + args.duration if args.duration > 0 else None
    while rclpy.ok() and not node.stop_requested and not node.estopped:
        if t_end is not None and time.time() >= t_end:
            break
        rclpy.spin_once(node, timeout_sec=0.01)

    # Cleanup with a LIVE context: hand the arm straight to its position PID,
    # which holds the present pose. We do NOT publish zero current first — the
    # servo keeps applying the last commanded current until the mode switch
    # completes, so the arm stays supported (zero-current would let it collapse).
    node.timer.cancel()
    print('[pdg] stopping — parking in position mode …')
    node._set_mode('position')
    if node.log:
        out = args.output or f'data/pdg_a{args.alpha}_{time.strftime("%Y%m%d_%H%M%S")}.npy'
        arr = np.vstack(node.log)
        np.save(out, arr)
        # steady-state droop over the last second (post ramp-in)
        tail = arr[arr[:, 0] > arr[-1, 0] - 1.0]
        droop = node.q_d - tail[:, 1:7].mean(axis=0)
        print(f'[pdg] steady-state droop (q_d−q) = {np.round(droop, 4).tolist()} rad')
        print(f'[pdg] log → {out}')
    if rclpy.ok():
        rclpy.shutdown()
    print('[pdg] done.')


if __name__ == '__main__':
    main()
