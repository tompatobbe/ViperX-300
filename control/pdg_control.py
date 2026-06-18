#!/usr/bin/env python3
"""pdg_control.py — PD + gravity-compensation current controller (vx300s).

The first control-phase script. It regulates the arm to a setpoint (or holds the
current pose) in **current/torque mode**, using our identified dynamic model for
gravity feed-forward:

    u(mA) = soft · [ G(q) + Kp·(q_ref − q) + Kd·(q̇_ref − q̇) ]                (PD+G)

Why this and not control/trq.py
-------------------------------
`trq.py` was jerky because it (a) had gravity compensation commented out, so an
integral term had to wind up to fight gravity; (b) fed back the *double-
differentiated* (very noisy) encoder acceleration; and (c) used large P+I gains
with no model feed-forward. This controller instead follows the paper authors'
`PD_GravityCompensation.py` — proven on this exact hardware — and:
  • feeds forward GRAVITY from a model (no integrator → no windup),
  • damps with the *measured* joint velocity (no noisy differentiation, no
    acceleration feedback term),
  • drives a smooth cubic reference that STARTS at the current pose, so the
    position error (and thus the command) begins near zero — no startup lurch,
  • commands directly in mA, the native unit of both G sources.

Gravity source (`--gravity`)
-----------------------------
  • model  (default): our identified URDF via Pinocchio, g = computeGeneralized
    Gravity(q) [Nm] → mA via 1/EFFORT_SCALE. This puts OUR identified model in
    the loop (the thesis point).
  • paper           : the paper authors' calculate_gravity(q) [already mA] —
    a cross-check / fallback (external/paper_model).
Both were shown to agree to ~5–10 % on the gravity benchmark (CHANGELOG
2026-06-13). `--gravity-scale` lets you trim the feed-forward (the static
holding current is ~0.6× of G — stiction supports the rest at rest — so 1.0 is
physically correct and conservative; lower only if the arm feels stiff at hold).

SAFETY
------
Current mode means the servo's position loop is OFF — the arm is only held by
our command. Start with the default **hold** mode (regulate to the current pose)
to confirm gravity comp before commanding a goal. On Ctrl-C the node switches the
arm back to **position** mode so the servo holds it rather than letting it fall.
Keep the e-stop within reach; gains and current limit are conservative defaults.

Prereqs: arm driver running, ROS + interbotix + Pinocchio sourced:
    ros2 launch interbotix_xsarm_control xsarm_control.launch.py robot_model:=vx300s
    source /opt/ros/humble/setup.bash

Usage
-----
    # gravity-comp HOLD (safest first bring-up): regulate to the current pose
    python3 control/pdg_control.py

    # move to a goal pose (rad), smooth 4 s ramp, our model's gravity
    python3 control/pdg_control.py --goal 0 -0.6 0.5 0 0.4 0 --ramp 4

    # cross-check with the paper's gravity vector
    python3 control/pdg_control.py --gravity paper
"""
import argparse
import math
import os
import sys
import time

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.srv import OperatingModes, RobotInfo

# Joint order + effort scaling — identical to sysid_feasible.py (kept inline so
# this control node stays light: only rclpy / numpy / pinocchio).
ARM_JOINTS = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]
N = 6
TORQUE_CONSTANT = 2.409                              # Nm/A (XM540-W270, geared)
MOTORS_PER_JOINT = np.array([1, 2, 2, 1, 1, 1])      # shoulder & elbow dual-motor
EFFORT_SCALE = (TORQUE_CONSTANT / 1000.0) * MOTORS_PER_JOINT   # mA → Nm
#   command mA = joint torque [Nm] / EFFORT_SCALE

# Default per-joint gains — the paper authors' PD_GravityCompensation values,
# tuned for the vx300s (mA/rad and mA/(rad/s)).
DEFAULT_KP = np.array([100.0, 5000.0, 4500.0, 1500.0, 700.0, 500.0])
DEFAULT_KD = np.array([5.0, 350.0, 200.0, 10.0, 30.0, 25.0])

DEFAULT_URDF = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "outputs", "urdf",
    "traj_run_200hz_20260612_131613__sysid_feasible-v1-5__cfg-a92e984c__"
    "phi_to_urdf-v1-1__cfg-3ef0a00c.urdf")
PAPER_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "external", "paper_model")


class CubicRef:
    """Vector cubic q_s→q_e over T s with zero velocity at both ends; holds q_e
    after T. Returns (q_ref, qd_ref). Smooth start/stop, no step in the error."""

    def __init__(self, q_s, q_e, T):
        self.q_s = np.asarray(q_s, float)
        self.q_e = np.asarray(q_e, float)
        self.d = self.q_e - self.q_s
        self.T = max(T, 1e-6)

    def __call__(self, t):
        if t >= self.T:
            return self.q_e.copy(), np.zeros(N)
        s = t / self.T
        q = self.q_s + (3 * s ** 2 - 2 * s ** 3) * self.d
        qd = (6 * s - 6 * s ** 2) / self.T * self.d
        return q, qd


def build_gravity(source, urdf, scale):
    """Return g(q_arm)→ command current [mA] (6,), in ARM_JOINTS order."""
    if source == "paper":
        sys.path.insert(0, PAPER_DIR)
        from Gravity_Compensation_Function import calculate_gravity   # already mA
        return lambda q: scale * np.asarray(calculate_gravity(*q), float)

    # 'model': our identified URDF via Pinocchio (Nm → mA).
    import pinocchio as pin
    if not hasattr(pin, "buildModelFromUrdf"):
        raise RuntimeError("real Pinocchio not found — source /opt/ros/humble/setup.bash")
    model = pin.buildModelFromUrdf(urdf)
    data = model.createData()
    idx_q = np.array([model.joints[model.getJointId(n)].idx_q for n in ARM_JOINTS])
    idx_v = np.array([model.joints[model.getJointId(n)].idx_v for n in ARM_JOINTS])

    def g(q_arm):
        q = pin.neutral(model)
        q[idx_q] = q_arm
        gv = pin.computeGeneralizedGravity(model, data, q)
        return scale * gv[idx_v] / EFFORT_SCALE

    return g


class PDGController(Node):
    def __init__(self, args):
        super().__init__("pdg_control")
        self.robot = args.robot_model
        self.log = self.get_logger()

        self.kp = DEFAULT_KP.copy()
        self.kd = DEFAULT_KD.copy()
        self.ulim = float(args.current_limit)
        self.ramp = float(args.ramp)
        self.soft = float(args.soft_start)
        self.goal_arg = np.array(args.goal, float) if args.goal else None

        self.gravity = build_gravity(args.gravity, args.urdf, args.gravity_scale)
        self.log.info(f"gravity source: {args.gravity} (scale {args.gravity_scale})")

        self.q = None
        self.qd = None
        self.name_to_idx = None
        self.ref = None
        self.t0 = None

        self.pub = self.create_publisher(
            JointGroupCommand, f"/{self.robot}/commands/joint_group", 10)
        self.sub = self.create_subscription(
            JointState, f"/{self.robot}/joint_states", self._joint_cb, 10)

        self._set_mode("current")
        self.timer = self.create_timer(0.005, self._control_step)   # 200 Hz
        self.log.info("PD+G controller running (200 Hz). Ctrl-C to stop safely.")

    # ---- joint feedback: map by NAME into ARM_JOINTS order ----
    def _joint_cb(self, msg):
        if self.name_to_idx is None:
            self.name_to_idx = [msg.name.index(j) for j in ARM_JOINTS]
        idx = self.name_to_idx
        self.q = np.array([msg.position[i] for i in idx])
        self.qd = np.array([msg.velocity[i] for i in idx])

    # ---- operating-mode service ----
    def _set_mode(self, mode):
        cli = self.create_client(OperatingModes, f"/{self.robot}/set_operating_modes")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.log.info("waiting for set_operating_modes …")
        req = OperatingModes.Request()
        req.cmd_type, req.name, req.mode = "group", "arm", mode
        req.profile_type, req.profile_velocity, req.profile_acceleration = "time", 0, 0
        fut = cli.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        self.log.info(f"arm → {mode} mode")

    # ---- 200 Hz control step ----
    def _control_step(self):
        if self.q is None:
            return
        now = time.time()
        if self.ref is None:                      # first valid state: build reference
            self.t0 = now
            goal = self.goal_arg if self.goal_arg is not None else self.q.copy()
            self.ref = CubicRef(self.q.copy(), goal, self.ramp)
            mode = "HOLD current pose" if self.goal_arg is None else f"goal {np.round(goal,2).tolist()}"
            self.log.info(f"reference: {mode} over {self.ramp:.1f}s from {np.round(self.q,2).tolist()}")

        t = now - self.t0
        q_ref, qd_ref = self.ref(t)
        g = self.gravity(self.q)
        u = g + self.kp * (q_ref - self.q) + self.kd * (qd_ref - self.qd)

        # Soft-start: ramp the whole command 0→1 over self.soft s to avoid a jolt
        # at the position-mode → current-mode transition.
        if self.soft > 0 and t < self.soft:
            u *= t / self.soft

        u = np.clip(u, -self.ulim, self.ulim)
        cmd = JointGroupCommand(name="arm", cmd=u.tolist())
        self.pub.publish(cmd)

    # ---- safe shutdown ----
    def safe_stop(self):
        try:
            self.pub.publish(JointGroupCommand(name="arm", cmd=[0.0] * N))
            self._set_mode("position")            # servo holds the pose; arm won't fall
        except Exception as e:
            self.log.error(f"safe_stop failed: {e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-model", default="vx300s")
    ap.add_argument("--gravity", choices=["model", "paper"], default="model",
                    help="gravity feed-forward source (default: our identified model)")
    ap.add_argument("--urdf", default=DEFAULT_URDF, help="URDF for --gravity model")
    ap.add_argument("--gravity-scale", type=float, default=1.0,
                    help="scale on the gravity feed-forward (1.0 = physical)")
    ap.add_argument("--goal", type=float, nargs=6, default=None,
                    metavar=("WAIST", "SHOULDER", "ELBOW", "FOREARM", "WRIST_A", "WRIST_R"),
                    help="goal joint angles [rad]; omit to HOLD the current pose")
    ap.add_argument("--ramp", type=float, default=4.0, help="cubic ramp time to goal [s]")
    ap.add_argument("--soft-start", type=float, default=0.5,
                    help="ramp the whole command 0→1 over this many s at startup")
    ap.add_argument("--current-limit", type=float, default=2000.0,
                    help="per-joint command clamp [mA] (paper used 3200)")
    args = ap.parse_args()

    rclpy.init()
    node = PDGController(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.log.info("Ctrl-C — switching back to position mode …")
    finally:
        node.safe_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
