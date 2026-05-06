import signal

import rclpy
from rclpy.node import Node

from std_msgs.msg import Bool
from sensor_msgs.msg import JointState
from interbotix_xs_msgs.msg import JointGroupCommand
from interbotix_xs_msgs.srv import OperatingModes, RobotInfo

#import pinocchio as pin
import numpy as np
import time


class CubicPoly:

    def __init__(self, p_s, p_e, T):
        self.delta = p_e - p_s

        self.a_0 = p_s
        self.a_1 = 3 / T ** 2
        self.a_2 = 2 / T ** 3

        self.b_1 = 6 / T ** 2
        self.b_2 = 6 / T ** 3

        self.c_1 = 6 / T ** 2
        self.c_2 = 12 / T ** 3

    def __call__(self, t):
        t_pow2 = t ** 2
        t_pow3 = t ** 3
        p = self.a_0    +   (self.a_1 * t_pow2   - self.a_2 * t_pow3) * self.delta
        v =                 (self.b_1 * t        - self.b_2 * t_pow2) * self.delta
        a =                 (self.c_1            - self.c_2 * t     ) * self.delta
        return p, v, a


class ToTargetAndBack:

    def __init__(self, p_s, p_e, T):
        self.traj_s = CubicPoly(p_s, p_e, T)
        self.traj_e = CubicPoly(p_e, p_s, T)
        self.T = T
        self.p_s = p_s
        self.p_e = p_e

    def __call__(self, t):
        if t <= self.T:
            return self.traj_s(t)
        elif t > self.T and t <= 2 * self.T:
            return self.traj_e(t - self.T)
        else:
            O_n = np.zeros_like(self.p_s)
            return self.p_s, O_n, O_n


# Override default SIGINT handling before spin starts

class InterbotixCurrentControl(Node):

    def __init__(self):
        super().__init__('interbotix_current_control')
        self.robot_name = 'vx300s'
        logger = self.get_logger()
        self.t_start = time.time()
        # Publisher for joint group commands
        self.cmd_pub = self.create_publisher(
            JointGroupCommand,
            f'/{self.robot_name}/commands/joint_group',
            10
        )
        if not self.is_current_mode():
            self.set_current_mode()
        else:
            logger.info("Robot mode is already set to current mode")
        # Timer to send currents periodically
        self.timer = self.create_timer(0.004, self.send_current_command)
        self.sub = self.create_subscription(
            JointState,
            f"{self.robot_name}/joint_states",
            self.set_position,
            10
        )
        self.pos = None
        self.vel = None

        self.int_err_pos = 0.0
        self.int_err_vel = 0.0
        self.v_prev = None
        self.acc_prev = None

        self.t_prev = None
        self.logged_data = []

        #p_urdf = "/home/berwul/interbotix_ws/src/fnl_cnl/helpers/vx300s_new.urdf"
        #self.model = pin.buildModelFromUrdf(str(p_urdf))
        #self.data = self.model.createData()

        mask = [True] * 6
        # self.pos_P_gain = np.r_[
        #     8_000,
        #     8_000,
        #     6_000
        # ][mask]
        # self.pos_I_gain = np.r_[
        #     00.0,
        #     00.0,
        #     00.0
        # ][mask]
        # self.vel_P_gain = np.r_[
        #     600,
        #     600.0,
        #     200.0
        # ][mask]
        # self.vel_I_gain = np.r_[
        #     0.0,
        #     0.0, # 1200.0,
        #     0.0, # 1200.0
        # ][mask]
        # self.acc_P_gain = np.r_[
        #     10.0,
        #     10.0,
        #     10.0
        # ][mask]
        #
        #
        #

        mask = [True] * 6
        # mask = [True, True, True, False, True, False]

        # mask[-3] = True

        self.pos_P_gain = np.r_[
            5_000,
            5_000,
            5_000,
            10_000,
            5_000,
            1_000
        ][mask]
        self.pos_I_gain = np.r_[
            5_000,
            5000.0,
            5000.0,
            0.0,
            5_000,
            0.0
        ][mask]
        self.vel_P_gain = np.r_[
            600,
            600,
            600,
            100,
            600,
            10.0,
        ][mask]
        self.vel_I_gain = np.r_[
            0.0,
            0.0,
            0.0,  # 1200.0,
            0.0,  # 1200.0
            0.0,
            0.0,
        ][mask]
        self.acc_P_gain = np.r_[
            5.0,
            5.0,
            5.0,
            5.0,
            5.0,
            1.0
        ][mask]

        self.indxs = np.r_[0, 1, 2, 3, 4, 5][mask]
        self.p_goal = np.r_[-0.5, 0.0, 0.0, 0.5, 0.0, 0.5][mask]

    def set_position(self, msg):
        self.pos = np.r_[msg.position]
        self.vel = np.r_[msg.velocity]
        self.effort = np.r_[msg.effort]

    def is_current_mode(self):
        logger = self.get_logger()
        mode_client = self.create_client(
            RobotInfo,
            f'/{self.robot_name}/get_robot_info'
        )
        cnt = 0
        while not mode_client.wait_for_service(timeout_sec=1.0):
            logger.info(f'Waiting for robot info service... {cnt}')
            cnt += 1
        req = RobotInfo.Request()
        req.cmd_type = 'group'
        req.name = 'arm'
        future = mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        result = future.result()
        logger.info(f'Robot mode {result.mode}')
        return result.mode == "current"

    def set_current_mode(self):
        logger = self.get_logger()
        mode_client = self.create_client(
            OperatingModes,
            f'/{self.robot_name}/set_operating_modes'
        )
        cnt = 0
        while not mode_client.wait_for_service(timeout_sec=1.0):
            logger.info(f'Waiting for operating mode service... {cnt}')
            cnt += 1
        req = OperatingModes.Request()
        req.cmd_type = 'group'
        req.name = 'arm'
        req.mode = 'current'
        req.profile_type = 'time'
        req.profile_velocity = 0
        req.profile_acceleration = 0
        future = mode_client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        logger.info('Arm switched to current mode')

    def send_current_command(self):
        if self.pos is None or self.vel is None:
            return
        t_now = time.time()

        indxs = self.indxs
        n = len(indxs)

        if self.t_prev is None:
            self.t_prev = t_now
            p_start = self.pos[indxs]
            self.ref = ToTargetAndBack(p_s=p_start, p_e=self.p_goal, T=5.0)
            self.t_start = t_now

        if self.v_prev is None:
            self.p_prev = self.pos[indxs]
            self.v_prev = np.zeros(n)
            self.acc_prev = np.zeros(n)

        t = t_now  - self.t_start
        dt = t_now - self.t_prev
        msg = JointGroupCommand()
        msg.name = 'arm'

        p_curr = self.pos[indxs]
        v_curr_ = self.vel[indxs]
        effort_curr = self.effort[indxs]


        p_ref, v_ref, acc_ref = self.ref(t)

        # https://emanual.robotis.com/docs/en/dxl/x/xm540-w270/

        err_pos = p_ref - p_curr
        self.int_err_pos += err_pos * dt

        v_est = (p_curr - self.p_prev) / dt if dt > 0 else np.zeros(n)
        tau = 0.025
        alpha = (dt / (tau + dt))
        v_curr = alpha * v_est + (1 - alpha) * self.v_prev

        err_vel = v_ref - v_curr
        self.int_err_vel += err_vel * dt

        acc_est = (v_curr - self.v_prev) / dt if dt > 0 else np.zeros(n)
        tau = 0.1
        alpha = (dt / (tau + dt))
        acc_curr = alpha * acc_est + (1 - alpha) * self.acc_prev

        # tau_f = 0.025
        # acc_curr = self.acc_prev * (dt / (dt + tau_f)) + (v_curr - self.v_prev) / (dt + tau_f)

        err_acc = acc_ref - acc_curr
        self.int_err_vel += err_vel * dt

        current_cmd = (
            self.pos_P_gain * err_pos
            +
            self.pos_I_gain * self.int_err_pos
            +
            self.vel_P_gain * err_vel
            +
            self.vel_I_gain * self.int_err_vel
            +
            self.acc_P_gain * err_acc
        )

        g = pin.computeGeneralizedGravity(self.model, self.data, self.pos[:6])
        # current_cmd = current_cmd

        # DATA SHEET:
        # ----
        # torque_constants_ = 2.409 [Nm/A]
        # current_units_ = 2.69 ** (1e-3)
        # ----
        # cmd = tau / torque_constant
        # cmd = current_units_ / current_units
        # ----


        # + (g[indxs] * (1.0 /  2.3) * 150.0)
        # + g[indxs] * (1000. / 2.3) * 0.01


        # M = pin.crba(model, data, q)
        # C = pin.computeCoriolisMatrix(model, data, q, q_d)
        # c = C @ q_d

        # kt = 154.3159866732714
        # current_cmd = current_cmd + g[indxs]

        u_lim = 2000.
        current_cmd = np.maximum(np.minimum(current_cmd, u_lim), -u_lim)

        currs = np.zeros(6)
        current_mA = current_cmd
        currs[indxs] = current_mA
        msg.cmd = currs.tolist()
        self.cmd_pub.publish(msg)

        self.t_prev = t_now
        self.p_prev = p_curr
        self.v_prev = v_curr
        self.acc_prev = acc_curr

        t_compute = time.time() - t_now
        d = np.r_[p_curr, p_ref, v_curr, v_ref, acc_curr, acc_ref, current_cmd, effort_curr, t_compute, t]
        self.logged_data.append(d)

    def send_emergency_stop(self):
        print('Emergency stop')
        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = [
            0.0,  # waist
            0.0,  # shoulder
            0.0,  # elbow
            0.0,  # forearm_roll
            0.0,  # wrist_angle
            0.0  # wrist_rotate
        ]
        self.cmd_pub.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = InterbotixCurrentControl()
    shutdown_requested = False
    def sigint_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            return  # prevent double Ctrl-C handling
        shutdown_requested = True
        print("\nSIGINT caught: sending emergency stop...")
        # Publish emergency stop while ROS context is still valid
        for i in range(10):
            node.send_emergency_stop()
        # Give DDS time to flush outgoing message
        time.sleep(0.2)
        # Stop executor cleanly
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    # Override default SIGINT handling before spin starts
    signal.signal(signal.SIGINT, sigint_handler)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()
        logged_data = np.vstack(node.logged_data)
        np.save("/home/berwul/interbotix_ws/src/fnl_cnl/helpers/logged_data.npy", logged_data)


if __name__ == '__main__':
    main()
