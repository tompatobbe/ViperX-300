#!/usr/bin/env python3
import time
import rclpy
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS

bot  = InterbotixManipulatorXS(robot_model="vx300s", group_name="arm", gripper_name="gripper")
node = bot.core.robot_node

bot.arm.set_joint_positions([0,0,0,0,0,0], moving_time=3.0, blocking=True)

def move_and_print(positions, duration=1.0):
    bot.arm.set_joint_positions(positions, moving_time=duration, blocking=False)
    end = time.time() + duration
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.02)
        js = bot.core.robot_get_joint_states()
        print(js.position[list(js.name).index("elbow")])
        time.sleep(0.02)

move_and_print([0, 0,  0.5, 0, 0, 0])
move_and_print([0, 0, -0.5, 0, 0, 0])
move_and_print([0, 0,  0.0, 0, 0, 0])
