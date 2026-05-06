import sys
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import rclpy

def main():
    # In ROS 2, we need to initialize the rclpy context first
    bot = InterbotixManipulatorXS(
        robot_model='vx300s',
        group_name='arm',
        gripper_name='gripper'
    )

    if (bot.arm.group_info.num_joints < 5):
        print("Error: Robot joints not detected. Is the driver running?")
        sys.exit(1)

    bot.arm.go_to_home_pose()
    bot.gripper.release()
    
    # Simple relative move
    bot.arm.set_single_joint_position(joint_name='waist', position=1.0)
    
    bot.arm.go_to_sleep_pose()
    bot.shutdown()

if __name__ == '__main__':
    main()