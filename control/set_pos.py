import argparse
import time
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS


def main():
    parser = argparse.ArgumentParser(description='Send joint positions to vx300s')
    parser.add_argument('joints', metavar='J', type=float, nargs=6,
                        help='6 joint positions in radians (waist shoulder elbow forearm wrist_angle wrist_rotate)')
    parser.add_argument('--moving-time', type=float, default=3.0,
                        help='Seconds to complete the move — longer = slower')
    parser.add_argument('--accel-time', type=float, default=0.5,
                        help='Acceleration ramp duration (s)')
    args = parser.parse_args()

    bot = InterbotixManipulatorXS(robot_model='vx300s', group_name='arm', gripper_name='gripper')

    # Mirror the sysid script: wait for ROS2 publisher connections to stabilise
    # before sending any command. 0.5 s is not enough; 3 s matches the time the
    # sysid script spends building its trajectory before its first move.
    time.sleep(3.0)

    print(f'Moving to {args.joints} over {args.moving_time} s …')
    bot.arm.set_joint_positions(
        args.joints,
        moving_time=args.moving_time,
        accel_time=args.accel_time,
        blocking=True,
    )
    print('Done.')


if __name__ == '__main__':
    main()
