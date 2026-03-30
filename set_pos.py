import rclpy
import sys
import argparse
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand

class ArmPositionCommand(Node):
    def __init__(self):
        super().__init__('manual_arm_commander')
        self.publisher_ = self.create_publisher(
            JointGroupCommand, 
            '/vx300s/commands/joint_group', 
            10
        )
        # Give the graph a moment to register the publisher
        self.get_logger().info('Arm Commander Node Started')

    def send_position(self, positions):
        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = [float(p) for p in positions]
        
        self.get_logger().info(f'Sending Joint Positions: {msg.cmd}')
        self.publisher_.publish(msg)

def main():
    # 1. Setup Argument Parser
    parser = argparse.ArgumentParser(description='Send joint positions to vx300s')
    parser.add_argument('joints', metavar='J', type=float, nargs=6,
                        help='6 joint positions in radians (waist, shoulder, elbow, forearm, wrist_angle, wrist_rotate)')
    
    args = parser.parse_args()

    # 2. Initialize ROS 2
    rclpy.init()
    commander = ArmPositionCommand()

    # Small sleep to ensure connection to the ROS graph
    import time
    time.sleep(0.5)

    # 3. Execution
    commander.send_position(args.joints)
    
    # Final sleep to ensure the packet leaves the network buffer
    time.sleep(0.1)
    
    commander.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()