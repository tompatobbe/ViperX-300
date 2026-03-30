import rclpy
import argparse
import time
from rclpy.node import Node
from interbotix_xs_msgs.msg import JointGroupCommand

# Safe current limits for VX300S (in mA)
MAX_CURRENT_MA = 500 

class ArmCurrentCommander(Node):
    def __init__(self):
        super().__init__('arm_current_commander')
        self.publisher_ = self.create_publisher(
            JointGroupCommand, 
            '/vx300s/commands/joint_group', 
            10
        )
        self.get_logger().info('Current Control Node Started. Press Ctrl+C to zero out and exit.')

    def send_current(self, currents):
        # Safety Clamping
        clamped_currents = []
        for val in currents:
            clamped = max(-MAX_CURRENT_MA, min(MAX_CURRENT_MA, val))
            if clamped != val:
                self.get_logger().warn(f"Current {val}mA exceeded safety limit! Clamped to {clamped}mA")
            clamped_currents.append(float(clamped))

        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = clamped_currents
        
        self.publisher_.publish(msg)
        self.get_logger().info(f'Publishing Currents (mA): {clamped_currents}')

def main():
    parser = argparse.ArgumentParser(description='Send current commands to vx300s')
    parser.add_argument('currents', metavar='C', type=float, nargs=6,
                        help='6 current values in mA (waist, shoulder, elbow, forearm, wrist_angle, wrist_rotate)')
    
    args = parser.parse_args()
    rclpy.init()
    commander = ArmCurrentCommander()

    try:
        # Small delay for ROS discovery
        time.sleep(0.5)
        
        # Send the requested currents
        commander.send_current(args.currents)
        
        # Keep the node alive so it can catch the Interrupt
        # If you want it to run indefinitely until Ctrl+C, use rclpy.spin(commander)
        # For a one-shot command that stays safe, we use a loop:
        while rclpy.ok():
            rclpy.spin_once(commander, timeout_sec=0.1)

    except KeyboardInterrupt:
        commander.get_logger().info('Interrupt detected! Zeroing currents for safety...')
    
    finally:
        # Emergency Stop: Command 0.0mA to all 6 joints
        commander.send_current([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Give the message a moment to actually hit the bus before killing the node
        time.sleep(0.2)
        
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()