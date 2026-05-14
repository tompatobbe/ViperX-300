import rclpy
from rclpy.node import Node
import numpy as np
import time # Added for the shutdown delay
from interbotix_xs_msgs.msg import JointGroupCommand, JointSingleCommand

class ArmHybridCommander(Node):
    def __init__(self):
        super().__init__('arm_hybrid_commander')
        
        self.waist_pub = self.create_publisher(JointSingleCommand, '/vx300s/commands/joint_single', 10)
        self.arm_pub = self.create_publisher(JointGroupCommand, '/vx300s/commands/joint_group', 10)
        
        self.amplitude = 0.5    
        self.frequency = 0.1    
        
        self.timer = self.create_timer(0.01, self.timer_callback)
        self.start_time = self.get_clock().now()
        self.get_logger().info('Hybrid Node Started. Press Ctrl+C to stop.')

    def timer_callback(self):
        now = self.get_clock().now()
        t = (now - self.start_time).nanoseconds / 1e9  
        
        # Waist Velocity Command
        waist_msg = JointSingleCommand(name='waist')
        waist_msg.cmd = self.amplitude * np.cos(2 * np.pi * self.frequency * t)
        self.waist_pub.publish(waist_msg)
        
        # Rest of Arm Position Command
        arm_msg = JointGroupCommand(name='arm', cmd=[0.0]*6)
        self.arm_pub.publish(arm_msg)

    def stop_arm(self):
        """Explicitly send zero velocity to the waist and stop the arm."""
        self.get_logger().info('Sending stop command to waist...')
        stop_msg = JointSingleCommand(name='waist', cmd=0.0)
        
        # Publish multiple times to ensure the driver catches it during shutdown
        for _ in range(5):
            self.waist_pub.publish(stop_msg)
            time.sleep(0.05) 

def main(args=None):
    rclpy.init(args=args)
    commander = ArmHybridCommander()
    try:
        rclpy.spin(commander)
    except KeyboardInterrupt:
        commander.get_logger().info('Keyboard Interrupt detected.')
    finally:
        # This block runs regardless of how the script ends
        commander.stop_arm()
        commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()