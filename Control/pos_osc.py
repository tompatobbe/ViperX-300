import rclpy
from rclpy.node import Node
import numpy as np
from interbotix_xs_msgs.msg import JointGroupCommand

class ArmSineWaveCommander(Node):
    def __init__(self):
        super().__init__('sine_wave_commander')
        
        # Publisher for the 'arm' group
        self.publisher_ = self.create_publisher(
            JointGroupCommand, 
            '/vx300s/commands/joint_group', 
            10
        )
        
        # --- Configuration ---
        self.amplitude = 0.8    # Radians (Approx 17 degrees - safer for testing)
        self.frequency = 0.2    # Hz (Cycles per second)
        self.offset = 0.0       
        
        # Base positions: [waist, shoulder, elbow, forearm, wrist_angle, wrist_rotate]
        # These are the "center" points for your motion.
        self.static_joints = [0.0, -0.5, 0.5, 0.0, 0.5, 0.0] 

        # Timer: 0.01s = 100Hz control loop
        self.timer = self.create_timer(0.01, self.timer_callback)
        
        # Time tracking
        self.start_time = self.get_clock().now()
        
        self.get_logger().info('Sine Wave Node Started. Oscillating the WAIST.')

    def timer_callback(self):
        # 1. Calculate elapsed time
        now = self.get_clock().now()
        diff = now - self.start_time
        t = diff.nanoseconds / 1e9 
        
        # 2. Calculate the sine wave value
        # Formula: y = A * sin(2 * pi * f * t) + offset
        oscillation = self.amplitude * np.sin(2 * np.pi * self.frequency * t) + self.offset
        
        # 3. Prepare the command list
        # We copy the static list so we don't overwrite the original defaults permanently
        commands = list(self.static_joints)
        
        # --- CHANGE MOTOR HERE ---
        # Index 0 = Waist
        # Index 1 = Shoulder
        # Index 2 = Elbow
        commands[4] = oscillation 
        # -------------------------

        # 4. Create and publish the message
        msg = JointGroupCommand()
        msg.name = 'arm'
        msg.cmd = commands
        
        self.publisher_.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    sine_commander = ArmSineWaveCommander()
    try:
        rclpy.spin(sine_commander)
    except KeyboardInterrupt:
        sine_commander.get_logger().info('Shutting down...')
    finally:
        sine_commander.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()