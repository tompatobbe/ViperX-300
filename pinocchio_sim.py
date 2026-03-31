import pinocchio as pin
import numpy as np
import os

# Get the directory of the current script for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, 'vx300s.urdf')

# Load the model with root joint (required for most URDF files)
model = pin.buildModelFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
print(f"Model loaded with {model.njoints} joints and {model.nq} DOF")

# Create data for the model
data = model.createData()

# Set joint positions (example: all joints at 0)
q = pin.neutral(model)
print(f"Neutral configuration: {q}")

# Compute forward kinematics
pin.forwardKinematics(model, data, q)

# Get the end-effector pose (assuming last joint is end-effector)
end_effector_id = model.getFrameId('/ee_gripper_link')  # Adjust based on URDF
if end_effector_id == -1:
    end_effector_id = model.nframes - 1  # Fallback to last frame

ee_pose = data.oMf[end_effector_id]
print(f"End-effector pose:\n{ee_pose}")

# Example: Set some joint angles and recompute
q[0] = 0.5  # First joint
q[1] = 0.3  # Second joint
pin.forwardKinematics(model, data, q)
ee_pose_new = data.oMf[end_effector_id]
print(f"New end-effector pose:\n{ee_pose_new}")

# Compute Jacobian
jacobian = pin.computeFrameJacobian(model, data, q, end_effector_id)
print(f"Jacobian shape: {jacobian.shape}")
print(f"Jacobian:\n{jacobian}")