import pinocchio as pin
import numpy as np
import os

# Get the directory of the current script for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, 'vx300s.urdf')

# Load the model with root joint (required for most URDF files)
model = pin.buildModelFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
print(f"Model loaded with {model.njoints} joints and {model.nq} DOF")

# Debug: Print joint placement (transforms)
print("\nJoint placements:")
for i in range(1, model.njoints):
    placement = model.jointPlacements[i]
    print(f"Joint {i} ({model.names[i]}): pos={placement.translation}, rot=\n{placement.rotation}")

# Create data for the model
data = model.createData()

# Set joint positions (example: all joints at 0)
q = pin.neutral(model)
print(f"Neutral configuration: {q}")

# Compute forward kinematics
pin.forwardKinematics(model, data, q)
# Update frame placements after forward kinematics
pin.updateFramePlacements(model, data)

# Show joint poses (not frame poses)
print("\nJoint poses at neutral configuration (oMi):")
for i in range(1, min(5, model.njoints)):
    print(f"Joint {i} ({model.names[i]}): pos={data.oMi[i].translation}")

# Show frame poses
print("\nFrame poses at neutral configuration (oMf):")
for i in range(min(10, model.nframes)):
    print(f"Frame {i} ({model.frames[i].name}): pos={data.oMf[i].translation}")

# Get the end-effector pose (assuming last joint is end-effector)
end_effector_id = model.getFrameId('/ee_gripper_link')  # Adjust based on URDF
if end_effector_id == -1:
    end_effector_id = model.nframes - 1  # Fallback to last frame

ee_pose = data.oMf[end_effector_id]
print(f"End-effector pose:\n{ee_pose}")
print(f"End-effector position: {ee_pose.translation}")
print(f"End-effector rotation:\n{ee_pose.rotation}")

# Example: Set some joint angles and recompute
# Note: indices 0-6 are the free flyer (3 pos + 4 quat), indices 7+ are actuated joints
print("\nOriginal q:", q)
q[7] = 0.5   # waist joint
q[8] = 0.3   # shoulder joint
print("Updated q:", q)
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)
ee_pose_new = data.oMf[end_effector_id]
print(f"\nNew end-effector pose:\n{ee_pose_new}")
print(f"New end-effector position: {ee_pose_new.translation}")
print(f"New end-effector rotation:\n{ee_pose_new.rotation}")

# Show multiple frames along the arm to debug
print("\nFrame positions at neutral configuration:")
for frame_name in ['/base_link', '/shoulder_link', '/upper_arm_link', '/upper_forearm_link', '/lower_forearm_link', '/wrist_link', '/gripper_link', '/ee_gripper_link']:
    fid = model.getFrameId(frame_name)
    if fid >= 0:
        print(f"{frame_name}: {data.oMf[fid].translation}")

print("\nFrame positions after joint changes:")
for frame_name in ['/base_link', '/shoulder_link', '/upper_arm_link', '/upper_forearm_link', '/lower_forearm_link', '/wrist_link', '/gripper_link', '/ee_gripper_link']:
    fid = model.getFrameId(frame_name)
    if fid >= 0:
        print(f"{frame_name}: {data.oMf[fid].translation}")

# Compute Jacobian
jacobian = pin.computeFrameJacobian(model, data, q, end_effector_id)
print(f"Jacobian shape: {jacobian.shape}")
print(f"Jacobian:\n{jacobian}")