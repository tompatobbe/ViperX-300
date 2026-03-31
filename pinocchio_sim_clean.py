import pinocchio as pin
import numpy as np
import os

# Get the directory of the current script for relative file paths
script_dir = os.path.dirname(os.path.abspath(__file__))
urdf_path = os.path.join(script_dir, 'vx300s.urdf')

# Load the model with root joint (required for most URDF files)
model = pin.buildModelFromUrdf(urdf_path, root_joint=pin.JointModelFreeFlyer())
print(f"Model loaded with {model.njoints} joints and {model.nq} DOF\n")

# Create data for the model
data = model.createData()

# Set joint positions to neutral (all zeros for actuated joints)
q = pin.neutral(model)
print("=" * 60)
print("NEUTRAL CONFIGURATION")
print("=" * 60)

# Compute forward kinematics and update frame placements
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

# Get the end-effector pose
end_effector_id = model.getFrameId('/ee_gripper_link')
ee_pose = data.oMf[end_effector_id]
ee_position_neutral = ee_pose.translation.copy()  # Store as copy to avoid reference issues
print(f"\nEnd-effector position: {ee_position_neutral}")
print(f"End-effector rotation:\n{ee_pose.rotation}")

# Show positions along the arm kinematic chain
print("\nFrame positions along the arm:")
for frame_name in ['/base_link', '/shoulder_link', '/upper_arm_link', '/upper_forearm_link', 
                   '/lower_forearm_link', '/wrist_link', '/gripper_link', '/ee_gripper_link']:
    fid = model.getFrameId(frame_name)
    if fid >= 0:
        print(f"  {frame_name:25s}: {data.oMf[fid].translation}")

# Modify joint angles (skip indices 0-6 which are the free flyer)
print("\n" + "=" * 60)
print("MODIFIED CONFIGURATION")
print("=" * 60)
print("\nChanges made:")
print("  q[7]  (waist joint):   0 -> 0.5 rad")
print("  q[8]  (shoulder joint): 0 -> 0.3 rad")

q[7] = 0.5   # waist joint
q[8] = 0.3   # shoulder joint

# Recompute forward kinematics with new configuration
pin.forwardKinematics(model, data, q)
pin.updateFramePlacements(model, data)

ee_pose_new = data.oMf[end_effector_id]
print(f"\nEnd-effector position: {ee_pose_new.translation}")
print(f"End-effector rotation:\n{ee_pose_new.rotation}")

print("\nFrame positions along the arm:")
for frame_name in ['/base_link', '/shoulder_link', '/upper_arm_link', '/upper_forearm_link', 
                   '/lower_forearm_link', '/wrist_link', '/gripper_link', '/ee_gripper_link']:
    fid = model.getFrameId(frame_name)
    if fid >= 0:
        print(f"  {frame_name:25s}: {data.oMf[fid].translation}")

# Compute and display Jacobian
print("\n" + "=" * 60)
print("JACOBIAN (end-effector velocity mapping)")
print("=" * 60)
jacobian = pin.computeFrameJacobian(model, data, q, end_effector_id)
print(f"Shape: {jacobian.shape} (6 DOF: 3 linear + 3 angular velocities, 15 joint velocities)")
print(f"\nJacobian:\n{jacobian}")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
position_change = ee_pose_new.translation - ee_position_neutral
print(f"\nEnd-effector position change: {position_change}")
print(f"Total displacement: {np.linalg.norm(position_change):.6f} m")
