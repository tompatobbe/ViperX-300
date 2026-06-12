import numpy as np

def dh_matrix(theta_deg, d, a, alpha_deg):
    """
    Compute one standard DH homogeneous transformation matrix (4×4).

    T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)

    Expanded:
    ┌  cθ  -sθ·cα   sθ·sα   a·cθ ┐
    │  sθ   cθ·cα  -cθ·sα   a·sθ │
    │   0     sα      cα      d   │
    └   0      0       0      1   ┘
    """
    θ = np.radians(theta_deg)
    α = np.radians(alpha_deg)
    cθ, sθ = np.cos(θ), np.sin(θ)
    cα, sα = np.cos(α), np.sin(α)

    return np.array([
        [cθ, -sθ*cα,  sθ*sα, a*cθ],
        [sθ,  cθ*cα, -cθ*sα, a*sθ],
        [0,     sα,     cα,    d ],
        [0,      0,      0,    1 ],
    ])

def forward_kinematics(dh_params):
    """
    Compute the forward kinematics for a robotic manipulator given DH parameters.

    dh_params: List of tuples (theta_deg, d, a, alpha_deg) representing each joint's DH parameters.
    """
    T = np.identity(4)  # Start with the identity transformation
    for params in dh_params:
        theta_deg, d, a, alpha_deg = params
        T_i = dh_matrix(theta_deg, d, a, alpha_deg)
        T = np.dot(T, T_i)  # Accumulate transformation matrices
    return T

# Example DH parameters for a 6-DOF manipulator
dh_params = [
    (0, 0, 1, np.pi/2),   # Joint 1: Rotational around z-axis
    (90, 0, 1, 0),        # Joint 2: Prismatic along z-axis
    (-90, 0, 1, np.pi/2), # Joint 3: Rotational around z-axis
    (0, 0, 1, -np.pi/2),   # Joint 4: Rotational around z-axis
    (0, 0, 1, np.pi/2),    # Joint 5: Rotational around z-axis
    (0, 0, 1, 0)          # Joint 6: Prismatic along z-axis
]

# Compute the forward kinematics for the entire manipulator
T = forward_kinematics(dh_params)
print("End-effector Transformation Matrix:\n", T)

# Extract position and orientation of the end-effector
position = T[:3, 3]
orientation = np.degrees(np.arctan2(T[1, 0], T[0, 0]))  # Convert to degrees if needed

print("End-effector Position:\n", position)
print("End-effector Orientation (in degrees):\n", orientation)