import numpy as np

def dh_matrix(theta_deg: float, d: float, a: float, alpha_deg: float) -> np.ndarray:
    """
    Compute one standard DH homogeneous transformation matrix (4×4).

    T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)

    Expanded:

    ┌ cθ -sθ·cα sθ·sα a·cθ ┐

    │ sθ cθ·cα -cθ·sα a·sθ │

    │ 0 sα cα d │

    └ 0 0 0 1 ┘
    """
    θ = np.radians(theta_deg)
    α = np.radians(alpha_deg)

    cθ, sθ = np.cos(θ), np.sin(θ)
    cα, sα = np.cos(α), np.sin(α)

    return np.array([
        [cθ, -sθ * cα, sθ * sα, a * cθ],
        [sθ,  cθ * cα, -cθ * sα, a * sθ],
        [0,    sα,       cα,       d     ],
        [0,    0,        0,       1     ]
    ])

def forward_kinematics(dh_params: list) -> np.ndarray:
    """
    Compute the overall transformation matrix for a 6-DOF manipulator
    given the DH parameters.

    :param dh_params: List of [θ_deg, d, a, α_deg] for each joint
    :return: Overall transformation matrix (4x4)
    """
    # Initialize the homogeneous transformation matrix with identity
    T_total = np.eye(4)

    for params in dh_params:
        theta_deg, d, a, alpha_deg = params
        T_link = dh_matrix(theta_deg, d, a, alpha_deg)
        T_total = np.dot(T_total, T_link)

    return T_total

# Define the DH parameters for your 6-DOF manipulator
dh_params = [
    [0, 0.2, 0.1, -90],  # Link 1
    [90, 0, 0, 0],      # Link 2
    [-90, 0.3, 0.4, 90], # Link 3
    [90, 0, 0.5, 0],     # Link 4
    [-90, 0.6, 0, -90],  # Link 5
    [0, 0.7, 0, 0]       # Link 6
]

# Compute the forward kinematics (overall transformation matrix)
T_total = forward_kinematics(dh_params)

print("Overall Transformation Matrix:")
print(T_total)

# Optionally, you can extract the end-effector position/pose from T_total
end_effector_position = T_total[:3, 3]
end_effector_orientation = T_total[:3, :3]

print("\nEnd-Effector Position (x, y, z):")
print(end_effector_position)

# You can also print the orientation matrix if needed
print("\nEnd-Effector Orientation Matrix:")
print(end_effector_orientation)