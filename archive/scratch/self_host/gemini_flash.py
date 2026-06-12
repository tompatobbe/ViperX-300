import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --- (Keep your dh_matrix function as is) ---
def dh_matrix(theta_deg: float, d: float, a: float, alpha_deg: float) -> np.ndarray:
    """
    Compute one standard DH homogeneous transformation matrix (4x4).
    T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)

    Expanded:
    ┌ cθ -sθ·cα sθ·sα a·cθ ┐
    │ sθ cθ·cα -cθ·sα a·sθ │
    │ 0 sα cα d │
    └ 0 0 0 1 ┘

    Args:
        theta_deg (float): Joint angle (or total theta for the frame) in degrees.
        d (float): Link offset.
        a (float): Link length.
        alpha_deg (float): Twist angle in degrees.

    Returns:
        np.ndarray: The 4x4 homogeneous transformation matrix.
    """
    theta = np.radians(theta_deg)
    alpha = np.radians(alpha_deg)

    c_theta, s_theta = np.cos(theta), np.sin(theta)
    c_alpha, s_alpha = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ c_theta, -s_theta * c_alpha,  s_theta * s_alpha,  a * c_theta ],
        [ s_theta,  c_theta * c_alpha, -c_theta * s_alpha,  a * s_theta ],
        [ 0,        s_alpha,           c_alpha,            d           ],
        [ 0,        0,                 0,                  1           ],
    ])

# --- RoboticManipulator Class ---
class RoboticManipulator:
    def __init__(self, dh_table: np.ndarray):
        if dh_table.shape[1] != 4:
            raise ValueError("DH table must have 4 columns: [d, a, alpha_deg, theta_offset_deg]")
        self.dh_table = dh_table
        self.num_dof = dh_table.shape[0]

    def forward_kinematics(self, joint_angles_deg: list) -> tuple[np.ndarray, list[np.ndarray]]:
        if len(joint_angles_deg) != self.num_dof:
            raise ValueError(f"Expected {self.num_dof} joint angles, but got {len(joint_angles_deg)}")

        current_transform = np.eye(4)
        transforms_0_i = [current_transform]

        for i in range(self.num_dof):
            d, a, alpha_deg, theta_offset_deg = self.dh_table[i]
            actual_theta_deg = joint_angles_deg[i] + theta_offset_deg

            T_prev_curr = dh_matrix(actual_theta_deg, d, a, alpha_deg)
            current_transform = current_transform @ T_prev_curr
            transforms_0_i.append(current_transform)

        return current_transform, transforms_0_i[1:]

    # --- Corrected plot_robot method ---
    def plot_robot(self, joint_angles_deg: list, ax: Axes3D = None, show_frames: bool = True):
        """
        Plots the robot in its current configuration.
        Args:
            joint_angles_deg: List of joint angles in degrees.
            ax: Optional matplotlib 3D axis to plot on. If None, a new figure/axis is created.
            show_frames: If True, plots coordinate frames (X, Y, Z axes) at each joint's origin.
        """
        _, transforms_0_i = self.forward_kinematics(joint_angles_deg)

        if ax is None:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X [m]')
            ax.set_ylabel('Y [m]')
            ax.set_zlabel('Z [m]')
            ax.set_title('Robotic Manipulator Simulation')
            ax.view_init(elev=20, azim=-60)
            # ax.set_aspect('equal') # <--- REMOVE OR COMMENT OUT THIS LINE


        joint_origins = [np.array([0, 0, 0])]
        for T in transforms_0_i:
            joint_origins.append(T[:3, 3])

        for i in range(len(joint_origins) - 1):
            p1 = joint_origins[i]
            p2 = joint_origins[i+1]
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    'ro-', linewidth=4, markersize=8, label='Links' if i == 0 else "")

        if show_frames:
            self._plot_frame(ax, np.eye(4), "0")
            for i, T in enumerate(transforms_0_i):
                self._plot_frame(ax, T, str(i + 1))

        # --- MODIFIED AUTO-SCALING FOR EQUAL ASPECT ---
        all_coords = np.array(joint_origins)
        if all_coords.size > 0:
            # Determine the overall min and max for all x, y, z coordinates
            min_vals = np.min(all_coords, axis=0)
            max_vals = np.max(all_coords, axis=0)

            # Calculate the range for each dimension
            ranges = max_vals - min_vals

            # Find the largest range across all dimensions
            max_dimension_range = np.max(ranges)

            # Find the center of the robot's occupied space
            center = (min_vals + max_vals) / 2.0

            # Calculate a buffer based on the largest range
            buffer = max_dimension_range * 0.2

            # Use the max_dimension_range and center to set equal limits
            # for all axes to create a cubic bounding box, effectively
            # giving an 'equal' aspect ratio visually.
            limit_half_span = (max_dimension_range / 2.0) + buffer

            ax.set_xlim([center[0] - limit_half_span, center[0] + limit_half_span])
            ax.set_ylim([center[1] - limit_half_span, center[1] + limit_half_span])
            ax.set_zlim([center[2] - limit_half_span, center[2] + limit_half_span])
        else: # Default limits if robot has no links (unlikely)
            ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([0, 1])

        return ax

    def _plot_frame(self, ax: Axes3D, T: np.ndarray, label: str, length: float = 0.15):
        """
        Helper function to plot a coordinate frame (X, Y, Z axes) at a given transformation.
        """
        origin = T[:3, 3]

        x_axis_dir = T[:3, 0]
        y_axis_dir = T[:3, 1]
        z_axis_dir = T[:3, 2]

        x_axis_end = origin + x_axis_dir * length
        y_axis_end = origin + y_axis_dir * length
        z_axis_end = origin + z_axis_dir * length

        ax.plot([origin[0], x_axis_end[0]], [origin[1], x_axis_end[1]], [origin[2], x_axis_end[2]], 'r-', linewidth=1)
        ax.plot([origin[0], y_axis_end[0]], [origin[1], y_axis_end[1]], [origin[2], y_axis_end[2]], 'g-', linewidth=1)
        ax.plot([origin[0], z_axis_end[0]], [origin[1], z_axis_end[1]], [origin[2], z_axis_end[2]], 'b-', linewidth=1)

        ax.text(origin[0] + length * 1.1, origin[1], origin[2], f'F{label}', color='black', fontsize=9, weight='bold')

# --- (Keep your example usage block as is) ---
if __name__ == "__main__":
    dh_params_6dof = np.array([
        [0.1,  0.0,  90.0,   0.0],
        [0.0,  0.5,   0.0, -90.0],
        [0.0,  0.4,   0.0,   0.0],
        [0.1,  0.0,  90.0,   0.0],
        [0.1,  0.0, -90.0,   0.0],
        [0.1,  0.0,   0.0,   0.0],
    ])

    robot = RoboticManipulator(dh_params_6dof)

    print("\n--- Robot at Home Position (q=[0,0,0,0,0,0]) ---")
    q_home = [0, 0, 0, 0, 0, 0]
    T_0_N_home, _ = robot.forward_kinematics(q_home)
    print("End-effector position (T_0_6) at Home:\n", T_0_N_home)

    ax1 = robot.plot_robot(q_home, show_frames=True)
    ax1.set_title('Robot at Home Position')

    print("\n--- Robot at Custom Configuration 1 (q=[30, 45, -30, 0, 60, 0]) ---")
    q_config1 = [30, 45, -30, 0, 60, 0]
    T_0_N_config1, _ = robot.forward_kinematics(q_config1)
    print("End-effector position (T_0_6) at Config 1:\n", T_0_N_config1)

    ax2 = robot.plot_robot(q_config1, show_frames=True)
    ax2.set_title('Robot at Custom Configuration 1')

    print("\n--- Robot Reaching Forward (q=[0, -45, 90, 0, 0, 0]) ---")
    q_reach_forward = [0, -45, 90, 0, 0, 0]
    T_0_N_reach, _ = robot.forward_kinematics(q_reach_forward)
    print("End-effector position (T_0_6) when Reaching Forward:\n", T_0_N_reach)

    ax3 = robot.plot_robot(q_reach_forward, show_frames=True)
    ax3.set_title('Robot Reaching Forward')

    plt.tight_layout()
    plt.show()