"""
3D Interactive Visualization of a 6-DOF Robotic Arm from DH Parameters
======================================================================

- Each DH frame is drawn at its origin with X (red), Y (green), Z (blue) axes.
- Links are drawn as gray segments between consecutive frame origins.
- One slider per joint lets you drive the arm in real time.
- End-effector position (in mm) is shown in the title.

Run:  python robot_arm_sim.py
Requires:  numpy, matplotlib  (use a GUI backend, e.g. TkAgg / Qt5Agg)
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# -----------------------------------------------------------------------------
# Robot geometry (mm) — adjust to match your real arm
# -----------------------------------------------------------------------------
L_1 = 126.75   # base height (waist -> shoulder)
L_2 = 305.94   
L_3 = 196.4   
L_4 = 103.62    
L_5 = 7.0    
L_6 = 136.58    

# -----------------------------------------------------------------------------
# DH Table:  [ alpha(deg),  a(mm),  d(mm),  theta_offset(rad) ]
# (theta_offset values containing np.pi are treated as radians)
# -----------------------------------------------------------------------------
DH_TABLE = np.array([
    #   α(°)   a(mm)   d(mm)        θ_off(rad)         # Joint
    [   0.0,   0.0,    L_1,         0.0           ],   # 1  Waist
    [ -90.0,   0.0,    0.0,        -0.437*np.pi   ],   # 2  Shoulder
    [   0.0,   L_2,    0.0,        -0.063*np.pi   ],   # 3  Elbow
    [ -90.0,   0.0,    L_3 + L_4,   0.0           ],   # 4  Wrist Rotate
    [  90.0,   0.0,    0.0,         0.0           ],   # 5  Wrist Pitch
    [ -90.0,   0.0,    0.0,         0.0           ],   # 6  Wrist Roll (EE)
])

JOINT_NAMES = ['Waist', 'Shoulder', 'Elbow', 'Wrist Rot', 'Wrist Pitch', 'Wrist Roll']

# Per-joint slider range in degrees (symmetric). 
JOINT_LIMITS_DEG = [
    (-180, 180),   # Waist
    (-106, 101),   # Shoulder
    (-101, 92),   # Elbow
    (-180, 180),   # Wrist Rotate
    (-101, 101),   # Wrist Pitch
    (-180, 180),   # Wrist Roll
]

# Physical link lengths that are zero in the DH table because the paper
# places multiple frames at the same origin.  Each entry (frame_idx, length_mm)
# adds a visual-only segment along that frame's z-axis for rendering.
VISUAL_OFFSETS = [
    (5, L_5),   # wrist pitch segment — moves with joint 5
    (6, L_6),   # wrist roll to end-effector — moves with joint 6
]


# -----------------------------------------------------------------------------
# DH math
# -----------------------------------------------------------------------------
def dh_matrix(theta_deg: float, d: float, a: float, alpha_deg: float) -> np.ndarray:
    """
    Compute one standard DH homogeneous transformation matrix (4×4).

    T = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)

    Expanded:
    ┌  cθ  -sθ·cα   sθ·sα   a·cθ ┐
    │  sθ   cθ·cα  -cθ·sα   a·sθ │
    │   0     sα      cα      d   │
    └   0      0       0      1   ┘
    """
    θ  = np.radians(theta_deg)
    α  = np.radians(alpha_deg)
    cθ, sθ = np.cos(θ), np.sin(θ)
    cα, sα = np.cos(α), np.sin(α)

    return np.array([
        [ cθ, -sθ*cα,  sθ*sα, a*cθ],
        [ sθ,  cθ*cα, -cθ*sα, a*sθ],
        [  0,     sα,     cα,    d ],
        [  0,      0,      0,    1 ],
    ])


def forward_kinematics(joint_angles_rad):
    """
    Returns a list of 4x4 frames in the world coordinate system:
        [F0 (base), F1, F2, F3, F4, F5, F6 (end effector)]
    `joint_angles_rad` is a length-6 array of joint values applied on top
    of each row's theta_offset.
    """
    frames = [np.eye(4)]
    T = np.eye(4)
    for i, (alpha_deg, a, d, theta_off_rad) in enumerate(DH_TABLE):
        theta_deg = np.degrees(joint_angles_rad[i] + theta_off_rad)
        T = T @ dh_matrix(theta_deg, d, a, alpha_deg)
        frames.append(T.copy())
    return frames


# -----------------------------------------------------------------------------
# Visualization
# -----------------------------------------------------------------------------
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
plt.subplots_adjust(left=0.0, right=0.65, bottom=0.05, top=0.98)

N_JOINTS = len(DH_TABLE)

AXIS_LEN = 45.0   # length of each frame's xyz axes (mm)
REACH = (L_1 + L_2 + L_3 + L_4) * 1.15

# Persistent view across redraws
view_state = {'elev': 22, 'azim': 35}


def draw(joint_angles_rad):
    # remember user's current view
    view_state['elev'] = ax.elev
    view_state['azim'] = ax.azim
    ax.clear()

    frames = forward_kinematics(joint_angles_rad)
    origins = np.array([T[:3, 3] for T in frames])

    # Build visual link chain for rendering.
    # Frames that share their position with the previous frame (d=0, a=0) are
    # skipped so they don't produce zero-length segments.  Visual extensions in
    # VISUAL_OFFSETS are appended serially from the current chain tip along
    # each frame's z-axis, so every slider visibly moves its own link.
    vis_chain = [frames[0][:3, 3]]
    for i in range(1, len(frames)):
        if not np.allclose(frames[i][:3, 3], frames[i - 1][:3, 3]):
            vis_chain.append(frames[i][:3, 3])
        for fi, length in VISUAL_OFFSETS:
            if fi == i:
                vis_chain.append(vis_chain[-1] + frames[i][:3, 2] * length)
    vis_chain = np.array(vis_chain)

    # --- Links: thick gray polyline through visual chain ---
    ax.plot(vis_chain[:, 0], vis_chain[:, 1], vis_chain[:, 2],
            color='#444444', linewidth=5, alpha=0.85, solid_capstyle='round',
            zorder=2, label='Links')

    # --- Frame triads (X=red, Y=green, Z=blue) ---
    axis_colors = ['#e63946', '#2a9d8f', '#1d4ed8']
    for i, T in enumerate(frames):
        o = T[:3, 3]
        for j in range(3):
            d = T[:3, j] * AXIS_LEN
            ax.quiver(o[0], o[1], o[2], d[0], d[1], d[2],
                      color=axis_colors[j], linewidth=1.8,
                      arrow_length_ratio=0.18, zorder=4)
        ax.text(o[0] + 8, o[1] + 8, o[2] + 8, f'{{{i}}}',
                fontsize=9, color='black', weight='bold', zorder=5)

    # --- Joint markers (frames 0..5) ---
    ax.scatter(origins[:-1, 0], origins[:-1, 1], origins[:-1, 2],
               color='black', s=45, zorder=6, label='Joints')

    # --- End effector ---
    ee = vis_chain[-1]
    ax.scatter(*ee, color='#d62fac', s=180, marker='*',
               zorder=7, edgecolors='black', linewidths=0.6, label='End Effector')

    # --- Ground plane hint ---
    gp = REACH * 0.95
    ax.plot([-gp, gp, gp, -gp, -gp], [-gp, -gp, gp, gp, -gp], [0]*5,
            color='lightgray', linewidth=0.6, alpha=0.6)

    # --- Styling ---
    ax.set_xlim(-REACH, REACH)
    ax.set_ylim(-REACH, REACH)
    ax.set_zlim(0, REACH * 1.2)
    ax.set_xlabel('X (mm)')
    ax.set_ylabel('Y (mm)')
    ax.set_zlabel('Z (mm)')
    ax.set_box_aspect([1, 1, 1.1])
    ax.set_title(f'{N_JOINTS}-DOF Robot Arm    EE = ({ee[0]:7.1f}, {ee[1]:7.1f}, {ee[2]:7.1f}) mm',
                 fontsize=11)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    ax.view_init(elev=view_state['elev'], azim=view_state['azim'])
    fig.canvas.draw_idle()


# -----------------------------------------------------------------------------
# Sliders + reset button
# -----------------------------------------------------------------------------
SLIDER_TOP = 0.88
SLIDER_DY  = min(0.095, (SLIDER_TOP - 0.12) / N_JOINTS)
SLIDER_H   = 0.025

sliders = []
for i, (name, (lo, hi)) in enumerate(zip(JOINT_NAMES, JOINT_LIMITS_DEG)):
    s_ax = fig.add_axes([0.72, SLIDER_TOP - i * SLIDER_DY, 0.25, SLIDER_H])
    s = Slider(s_ax, f'{name}  (°)', lo, hi, valinit=0.0,
               color='#4a90d9', track_color='#e6e6e6')
    s.label.set_fontsize(9)
    sliders.append(s)


def on_slider_change(_):
    angles_rad = np.array([np.radians(s.val) for s in sliders])
    draw(angles_rad)


for s in sliders:
    s.on_changed(on_slider_change)

reset_y = SLIDER_TOP - N_JOINTS * SLIDER_DY - 0.01
reset_ax = fig.add_axes([0.72, reset_y, 0.25, 0.045])
reset_btn = Button(reset_ax, 'Reset Joints', color='#f5f5f5', hovercolor='#dddddd')


def on_reset(_):
    for s in sliders:
        s.reset()


reset_btn.on_clicked(on_reset)

# Initial draw
draw(np.zeros(N_JOINTS))
plt.show()
