import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ── Joint angles (degrees) — change these to move the arm ────────────────────
q = [0, 0, 0, 0, 0, 0]   # [waist, shoulder, elbow, wrist_rotate, wrist_pitch, wrist_roll]

# ── ViperX-300 DH parameters ──────────────────────────────────────────────────
# Each row: [alpha(°), a(mm), d(mm), theta_offset(°)]
# Transform: T = Rot_z(q+theta_offset) @ Trans_z(d) @ Trans_x(a) @ Rot_x(alpha)
DH = [
    #  alpha    a       d      theta_off
    [  90.0,   0.0,  126.75,   0.0  ],  # joint 1 — waist
    [   0.0, 305.94,   0.0,   78.66 ],  # joint 2 — shoulder
    [  90.0, 300.0,    0.0,  -90.0  ],  # joint 3 — elbow
    [   0.0,   0.0,   50.0,  -90.0  ],  # joint 4 — wrist rotate
    [  90.0,   0.0,    0.0,   90.0  ],  # joint 5 — wrist pitch
    [ -90.0,   0.0,   50.0,  -90.0  ],  # joint 6 — wrist roll
]


def rot_z(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([
        [ c, -s, 0, 0],
        [ s,  c, 0, 0],
        [ 0,  0, 1, 0],
        [ 0,  0, 0, 1],
    ])

def rot_x(alpha):
    c, s = np.cos(alpha), np.sin(alpha)
    return np.array([
        [1,  0,  0, 0],
        [0,  c, -s, 0],
        [0,  s,  c, 0],
        [0,  0,  0, 1],
    ])

def trans_z(d):
    return np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, d],
        [0, 0, 0, 1],
    ])

def trans_x(a):
    return np.array([
        [1, 0, 0, a],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])


def dh_matrix(theta_deg, d, a, alpha_deg):
    t = np.radians(theta_deg)
    al = np.radians(alpha_deg)
    return rot_z(t) @ trans_z(d) @ trans_x(a) @ rot_x(al)




# ── Build frames joint by joint ───────────────────────────────────────────────
T = np.eye(4)
frames = [T.copy()]   # frame 0 = base

for i, (params, angle) in enumerate(zip(DH, q)):
    alpha, a, d, theta_off = params
    Ti = dh_matrix(angle + theta_off, d, a, alpha)
    T = T @ Ti
    frames.append(T.copy())
    print(f"Joint {i+1}: origin = {T[:3,3].round(1)}")

# ── Plot ──────────────────────────────────────────────────────────────────────
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Draw links
positions = [f[:3, 3] for f in frames]
for i in range(len(positions) - 1):
    p1, p2 = positions[i], positions[i+1]
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], 'o-', linewidth=2)

# Draw a coordinate frame at a given transform T
def draw_frame(ax, T, scale=30, label=None):
    o = T[:3, 3]
    ax.quiver(*o, *T[:3,0]*scale, color='red')    # x
    ax.quiver(*o, *T[:3,1]*scale, color='green')  # y
    ax.quiver(*o, *T[:3,2]*scale, color='blue')   # z
    if label:
        ax.text(*o, label, fontsize=8)

for i, frame in enumerate(frames):
    # scale arrows to 20% of the distance to the next frame, capped at 40 mm
    if i < len(frames) - 1:
        dist = np.linalg.norm(frames[i+1][:3, 3] - frame[:3, 3])
        arrow_scale = min(max(dist * 0.2, 8), 40)
    else:
        arrow_scale = 20
    draw_frame(ax, frame, scale=arrow_scale, label=str(i))

ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
ax.set_title('ViperX-300 — sandbox')

# equal aspect ratio so arrow lengths are consistent across all axes
all_pts = np.array(positions)
center = all_pts.mean(axis=0)
half = np.ptp(all_pts, axis=0).max() / 2 * 1.1
ax.set_xlim(center[0]-half, center[0]+half)
ax.set_ylim(center[1]-half, center[1]+half)
ax.set_zlim(center[2]-half, center[2]+half)
ax.set_box_aspect([1, 1, 1])
plt.tight_layout()
plt.show()
