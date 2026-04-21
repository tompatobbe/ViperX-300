#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
  ViperX 300s — Denavit-Hartenberg 3D Visualizer
  An interactive educational tool for understanding DH convention
═══════════════════════════════════════════════════════════════════════════════

  STANDARD (CLASSIC) DH CONVENTION
  ─────────────────────────────────
  Each joint i has 4 parameters that describe the rigid-body transform
  from frame {i-1} to frame {i}:

      T_i = Rot_z(θ_i) · Trans_z(d_i) · Trans_x(a_i) · Rot_x(α_i)

  ┌─────────┬───────────────────────────────────────────────────────────────┐
  │ θ (theta)│ Rotation around z_{i-1} — the JOINT VARIABLE for revolute   │
  │ d        │ Translation along z_{i-1} — link offset                      │
  │ a        │ Translation along x_i    — link length                       │
  │ α (alpha)│ Rotation around x_i      — link twist                        │
  └──────────┴──────────────────────────────────────────────────────────────┘

  FRAME AXES:  X = Red ←  Y = Green ←  Z = Blue

  USAGE
  ─────
  • Drag joint sliders to animate the arm
  • Watch each DH frame move with its joint
  • "Reset" returns all joints to 0°
  • Modify DH_TABLE below to model a different robot

  REQUIREMENTS
  ────────────
  pip install numpy matplotlib
"""

import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

# ─────────────────────────────────────────────────────────────────────────────
#  ViperX 300s  ·  Standard DH Parameters
#  Source: Trossen Robotics ViperX 300 6DOF specifications
#  Units: millimetres and degrees
# ─────────────────────────────────────────────────────────────────────────────
#
#  DH_TABLE  columns:  [ θ_offset(°),  d(mm),  a(mm),  α(°) ]
#
#  θ_offset is a constant added to the joint variable θ to correct
#  the zero-pose alignment (common in real robots).
#
DH_TABLE = np.array([
    #   θ_off     d       a       α        Joint name
    [    0.0,  126.75,    0.0,  -90.0  ],  # 1  Waist
    [    0.0,    0.0,    0.0,    0.0  ],  # 2  Shoulder
    [    0.0,    0.0,  305.94,    0.0  ],  # 3  Elbow
    [    0.0,  300.0,    0.0,  -90.0  ],  # 4  Wrist Rotate
    [    0.0,    0.0,    0.0,   90.0  ],  # 5  Wrist Pitch
    [    0.0,   0.0,    0.0,  -90.0  ],  # 6  Wrist Roll (to EE)
])

JOINT_NAMES  = ["Waist", "Shoulder", "Elbow", "Wrist Rotate", "Wrist Pitch", "Wrist Roll"]
JOINT_LIMITS = [(-180, 180), (-106, 101), (-101, 92), (-180, 180), (-101, 101), (-180, 180)]
N_JOINTS     = 6

# Visual constants
FRAME_SCALE  = 45    # length of each frame axis arrow (mm)
JOINT_COLORS = ["#e74c3c", "#e67e22", "#f1c40f", "#2ecc71", "#3498db", "#9b59b6"]
BG_DARK      = "#0d0d1a"
BG_PANEL     = "#13132b"
BG_MID       = "#1a1a3e"


# ─────────────────────────────────────────────────────────────────────────────
#  Mathematics
# ─────────────────────────────────────────────────────────────────────────────

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


def forward_kinematics(q_deg: np.ndarray, dh: np.ndarray):
    """
    Return a list of 4×4 transforms [T_base, T_0→1, T_0→2, ..., T_0→6].
    The first entry is the world/base frame (identity).
    """
    frames = [np.eye(4)]
    T = np.eye(4)
    for i in range(N_JOINTS):
        θ  = q_deg[i] + dh[i, 0]    # joint angle + offset
        Ti = dh_matrix(θ, dh[i, 1], dh[i, 2], dh[i, 3])
        T  = T @ Ti
        frames.append(T.copy())
    return frames


# ─────────────────────────────────────────────────────────────────────────────
#  Drawing helpers
# ─────────────────────────────────────────────────────────────────────────────

def draw_frame(ax, T: np.ndarray, scale: float = 40,
               alpha: float = 1.0, label: str = None, linewidth: float = 2.0):
    """Draw a right-handed coordinate frame.  X=red  Y=green  Z=blue."""
    o  = T[:3, 3]
    axes_rgb = [("X", T[:3, 0], "#ff4444"),
                ("Y", T[:3, 1], "#44ff88"),
                ("Z", T[:3, 2], "#4488ff")]

    for name, axis, color in axes_rgb:
        ax.quiver(o[0], o[1], o[2],
                  axis[0]*scale, axis[1]*scale, axis[2]*scale,
                  color=color, alpha=alpha, linewidth=linewidth,
                  arrow_length_ratio=0.25)

    if label is not None:
        ax.text(o[0], o[1], o[2] + scale * 0.45,
                label, fontsize=8, ha='center', va='bottom',
                color='white',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='#222244',
                          edgecolor='#555588', alpha=0.85))


def draw_cylinder(ax, p1, p2, radius=8, color="#cccccc", alpha=0.55, n=12):
    """Draw a solid cylindrical link between two 3-D points."""
    d   = p2 - p1
    L   = np.linalg.norm(d)
    if L < 1e-6:
        return
    d   = d / L

    # Build a perpendicular basis
    perp = np.array([0, 0, 1]) if abs(d[2]) < 0.9 else np.array([1, 0, 0])
    u    = np.cross(d, perp);  u /= np.linalg.norm(u)
    v    = np.cross(d, u)

    phi = np.linspace(0, 2*np.pi, n)
    ring = radius * (np.outer(np.cos(phi), u) + np.outer(np.sin(phi), v))

    for i in range(n - 1):
        xs = [p1[0]+ring[i,0], p1[0]+ring[i+1,0],
              p2[0]+ring[i+1,0], p2[0]+ring[i,0]]
        ys = [p1[1]+ring[i,1], p1[1]+ring[i+1,1],
              p2[1]+ring[i+1,1], p2[1]+ring[i,1]]
        zs = [p1[2]+ring[i,2], p1[2]+ring[i+1,2],
              p2[2]+ring[i+1,2], p2[2]+ring[i,2]]
        ax.plot_surface(np.array([xs[:2], xs[2:]]),
                        np.array([ys[:2], ys[2:]]),
                        np.array([zs[:2], zs[2:]]),
                        color=color, alpha=alpha, shade=True,
                        linewidth=0, antialiased=True)


# ─────────────────────────────────────────────────────────────────────────────
#  Main application class
# ─────────────────────────────────────────────────────────────────────────────

class DHVisualizer:

    def __init__(self):
        self.q   = np.zeros(N_JOINTS)          # current joint angles
        self.dh  = DH_TABLE.copy()

        # ── Figure ────────────────────────────────────────────────────────
        self.fig = plt.figure(figsize=(18, 10), facecolor=BG_DARK)
        self.fig.canvas.manager.set_window_title(
            "ViperX 300s  ·  Denavit-Hartenberg 3D Visualizer")

        # ── 3D viewport  (left 60 %) ──────────────────────────────────────
        self.ax3d = self.fig.add_axes([0.01, 0.02, 0.55, 0.95],
                                      projection='3d')
        self.ax3d.set_facecolor(BG_DARK)

        # ── DH table (top-right) ──────────────────────────────────────────
        self.ax_tbl = self.fig.add_axes([0.58, 0.55, 0.40, 0.40])
        self.ax_tbl.set_facecolor(BG_PANEL)
        self.ax_tbl.axis('off')

        # ── Legend (mid-right) ────────────────────────────────────────────
        self.ax_leg = self.fig.add_axes([0.58, 0.40, 0.40, 0.14])
        self.ax_leg.set_facecolor(BG_PANEL)
        self.ax_leg.axis('off')

        # ── EE readout (bottom-right, above sliders) ──────────────────────
        self.ax_ee = self.fig.add_axes([0.58, 0.32, 0.40, 0.075])
        self.ax_ee.set_facecolor(BG_PANEL)
        self.ax_ee.axis('off')

        # ── Sliders ───────────────────────────────────────────────────────
        self._build_sliders()

        # ── Reset button ──────────────────────────────────────────────────
        ax_btn = self.fig.add_axes([0.765, 0.02, 0.09, 0.038])
        self.btn_reset = Button(ax_btn, "Reset Joints",
                                color="#2c2c54", hovercolor="#3d3d7a")
        self.btn_reset.label.set_color("white")
        self.btn_reset.label.set_fontsize(8)
        self.btn_reset.on_clicked(self._reset)

        # ── First draw ────────────────────────────────────────────────────
        self._draw_legend()
        self._redraw()

    # ── Slider construction ───────────────────────────────────────────────

    def _build_sliders(self):
        self.sliders = []
        y0, dy = 0.265, 0.048
        for i in range(N_JOINTS):
            lo, hi = JOINT_LIMITS[i]
            ax_s = self.fig.add_axes(
                [0.60, y0 - i * dy, 0.355, 0.025],
                facecolor="#1e1e40")
            s = Slider(ax_s,
                       f"J{i+1}  {JOINT_NAMES[i]:<13}",
                       lo, hi, valinit=0.0,
                       color=JOINT_COLORS[i], track_color="#2a2a50")
            s.label.set_color("white")
            s.label.set_fontsize(8.5)
            s.valtext.set_color(JOINT_COLORS[i])
            s.valtext.set_fontsize(8)
            s.on_changed(self._on_change)
            self.sliders.append(s)

    def _on_change(self, _):
        self.q = np.array([s.val for s in self.sliders])
        self._redraw()

    def _reset(self, _):
        for s in self.sliders:
            s.set_val(0.0)

    # ── Static legend ─────────────────────────────────────────────────────

    def _draw_legend(self):
        ax = self.ax_leg
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax.text(0.5, 0.95, "AXIS CONVENTION",
                ha='center', va='top', color='#aaaacc',
                fontsize=8, fontweight='bold', transform=ax.transAxes)

        entries = [("X axis", "#ff4444"), ("Y axis", "#44ff88"),
                   ("Z axis", "#4488ff")]
        for k, (lbl, col) in enumerate(entries):
            x = 0.08 + k * 0.32
            ax.annotate("", xy=(x+0.12, 0.5), xytext=(x, 0.5),
                        xycoords='axes fraction', textcoords='axes fraction',
                        arrowprops=dict(arrowstyle='->', color=col, lw=2))
            ax.text(x + 0.14, 0.50, lbl, ha='left', va='center',
                    color=col, fontsize=8.5, transform=ax.transAxes)

        ax.text(0.5, 0.08,
                "T_i = Rot_z(θ) · Trans_z(d) · Trans_x(a) · Rot_x(α)",
                ha='center', va='bottom', color='#888899',
                fontsize=7.5, style='italic', transform=ax.transAxes,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d0d1a',
                          edgecolor='#333366', alpha=0.8))

    # ── DH table ──────────────────────────────────────────────────────────

    def _draw_table(self):
        ax = self.ax_tbl
        ax.cla(); ax.set_facecolor(BG_PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        ax.text(0.5, 0.97, "DH Parameter Table  ·  Standard Convention",
                ha='center', va='top', color='white',
                fontsize=9.5, fontweight='bold', transform=ax.transAxes)

        # ── Column headers ────────────────────────────────────────────────
        headers = ["Joint", "θ_offset (°)", "d (mm)", "a (mm)", "α (°)",
                   "θ_joint (°)"]
        col_x   = [0.00, 0.18, 0.34, 0.48, 0.62, 0.78]
        col_w   = [0.18, 0.16, 0.14, 0.14, 0.16, 0.22]
        n_rows  = N_JOINTS
        row_h   = 0.10
        hdr_y   = 0.88

        for cx, cw, hdr in zip(col_x, col_w, headers):
            rect = plt.Rectangle((cx, hdr_y - row_h*0.8), cw - 0.01, row_h*0.8,
                                  facecolor='#2c2c5a', edgecolor='#44449a',
                                  transform=ax.transAxes, clip_on=True)
            ax.add_patch(rect)
            ax.text(cx + cw*0.47, hdr_y - row_h*0.35,
                    hdr, ha='center', va='center',
                    color='#ccccee', fontsize=7.5, fontweight='bold',
                    transform=ax.transAxes)

        # ── Data rows ─────────────────────────────────────────────────────
        for i in range(n_rows):
            y = hdr_y - (i + 1.2) * row_h
            jcol = JOINT_COLORS[i]
            bg   = "#16162e" if i % 2 == 0 else "#1c1c38"

            rect = plt.Rectangle((0, y), 1.0 - 0.01, row_h * 0.88,
                                  facecolor=bg, edgecolor='#2a2a55',
                                  transform=ax.transAxes, clip_on=True)
            ax.add_patch(rect)

            theta_total = self.q[i] + self.dh[i, 0]
            values = [f"{i+1}: {JOINT_NAMES[i]}",
                      f"{self.dh[i, 0]:+.1f}",
                      f"{self.dh[i, 1]:.1f}",
                      f"{self.dh[i, 2]:.1f}",
                      f"{self.dh[i, 3]:+.1f}",
                      f"{theta_total:+.1f}"]
            colors = [jcol, '#dddddd', '#dddddd', '#dddddd', '#dddddd', '#f0d060']

            for cx, cw, val, col in zip(col_x, col_w, values, colors):
                ax.text(cx + cw*0.47, y + row_h*0.38,
                        val, ha='center', va='center',
                        color=col, fontsize=7.5, transform=ax.transAxes)

    # ── EE readout ────────────────────────────────────────────────────────

    def _draw_ee(self, frames):
        ax = self.ax_ee
        ax.cla(); ax.set_facecolor(BG_PANEL); ax.axis('off')
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        T_ee = frames[-1]
        pos  = T_ee[:3, 3]
        rx   = np.degrees(np.arctan2( T_ee[2,1], T_ee[2,2]))
        ry   = np.degrees(np.arctan2(-T_ee[2,0], np.sqrt(T_ee[2,1]**2 + T_ee[2,2]**2)))
        rz   = np.degrees(np.arctan2( T_ee[1,0], T_ee[0,0]))

        ax.text(0.5, 0.92, "End-Effector Pose",
                ha='center', va='top', color='#aaaacc',
                fontsize=8, fontweight='bold', transform=ax.transAxes)

        pstr = (f"X={pos[0]:+7.1f}mm   Y={pos[1]:+7.1f}mm   Z={pos[2]:+7.1f}mm  "
                f"   Rx={rx:+6.1f}°  Ry={ry:+6.1f}°  Rz={rz:+6.1f}°")
        ax.text(0.5, 0.28, pstr,
                ha='center', va='center', color='#f0d060',
                fontsize=8, fontfamily='monospace', transform=ax.transAxes)

    # ── 3D scene ──────────────────────────────────────────────────────────

    def _draw_3d(self, frames):
        ax = self.ax3d
        ax.cla()

        # Pane / grid style
        for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
            pane.fill = False
            pane.set_edgecolor("#222244")
        ax.grid(True, linestyle=':', alpha=0.25, color='#334466')

        ax.set_facecolor(BG_DARK)
        ax.set_xlabel("X  (mm)", color='#6666aa', fontsize=8, labelpad=4)
        ax.set_ylabel("Y  (mm)", color='#6666aa', fontsize=8, labelpad=4)
        ax.set_zlabel("Z  (mm)", color='#6666aa', fontsize=8, labelpad=4)
        ax.tick_params(colors='#555577', labelsize=7)
        ax.set_title("ViperX 300s  ·  DH Frames in 3D Space",
                     color='white', fontsize=11, pad=8)

        R = 450
        ax.set_xlim(-R, R); ax.set_ylim(-R, R); ax.set_zlim(0, 750)

        # ── Ground disk ────────────────────────────────────────────────────
        phi  = np.linspace(0, 2*np.pi, 64)
        r_gd = 90
        ax.plot(r_gd*np.cos(phi), r_gd*np.sin(phi), np.zeros(64),
                color='#555588', linewidth=1.5, alpha=0.6)
        ax.plot(0, 0, 0, 'o', color='white', markersize=7, zorder=10)

        # ── Links & joints ─────────────────────────────────────────────────
        positions = [f[:3, 3] for f in frames]

        for i in range(len(positions) - 1):
            p1, p2 = positions[i], positions[i + 1]
            jcol   = JOINT_COLORS[min(i, N_JOINTS - 1)]

            # Cylinder link
            draw_cylinder(ax, p1, p2, radius=9, color=jcol, alpha=0.50)

            # Core line (always visible)
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                    '-', color=jcol, linewidth=3, alpha=0.9, zorder=4)

            # Joint sphere
            ax.scatter(*p2, color=jcol, s=80, zorder=6, edgecolors='white',
                       linewidths=0.6)

        # ── DH frames at each joint ────────────────────────────────────────
        for i, frame in enumerate(frames):
            label     = "{0}" if i == 0 else f"{{{i}}}"
            is_ee     = (i == N_JOINTS)
            fscale    = FRAME_SCALE * 1.5 if is_ee else FRAME_SCALE
            lw        = 2.8 if is_ee else 2.0
            draw_frame(ax, frame, scale=fscale, label=label, linewidth=lw)

        # ── Annotation: z-axis of joint 1 for context ─────────────────────
        ax.quiver(0, 0, 0, 0, 0, 70, color='#4488ff',
                  alpha=0.4, linewidth=1.5, arrow_length_ratio=0.15,
                  linestyle='dashed')

    # ── Master redraw ─────────────────────────────────────────────────────

    def _redraw(self):
        frames = forward_kinematics(self.q, self.dh)
        self._draw_3d(frames)
        self._draw_table()
        self._draw_ee(frames)
        self.fig.canvas.draw_idle()

    def run(self):
        plt.show()


# ─────────────────────────────────────────────────────────────────────────────
#  Entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(__doc__)
    print("Launching visualizer …  (close the window to exit)\n")
    try:
        app = DHVisualizer()
        app.run()
    except KeyboardInterrupt:
        print("\nExited.")
        sys.exit(0)