#!/usr/bin/env python3
"""
ViperX 300s — Joint Data Viewer
Interactive viewer for arm_data.csv — position, velocity & acceleration per joint.

Usage:
python3 visualize_arm_data.py [path/to/arm_data.csv]

Controls:
    Checkboxes on the right  — toggle individual joints on/off
    Reset button             — re-enable all joints
"""

import sys
import os
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.widgets import CheckButtons, Button

# ─── Visual style (matches Visualizer_org.py palette) ────────────────────────
BG_DARK  = "#0d0d1a"
BG_PANEL = "#13132b"
BG_MID   = "#1a1a3e"

JOINTS = [
    ("waist",        "#e74c3c"),
    ("shoulder",     "#e67e22"),
    ("elbow",        "#f1c40f"),
    ("forearm_roll", "#2ecc71"),
    ("wrist_angle",  "#3498db"),
    ("wrist_rotate", "#9b59b6"),
    ("gripper",      "#1abc9c"),
    ("left_finger",  "#f0a0ff"),
    ("right_finger", "#a0ffc8"),
]

SMOOTH_WIN = 9   # moving-average window for acceleration (samples)


# ─── Data helpers ─────────────────────────────────────────────────────────────

def _smooth(x, w):
    if w < 3 or len(x) < w:
        return x
    return np.convolve(x, np.ones(w) / w, mode="same")


def load_data(path):
    df = pd.read_csv(path)
    t = df["time"].values
    t = t - t[0]
    return t, df


def make_accel(t, vel_arr, smooth_w=SMOOTH_WIN):
    a = np.gradient(vel_arr, t)
    return _smooth(a, smooth_w)


# ─── Viewer ───────────────────────────────────────────────────────────────────

def run(path="data/arm_data.csv"):
    print(f"Loading {path} …")
    t, df = load_data(path)
    n_samples = len(t)
    duration  = t[-1]

    # Keep only joints that are actually present in the CSV
    available = [(nm, col) for nm, col in JOINTS if f"{nm}_pos" in df.columns]
    if not available:
        print("No recognisable joint columns found in CSV.")
        sys.exit(1)

    names  = [j[0] for j in available]
    colors = [j[1] for j in available]
    n      = len(names)

    pos    = [df[f"{nm}_pos"].values for nm in names]
    vel    = [df[f"{nm}_vel"].values for nm in names]
    accel  = [make_accel(t, df[f"{nm}_vel"].values) for nm in names]
    effort = [df[f"{nm}_effort"].values if f"{nm}_effort" in df.columns
              else np.zeros(len(t)) for nm in names]

    # ── Figure & axes ─────────────────────────────────────────────────────────
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(17, 12), facecolor=BG_DARK)
    fig.canvas.manager.set_window_title("ViperX 300s  ·  Joint Data Viewer")

    LX, LW = 0.06, 0.74          # left plots: x-start, width
    RX, RW = 0.82, 0.17          # right panel: x-start, width
    ROW_H  = 0.190
    ROWS_Y = [0.760, 0.545, 0.330, 0.115]  # y-start for pos / vel / accel / effort rows

    ax_pos  = fig.add_axes([LX, ROWS_Y[0], LW, ROW_H], facecolor=BG_MID)
    ax_vel  = fig.add_axes([LX, ROWS_Y[1], LW, ROW_H], facecolor=BG_MID)
    ax_acc  = fig.add_axes([LX, ROWS_Y[2], LW, ROW_H], facecolor=BG_MID)
    ax_eff  = fig.add_axes([LX, ROWS_Y[3], LW, ROW_H], facecolor=BG_MID)
    ax_chk  = fig.add_axes([RX, 0.13,      RW, 0.80],  facecolor=BG_PANEL)

    all_axes   = [ax_pos, ax_vel, ax_acc, ax_eff]
    all_data   = [pos,    vel,    accel,  effort]
    ylabels    = ["Position  (rad)", "Velocity  (rad/s)", "Acceleration  (rad/s²)", "Effort  (Nm)"]
    row_titles = ["Position", "Velocity", "Acceleration", "Effort"]

    # ── Plot lines ────────────────────────────────────────────────────────────
    lines_by_row = []
    for ax, d_list, title, ylabel in zip(all_axes, all_data, row_titles, ylabels):
        row = []
        for nm, col, d in zip(names, colors, d_list):
            ln, = ax.plot(t, d, color=col, linewidth=1.5, label=nm, alpha=0.92)
            row.append(ln)
        lines_by_row.append(row)

        ax.set_title(title, color="white", fontsize=10, pad=5, loc="left",
                     fontweight="bold")
        ax.set_ylabel(ylabel, color="#aaaacc", fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.28, color="#334466")
        ax.tick_params(colors="#7777aa", labelsize=8)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a55")

    # Share x-axis ticks; only bottom row gets x-label
    ax_pos.set_xticklabels([])
    ax_vel.set_xticklabels([])
    ax_acc.set_xticklabels([])
    ax_eff.set_xlabel("Time  (s)", color="#aaaacc", fontsize=9)

    # Legend inside position plot
    legend = ax_pos.legend(
        ncol=min(n, 5), loc="upper right",
        fontsize=8, framealpha=0.35,
        facecolor=BG_PANEL, edgecolor="#333366",
    )
    handles = getattr(legend, "legend_handles", None) or legend.legendHandles
    for lh, col in zip(handles, colors):
        lh.set_color(col)

    # ── Header text ───────────────────────────────────────────────────────────
    fig.text(0.5, 0.975,
             "ViperX 300s  ·  Joint Data Viewer",
             ha="center", va="top", color="white",
             fontsize=13, fontweight="bold")
    fig.text(0.5, 0.952,
             f"File: {path}   |   {n_samples} samples   |   "
             f"Duration: {duration:.2f} s",
             ha="center", va="top", color="#888899", fontsize=8.5)

    # ── CheckButtons (joint toggles) ──────────────────────────────────────────
    ax_chk.set_facecolor(BG_PANEL)
    for sp in ax_chk.spines.values():
        sp.set_edgecolor("#2a2a55")

    fig.text(RX + RW / 2, 0.945, "Joints",
             ha="center", va="top", color="#ccccee",
             fontsize=9.5, fontweight="bold")

    active = [True] * n
    chk = CheckButtons(ax_chk, names, active)

    # Style checkboxes
    for rect in getattr(chk, 'rectangles', getattr(chk, '_boxes', [])):
        rect.set_facecolor(BG_MID)
        rect.set_edgecolor("#555588")
        rect.set_linewidth(1.2)
    for lbl, col in zip(chk.labels, colors):
        lbl.set_color(col)
        lbl.set_fontsize(9.5)

    def _rebuild_legend():
        visible_lines  = [ln for ln in lines_by_row[0] if ln.get_visible()]
        visible_labels = [ln.get_label() for ln in visible_lines]
        legend = ax_pos.legend(
            visible_lines, visible_labels,
            ncol=min(len(visible_lines), 5), loc="upper right",
            fontsize=8, framealpha=0.35,
            facecolor=BG_PANEL, edgecolor="#333366",
        )
        handles = getattr(legend, "legend_handles", None) or legend.legendHandles
        for lh, ln in zip(handles, visible_lines):
            lh.set_color(ln.get_color())

    def on_toggle(label):
        idx = names.index(label)
        vis = not lines_by_row[0][idx].get_visible()
        for row in lines_by_row:
            row[idx].set_visible(vis)
        _rebuild_legend()
        fig.canvas.draw_idle()

    chk.on_clicked(on_toggle)

    # ── Reset button ──────────────────────────────────────────────────────────
    ax_btn = fig.add_axes([RX + 0.02, 0.055, RW - 0.04, 0.042])
    btn = Button(ax_btn, "Show All", color="#2c2c54", hovercolor="#3d3d7a")
    btn.label.set_color("white")
    btn.label.set_fontsize(9)

    def on_reset(_):
        for i in range(n):
            for row in lines_by_row:
                row[i].set_visible(True)
        # Sync checkbox state
        for i, status in enumerate(chk.get_status()):
            if not status:
                chk.set_active(i)
        _rebuild_legend()
        fig.canvas.draw_idle()

    btn.on_clicked(on_reset)

    # ── Save button ───────────────────────────────────────────────────────────
    ax_save_btn = fig.add_axes([RX + 0.02, 0.002, RW - 0.04, 0.042])
    save_btn = Button(ax_save_btn, "Save Figure", color="#2c5f2c", hovercolor="#3d8f3d")
    save_btn.label.set_color("white")
    save_btn.label.set_fontsize(9)

    def on_save(_):
        # Ensure figures folder exists
        os.makedirs("figures", exist_ok=True)
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"figures/arm_data_{timestamp}.png"
        fig.savefig(filename, facecolor=BG_DARK, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {filename}")

    save_btn.on_clicked(on_save)

    plt.show()


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/arm_data.csv"
    run(path)
