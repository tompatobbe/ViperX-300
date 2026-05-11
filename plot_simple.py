#!/usr/bin/env python3
"""
Simple 4-panel plot: position, velocity, acceleration, effort for all joints.

Usage:
    python3 plot_simple.py [path/to/arm_data.csv]
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

def main(path="data/joint3_test.csv"):
    df = pd.read_csv(path)
    t  = df["time"].values
    t  = t - t[0]

    available = [(name, color) for name, color in JOINTS if f"{name}_pos" in df.columns]
    if not available:
        print("No recognisable joint columns found.")
        sys.exit(1)

    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"Joint Data  —  {path}", fontsize=12)

    labels_done = set()
    for name, color in available:
        pos = df[f"{name}_pos"].values
        vel = df[f"{name}_vel"].values
        acc = np.gradient(vel, t)
        eff = df[f"{name}_effort"].values

        label = name if name not in labels_done else "_nolegend_"
        labels_done.add(name)

        axes[0].plot(t, pos, color=color, linewidth=1.4, label=name)
        axes[1].plot(t, vel, color=color, linewidth=1.4)
        axes[2].plot(t, acc, color=color, linewidth=1.4)
        axes[3].plot(t, eff, color=color, linewidth=1.4)

    titles  = ["Position (rad)", "Velocity (rad/s)", "Acceleration (rad/s²)", "Effort (Nm)"]
    for ax, title in zip(axes, titles):
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[3].set_xlabel("Time (s)", fontsize=9)
    axes[0].legend(ncol=3, fontsize=8, loc="upper right")

    plt.tight_layout()
    plt.show()

def main_elbow(path="data/joint3_test.csv"):
    df = pd.read_csv(path); t = df["time"].values - df["time"].values[0]
    fig, axes = plt.subplots(4, 1, sharex=True, figsize=(10, 7), tight_layout=True)
    for ax, ylabel, y in zip(axes, ["pos (rad)", "vel (rad/s)", "accel (rad/s²)", "effort (Nm)"],
                                   [df.elbow_pos, df.elbow_vel, np.gradient(df.elbow_vel, t), df.elbow_effort]):
        ax.plot(t, y); ax.set_ylabel(ylabel); ax.grid(True, linestyle=":", alpha=0.4)
    plt.suptitle("Elbow — joint 3"); plt.show()


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/joint3_test.csv"
    main_elbow(path)
