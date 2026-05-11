import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/arm_data.csv")

t = df["time"] - df["time"].iloc[0]  # relative time in seconds

joints = [
    "waist", "shoulder", "elbow", "forearm_roll",
    "wrist_angle", "wrist_rotate", "gripper", "left_finger", "right_finger"
]

fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
fig.suptitle("Arm Joint Data", fontsize=14)

titles = ["Position (rad)", "Velocity (rad/s)", "Acceleration (rad/s²)", "Effort"]

for joint in joints:
    pos = df[f"{joint}_pos"].values
    vel = df[f"{joint}_vel"].values
    effort = df[f"{joint}_effort"].values
    accel = np.gradient(vel, t.values)

    axes[0].plot(t, pos, label=joint)
    axes[1].plot(t, vel, label=joint)
    axes[2].plot(t, accel, label=joint)
    axes[3].plot(t, effort, label=joint)

for ax, title in zip(axes, titles):
    ax.set_ylabel(title)
    ax.grid(True, alpha=0.3)

axes[3].set_xlabel("Time (s)")
axes[0].legend(loc="upper right", fontsize=7, ncol=3)

plt.tight_layout()
plt.savefig("plots/arm_data_plot.png", dpi=150)
plt.show()
print("Saved plots/arm_data_plot.png")
