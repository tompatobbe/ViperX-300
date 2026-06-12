import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data/joint3_test.csv")

time_values = data.iloc[:, 0]
Joint3_pos = data.iloc[:, 7]

plt.figure(figsize=(10, 6))
plt.plot(data['time'], data['elbow_pos'], linewidth=2)
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Joint 3 Position (rad)', fontsize=12)
plt.title('Joint 3 Position vs Time', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()