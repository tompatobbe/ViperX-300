#!/usr/bin/env python3
"""
Visualization script for arm_data.csv

Plots position, velocity, and acceleration for all joints over time.
"""
import csv
import sys
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


def load_csv(filepath):
    """Load CSV and return time and joint data"""
    times = []
    joint_data = defaultdict(lambda: {'pos': [], 'vel': [], 'eff': []})
    
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        # Parse header to identify columns
        time_idx = header.index('time')
        joint_names = set()
        
        for i, col in enumerate(header):
            if col.endswith('_pos'):
                joint_name = col[:-4]
                joint_names.add(joint_name)
        
        joint_names = sorted(list(joint_names))
        
        # Read data
        for row in reader:
            times.append(float(row[time_idx]))
            for joint in joint_names:
                try:
                    pos_idx = header.index(f"{joint}_pos")
                    joint_data[joint]['pos'].append(float(row[pos_idx]))
                except (ValueError, IndexError):
                    pass
                
                try:
                    vel_idx = header.index(f"{joint}_vel")
                    joint_data[joint]['vel'].append(float(row[vel_idx]))
                except (ValueError, IndexError):
                    pass
                
                try:
                    eff_idx = header.index(f"{joint}_eff")
                    joint_data[joint]['eff'].append(float(row[eff_idx]))
                except (ValueError, IndexError):
                    pass
    
    # Convert times to relative (seconds from start)
    if times:
        times = np.array(times)
        times = times - times[0]
    
    return times, joint_data


def calculate_acceleration(times, velocities):
    """Calculate acceleration from velocity using numerical differentiation"""
    accel = np.zeros_like(velocities)
    dt = np.diff(times)
    
    for i in range(1, len(velocities)):
        if dt[i-1] > 0:
            accel[i] = (velocities[i] - velocities[i-1]) / dt[i-1]
    
    return accel


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize arm joint data from CSV')
    parser.add_argument('filepath', nargs='?', default='arm_data.csv', help='Path to CSV file')
    parser.add_argument('--save', help='Save plot to file instead of displaying')
    parser.add_argument('--skip-accel', action='store_true', help='Skip acceleration plots')
    parser.add_argument('--joints', help='Comma-separated list of joints to plot (default: all)')
    args = parser.parse_args()
    
    print(f"Loading data from {args.filepath}...")
    times, joint_data = load_csv(args.filepath)
    
    if not times.size:
        print("No data found in CSV file")
        sys.exit(1)
    
    joint_names = sorted(list(joint_data.keys()))
    
    # Filter joints if specified
    if args.joints:
        selected = args.joints.split(',')
        joint_names = [j for j in joint_names if j in selected]
    
    num_joints = len(joint_names)
    print(f"Found {num_joints} joints: {', '.join(joint_names)}")
    
    # Create subplots: 2-3 rows (pos, vel, accel) x num_joints columns
    num_rows = 2 if args.skip_accel else 3
    fig, axes = plt.subplots(num_rows, num_joints, figsize=(3.5*num_joints, 2.5*num_rows))
    fig.suptitle('Arm Joint Data: Position, Velocity, and Acceleration', fontsize=12)
    
    # Handle single joint case (axes is 1D)
    if num_joints == 1:
        axes = axes.reshape(num_rows, 1)
    
    row_idx = 0
    for col, joint in enumerate(joint_names):
        pos = np.array(joint_data[joint]['pos'])
        vel = np.array(joint_data[joint]['vel'])
        
        # Position plot
        axes[0, col].plot(times, pos, 'b-', linewidth=1.5)
        axes[0, col].set_title(f'{joint} - Position', fontsize=10)
        axes[0, col].set_ylabel('Position (rad)', fontsize=9)
        axes[0, col].grid(True, alpha=0.3)
        
        # Velocity plot
        axes[1, col].plot(times, vel, 'g-', linewidth=1.5)
        axes[1, col].set_title(f'{joint} - Velocity', fontsize=10)
        axes[1, col].set_ylabel('Velocity (rad/s)', fontsize=9)
        axes[1, col].grid(True, alpha=0.3)
        
        # Acceleration plot
        if not args.skip_accel:
            accel = calculate_acceleration(times, vel)
            axes[2, col].plot(times, accel, 'r-', linewidth=1.5)
            axes[2, col].set_title(f'{joint} - Acceleration', fontsize=10)
            axes[2, col].set_ylabel('Acceleration (rad/s²)', fontsize=9)
            axes[2, col].set_xlabel('Time (s)', fontsize=9)
            axes[2, col].grid(True, alpha=0.3)
        else:
            axes[1, col].set_xlabel('Time (s)', fontsize=9)
    
    plt.tight_layout()
    
    if args.save:
        print(f"Saving plot to {args.save}...")
        plt.savefig(args.save, dpi=100, bbox_inches='tight')
        print(f"Saved to {args.save}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
