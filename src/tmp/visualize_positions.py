"""
Visualize where the ball positions are relative to the court.
Show both ground truth and Hawkeye output on a court diagram.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Court dimensions (from Blender)
court_length = 18.0  # Y-direction
court_width = 9.0    # X-direction
court_center = np.array([-0.031, -4.500, 0.015])

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Hawkeye outputs (from our test)
hawkeye_outputs = {
    90: np.array([0.171, -6.715, 2.477]),
    93: np.array([0.034, -6.228, 2.284]),
    96: np.array([-0.102, -5.579, 2.284]),
    99: np.array([-0.211, -4.933, 1.986])
}

# Create top-down view (X-Y plane)
fig, ax = plt.subplots(figsize=(10, 12))

# Draw court
court_x = [-court_width/2, court_width/2, court_width/2, -court_width/2, -court_width/2]
court_y = [-court_length/2, -court_length/2, court_length/2, court_length/2, -court_length/2]

# Adjust for court center offset
court_x = [x + court_center[0] for x in court_x]
court_y = [y + court_center[1] for y in court_y]

ax.plot(court_x, court_y, 'k-', linewidth=2, label='Court')

# Draw net (at Y=0 relative to court center)
net_y = court_center[1]
ax.plot([court_x[0], court_x[1]], [net_y, net_y], 'k--', linewidth=1, label='Net')

# Plot ground truth positions
for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        row = blender_df[blender_df['frame'] == frame].iloc[0]
        ax.plot(row['x'], row['y'], 'go', markersize=10, label='Ground Truth' if frame == 90 else '')

# Plot Hawkeye outputs
for frame in [90, 93, 96, 99]:
    if frame in hawkeye_outputs:
        pos = hawkeye_outputs[frame]
        ax.plot(pos[0], pos[1], 'rx', markersize=10, label='Hawkeye' if frame == 90 else '')

# Draw lines connecting GT to Hawkeye for each frame
for frame in [90, 93, 96, 99]:
    if frame in hawkeye_outputs and frame in blender_df['frame'].values:
        row = blender_df[blender_df['frame'] == frame].iloc[0]
        gt = np.array([row['x'], row['y']])
        hw = hawkeye_outputs[frame][:2]
        ax.plot([gt[0], hw[0]], [gt[1], hw[1]], 'b--', alpha=0.3, linewidth=1)
        
        # Add frame label
        mid = (gt + hw) / 2
        ax.text(mid[0], mid[1], f'{frame}', fontsize=8)

ax.set_xlabel('X (m)', fontsize=12)
ax.set_ylabel('Y (m)', fontsize=12)
ax.set_title('Court Top View: Ground Truth vs Hawkeye Output', fontsize=14)
ax.axis('equal')
ax.grid(True, alpha=0.3)
ax.legend()

# Add text showing errors
error_text = "Errors (3D):\n"
for frame in [90, 93, 96, 99]:
    if frame in hawkeye_outputs and frame in blender_df['frame'].values:
        row = blender_df[blender_df['frame'] == frame].iloc[0]
        gt = np.array([row['x'], row['y'], row['z']])
        hw = hawkeye_outputs[frame]
        error = np.linalg.norm(hw - gt)
        error_text += f"Frame {frame}: {error:.2f}m\n"

ax.text(0.02, 0.98, error_text, transform=ax.transAxes, 
        verticalalignment='top', fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('court_position_comparison.png', dpi=150)
print("Saved court_position_comparison.png")

# Also create a 3D view
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Draw court in 3D (just the floor)
court_x_3d = [-court_width/2, court_width/2, court_width/2, -court_width/2, -court_width/2]
court_y_3d = [-court_length/2, -court_length/2, court_length/2, court_length/2, -court_length/2]
court_z_3d = [0, 0, 0, 0, 0]

court_x_3d = [x + court_center[0] for x in court_x_3d]
court_y_3d = [y + court_center[1] for y in court_y_3d]

ax.plot(court_x_3d, court_y_3d, court_z_3d, 'k-', linewidth=2, label='Court')

# Plot positions in 3D
for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        row = blender_df[blender_df['frame'] == frame].iloc[0]
        ax.scatter(row['x'], row['y'], row['z'], c='g', marker='o', s=100, label='GT' if frame == 90 else '')
        
    if frame in hawkeye_outputs:
        pos = hawkeye_outputs[frame]
        ax.scatter(pos[0], pos[1], pos[2], c='r', marker='x', s=100, label='HE' if frame == 90 else '')
        
        # Connect with line
        if frame in blender_df['frame'].values:
            row = blender_df[blender_df['frame'] == frame].iloc[0]
            ax.plot([row['x'], pos[0]], [row['y'], pos[1]], [row['z'], pos[2]], 'b--', alpha=0.3)

ax.set_xlabel('X (m)')
ax.set_ylabel('Y (m)')
ax.set_zlabel('Z (m)')
ax.set_title('3D View: Ground Truth vs Hawkeye Output')
ax.legend()

plt.tight_layout()
plt.savefig('position_comparison_3d.png', dpi=150)
print("Saved position_comparison_3d.png")

print("\n" + "="*70)
print("POSITION ANALYSIS")
print("="*70)
print("\nGround Truth pattern (Y values): Moving from -9.3m → -2.7m (approaching net)")
print("Hawkeye pattern (Y values): -6.7m → -4.9m (also approaching, but offset by ~2.6m)")
print("\nThe issue: Hawkeye positions are systematically offset in Y-direction by ~2.6m")
print("This creates a consistent bias - all balls appear closer to net than they actually are.")
print("="*70)
