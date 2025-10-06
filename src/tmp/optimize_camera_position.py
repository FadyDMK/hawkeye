"""
Calculate optimal camera position accounting for court not being at origin.

The issue: We assumed court center is at world origin (0,0,0), but it's not!
This causes a systematic offset in all calculated positions.
"""

import numpy as np
import pandas as pd

# Hawkeye calculated positions (camera space, with baseline 2.115m)
camera_coords = {
    90: np.array([7.5375, 2.1150, 18.0000]),
    93: np.array([7.0500, 1.9785, 18.1935]),
    96: np.array([6.4019, 1.8421, 18.1935]),
    99: np.array([5.7556, 1.7336, 18.4918])
}

# Load Blender ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Current transform
R_current = np.array([[1.0, 0.0, 0.0],
                      [0.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0]])
t_current = np.array([18.0, -6.0, 3.0])

print("="*70)
print("OPTIMAL CAMERA POSITION CALCULATION")
print("(Accounting for court not at origin)")
print("="*70)

# For each test frame, calculate what t would give us the correct position
optimal_ts = []

for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        # Get ground truth
        blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
        target = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
        
        # Get camera coords
        cam_pos = camera_coords[frame]
        
        # Apply coordinate conversion: [-Z, X, -Y] (corrected for baseline 2.115m)
        cam_blender = np.array([-cam_pos[2], cam_pos[0], -cam_pos[1]])
        
        # Calculate what t should be: t = target - R @ cam_blender
        optimal_t = target - (R_current @ cam_blender)
        optimal_ts.append(optimal_t)
        
        print(f"\nFrame {frame}:")
        print(f"  Camera coords:     [{cam_pos[0]:7.3f}, {cam_pos[1]:7.3f}, {cam_pos[2]:7.3f}]")
        print(f"  → Blender offset:  [{cam_blender[0]:7.3f}, {cam_blender[1]:7.3f}, {cam_blender[2]:7.3f}]")
        print(f"  Ground truth:      [{target[0]:7.3f}, {target[1]:7.3f}, {target[2]:7.3f}]")
        print(f"  Calculated t:      [{optimal_t[0]:7.3f}, {optimal_t[1]:7.3f}, {optimal_t[2]:7.3f}]")

# Calculate average optimal t
optimal_t_avg = np.mean(optimal_ts, axis=0)
std_t = np.std(optimal_ts, axis=0)

print("\n" + "="*70)
print("RESULTS:")
print("="*70)
print(f"\nCurrent t:   [{t_current[0]:7.3f}, {t_current[1]:7.3f}, {t_current[2]:7.3f}]")
print(f"Optimal t:   [{optimal_t_avg[0]:7.3f}, {optimal_t_avg[1]:7.3f}, {optimal_t_avg[2]:7.3f}]")
print(f"Std dev:     [{std_t[0]:7.3f}, {std_t[1]:7.3f}, {std_t[2]:7.3f}]")
print(f"\nOffset needed: [{optimal_t_avg[0]-t_current[0]:7.3f}, {optimal_t_avg[1]-t_current[1]:7.3f}, {optimal_t_avg[2]-t_current[2]:7.3f}]")

# Test the optimal t
print("\n" + "="*70)
print("TESTING OPTIMAL T:")
print("="*70)

total_error = 0
errors = []
for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
        target = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
        
        cam_pos = camera_coords[frame]
        cam_blender = np.array([-cam_pos[2], cam_pos[0], -cam_pos[1]])
        
        # Calculate world position with optimal t
        world_pos = (R_current @ cam_blender) + optimal_t_avg
        
        error_vec = world_pos - target
        error = np.linalg.norm(error_vec)
        total_error += error
        errors.append(error)
        
        print(f"\nFrame {frame}:")
        print(f"  Blender:  X={target[0]:7.3f}, Y={target[1]:7.3f}, Z={target[2]:7.3f}")
        print(f"  Hawkeye:  X={world_pos[0]:7.3f}, Y={world_pos[1]:7.3f}, Z={world_pos[2]:7.3f}")
        print(f"  Error:    X={error_vec[0]:7.3f}, Y={error_vec[1]:7.3f}, Z={error_vec[2]:7.3f}  (3D: {error:.3f}m)")

print(f"\n{'='*70}")
print(f"Average error with optimal t: {total_error/len(errors):.3f}m")
print(f"Max error: {max(errors):.3f}m")
print(f"Min error: {min(errors):.3f}m")
print(f"{'='*70}")

print("\n📝 RECOMMENDATION:")
print("Update hawkeye_pipeline.py with:")
print(f"  self.t = [{optimal_t_avg[0]:.4f}, {optimal_t_avg[1]:.4f}, {optimal_t_avg[2]:.4f}]")
