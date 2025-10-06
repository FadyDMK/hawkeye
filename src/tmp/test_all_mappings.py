"""
Systematically try all coordinate mapping combinations to find the correct one
"""

import numpy as np
import pandas as pd

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Test frames camera coords
camera_coords = {
    90: np.array([-2.5376, 0.4624, 18.0451]),
    93: np.array([-0.0899, 0.2809, 17.9775]),
    96: np.array([2.1250, 0.4659, 18.1818]),
    99: np.array([3.8864, 1.3409, 18.1818])
}

# Try all 8 possible sign combinations for [±opencv_X, ±opencv_Y, ±opencv_Z]
# mapping to [Blender_X, Blender_Y, Blender_Z]
mappings = [
    ([1, 0, 0], [0, 1, 0], [0, 0, 1], "X, Y, Z"),
    ([1, 0, 0], [0, -1, 0], [0, 0, 1], "X, -Y, Z"),
    ([1, 0, 0], [0, 0, 1], [0, 1, 0], "X, Z, Y"),
    ([1, 0, 0], [0, 0, -1], [0, 1, 0], "X, -Z, Y"),
    ([1, 0, 0], [0, 1, 0], [0, 0, -1], "X, Y, -Z"),
    ([1, 0, 0], [0, -1, 0], [0, 0, -1], "X, -Y, -Z"),
    ([1, 0, 0], [0, 0, 1], [0, 1, 0], "X, Z, Y"),
    ([1, 0, 0], [0, 0, -1], [0, 1, 0], "X, -Z, Y"),
    # Negated X versions
    ([-1, 0, 0], [0, 1, 0], [0, 0, 1], "-X, Y, Z"),
    ([-1, 0, 0], [0, -1, 0], [0, 0, 1], "-X, -Y, Z"),
    ([-1, 0, 0], [0, 0, 1], [0, 1, 0], "-X, Z, Y"),
    ([-1, 0, 0], [0, 0, -1], [0, 1, 0], "-X, -Z, Y"),
    ([-1, 0, 0], [0, 1, 0], [0, 0, -1], "-X, Y, -Z"),
    ([-1, 0, 0], [0, -1, 0], [0, 0, -1], "-X, -Y, -Z"),
    # Swapped versions (Y to X, X to Y)
    ([0, 1, 0], [1, 0, 0], [0, 0, 1], "Y, X, Z"),
    ([0, 1, 0], [-1, 0, 0], [0, 0, 1], "Y, -X, Z"),
    ([0, -1, 0], [1, 0, 0], [0, 0, 1], "-Y, X, Z"),
    ([0, -1, 0], [-1, 0, 0], [0, 0, 1], "-Y, -X, Z"),
    # Z to X variations
    ([0, 0, 1], [1, 0, 0], [0, 1, 0], "Z, X, Y"),
    ([0, 0, -1], [1, 0, 0], [0, 1, 0], "-Z, X, Y"),
    ([0, 0, 1], [-1, 0, 0], [0, 1, 0], "Z, -X, Y"),
    ([0, 0, -1], [-1, 0, 0], [0, 1, 0], "-Z, -X, Y"),
    ([0, 0, 1], [0, 1, 0], [1, 0, 0], "Z, Y, X"),
    ([0, 0, -1], [0, 1, 0], [1, 0, 0], "-Z, Y, X"),
]

print("Testing all coordinate mappings...")
print("="*80)

best_error = float('inf')
best_mapping = None
best_t = None

for mx, my, mz, desc in mappings:
    M = np.array([mx, my, mz]).T  # Transformation matrix
    R = np.eye(3)
    
    # Calculate optimal t for this mapping
    optimal_ts = []
    for frame in [90, 93, 96, 99]:
        if frame in blender_df['frame'].values:
            gt_row = blender_df[blender_df['frame'] == frame].iloc[0]
            target = np.array([gt_row['x'], gt_row['y'], gt_row['z']])
            
            cam_pos = camera_coords[frame]
            cam_blender = M @ cam_pos
            
            optimal_t = target - (R @ cam_blender)
            optimal_ts.append(optimal_t)
    
    optimal_t_avg = np.mean(optimal_ts, axis=0)
    
    # Test error with this mapping
    total_error = 0
    for frame in [90, 93, 96, 99]:
        if frame in blender_df['frame'].values:
            gt_row = blender_df[blender_df['frame'] == frame].iloc[0]
            target = np.array([gt_row['x'], gt_row['y'], gt_row['z']])
            
            cam_pos = camera_coords[frame]
            cam_blender = M @ cam_pos
            world_pos = (R @ cam_blender) + optimal_t_avg
            
            error = np.linalg.norm(world_pos - target)
            total_error += error
    
    avg_error = total_error / 4
    
    if avg_error < best_error:
        best_error = avg_error
        best_mapping = (mx, my, mz, desc)
        best_t = optimal_t_avg

print(f"\n🎯 BEST MAPPING FOUND:")
print(f"   Mapping: [{best_mapping[3]}]")
print(f"   Average Error: {best_error:.3f}m")
print(f"   Optimal t: [{best_t[0]:.4f}, {best_t[1]:.4f}, {best_t[2]:.4f}]")
print(f"\n   Translation matrix:")
mx, my, mz, _ = best_mapping
print(f"   [[{mx[0]}, {mx[1]}, {mx[2]}],")
print(f"    [{my[0]}, {my[1]}, {my[2]}],")
print(f"    [{mz[0]}, {mz[1]}, {mz[2]}]]")
