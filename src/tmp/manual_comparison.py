"""
Manual comparison of test frames 90, 93, 96, 99
Using data from our diagnostic runs
"""

import pandas as pd
import numpy as np

# Hawkeye calculated positions (UPDATED with optimized t values)
hawkeye_data = {
    90: {'x': 0.024, 'y': -9.247, 'z': 2.433},
    93: {'x': 0.092, 'y': -6.800, 'z': 2.614},
    96: {'x': -0.112, 'y': -4.585, 'z': 2.429},
    99: {'x': -0.112, 'y': -2.823, 'z': 1.554}
}

# Load Blender ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

print("="*70)
print("MANUAL COMPARISON: Hawkeye vs Blender Ground Truth")
print("="*70)
print("\nComparing test frames: 90, 93, 96, 99\n")

total_error_3d = 0
count = 0

for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
        hawkeye = hawkeye_data[frame]
        
        # Calculate errors
        error_x = hawkeye['x'] - blender_row['x']
        error_y = hawkeye['y'] - blender_row['y']
        error_z = hawkeye['z'] - blender_row['z']
        error_3d = np.sqrt(error_x**2 + error_y**2 + error_z**2)
        
        total_error_3d += error_3d
        count += 1
        
        print(f"Frame {frame}:")
        print(f"  Blender:  X={blender_row['x']:7.3f}, Y={blender_row['y']:7.3f}, Z={blender_row['z']:7.3f}")
        print(f"  Hawkeye:  X={hawkeye['x']:7.3f}, Y={hawkeye['y']:7.3f}, Z={hawkeye['z']:7.3f}")
        print(f"  Error:    X={error_x:7.3f}, Y={error_y:7.3f}, Z={error_z:7.3f}  (3D: {error_3d:.3f}m)")
        print()

print("="*70)
print(f"Average 3D error: {total_error_3d/count:.3f}m")
print("="*70)
