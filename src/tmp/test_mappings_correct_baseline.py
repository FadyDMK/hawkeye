"""
Test all coordinate mappings with CORRECT baseline (2.115m).
"""

import numpy as np
import pandas as pd

# Camera coords with baseline 2.115m
camera_coords = {
    90: np.array([7.5375, 2.1150, 18.0000]),
    93: np.array([7.0500, 1.9785, 18.1935]),
    96: np.array([6.4019, 1.8421, 18.1935]),
    99: np.array([5.7556, 1.7336, 18.4918])
}

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Identity rotation
R = np.eye(3)

# Test all 48 possible mappings
axes = [0, 1, 2]  # X, Y, Z
signs = [-1, 1]

import itertools

results = []

for perm in itertools.permutations(axes):
    for sign_combo in itertools.product(signs, repeat=3):
        # Build mapping
        mapping = [perm[i] * sign_combo[i] for i in range(3)]
        
        # Calculate optimal t and error for this mapping
        optimal_ts = []
        errors = []
        
        for frame in [90, 93, 96, 99]:
            if frame in blender_df['frame'].values:
                blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
                target = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
                
                cam_pos = camera_coords[frame]
                
                # Apply mapping
                cam_blender = np.array([
                    cam_pos[abs(mapping[0])] * np.sign(mapping[0]) if mapping[0] != 0 else -cam_pos[abs(mapping[0])],
                    cam_pos[abs(mapping[1])] * np.sign(mapping[1]) if mapping[1] != 0 else -cam_pos[abs(mapping[1])],
                    cam_pos[abs(mapping[2])] * np.sign(mapping[2]) if mapping[2] != 0 else -cam_pos[abs(mapping[2])]
                ])
                
                optimal_t = target - (R @ cam_blender)
                optimal_ts.append(optimal_t)
        
        # Calculate average t
        avg_t = np.mean(optimal_ts, axis=0)
        
        # Calculate errors with this avg t
        for frame in [90, 93, 96, 99]:
            if frame in blender_df['frame'].values:
                blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
                target = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
                
                cam_pos = camera_coords[frame]
                
                # Apply mapping
                cam_blender = np.array([
                    cam_pos[abs(mapping[0])] * np.sign(mapping[0]) if mapping[0] != 0 else -cam_pos[abs(mapping[0])],
                    cam_pos[abs(mapping[1])] * np.sign(mapping[1]) if mapping[1] != 0 else -cam_pos[abs(mapping[1])],
                    cam_pos[abs(mapping[2])] * np.sign(mapping[2]) if mapping[2] != 0 else -cam_pos[abs(mapping[2])]
                ])
                
                hawkeye_pos = R @ cam_blender + avg_t
                error = np.linalg.norm(hawkeye_pos - target)
                errors.append(error)
        
        avg_error = np.mean(errors)
        
        # Convert mapping to readable form
        axis_names = ['X', 'Y', 'Z']
        mapping_str = f"[{'+' if mapping[0] > 0 else '-'}{axis_names[abs(mapping[0])]}, " \
                      f"{'+' if mapping[1] > 0 else '-'}{axis_names[abs(mapping[1])]}, " \
                      f"{'+' if mapping[2] > 0 else '-'}{axis_names[abs(mapping[2])]}]"
        
        results.append({
            'mapping': mapping_str,
            'mapping_code': mapping,
            'avg_error': avg_error,
            'optimal_t': avg_t
        })

# Sort by error
results.sort(key=lambda x: x['avg_error'])

print("="*70)
print("TOP 10 COORDINATE MAPPINGS (with baseline 2.115m)")
print("="*70)

for i, result in enumerate(results[:10]):
    print(f"\n#{i+1}: {result['mapping']}")
    print(f"   Avg Error: {result['avg_error']:.3f}m")
    print(f"   Optimal t: [{result['optimal_t'][0]:.4f}, {result['optimal_t'][1]:.4f}, {result['optimal_t'][2]:.4f}]")

print("\n" + "="*70)
print("🎯 BEST MAPPING:")
print("="*70)
best = results[0]
print(f"Mapping: {best['mapping']}")
print(f"Average Error: {best['avg_error']:.3f}m")
print(f"Optimal t: [{best['optimal_t'][0]:.4f}, {best['optimal_t'][1]:.4f}, {best['optimal_t'][2]:.4f}]")

# Show individual frame errors for best mapping
print("\n" + "="*70)
print("INDIVIDUAL FRAME ERRORS (Best Mapping):")
print("="*70)

mapping = best['mapping_code']
avg_t = best['optimal_t']

for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
        target = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
        
        cam_pos = camera_coords[frame]
        
        # Apply mapping
        cam_blender = np.array([
            cam_pos[abs(mapping[0])] * np.sign(mapping[0]) if mapping[0] != 0 else -cam_pos[abs(mapping[0])],
            cam_pos[abs(mapping[1])] * np.sign(mapping[1]) if mapping[1] != 0 else -cam_pos[abs(mapping[1])],
            cam_pos[abs(mapping[2])] * np.sign(mapping[2]) if mapping[2] != 0 else -cam_pos[abs(mapping[2])]
        ])
        
        hawkeye_pos = R @ cam_blender + avg_t
        error_vec = hawkeye_pos - target
        error_3d = np.linalg.norm(error_vec)
        
        print(f"\nFrame {frame}:")
        print(f"  Ground Truth: X={target[0]:7.3f}, Y={target[1]:7.3f}, Z={target[2]:7.3f}")
        print(f"  Hawkeye:      X={hawkeye_pos[0]:7.3f}, Y={hawkeye_pos[1]:7.3f}, Z={hawkeye_pos[2]:7.3f}")
        print(f"  Error:        X={error_vec[0]:7.3f}, Y={error_vec[1]:7.3f}, Z={error_vec[2]:7.3f}")
        print(f"  3D Error: {error_3d:.3f}m")
