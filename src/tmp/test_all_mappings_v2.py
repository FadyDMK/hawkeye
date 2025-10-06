"""Test all possible coordinate mappings with the new camera coords"""
import numpy as np
import pandas as pd
from itertools import permutations, product

# New camera coords with baseline 1.442m
camera_coords = {
    90: np.array([5.1390, 1.4420, 12.2723]),
    93: np.array([4.8067, 1.3490, 12.4043]),
    96: np.array([4.3648, 1.2559, 12.4043]),
    99: np.array([3.9241, 1.1820, 12.6077])
}

# Load ground truth
df = pd.read_csv('ball_positions_blender.csv')

R = np.eye(3)

print("Testing all coordinate mappings...")
print("="*70)

best_error = float('inf')
best_mapping = None
best_t = None

# Try all axis permutations and sign combinations
axes_perms = list(permutations([0, 1, 2]))  # Which axis maps to which
sign_combos = list(product([-1, 1], repeat=3))  # Sign for each axis

for axes in axes_perms:
    for signs in sign_combos:
        # Calculate optimal t for this mapping
        optimal_ts = []
        
        for frame in [90, 93, 96, 99]:
            gt_row = df[df['frame'] == frame].iloc[0]
            target = np.array([gt_row['x'], gt_row['y'], gt_row['z']])
            
            cam_pos = camera_coords[frame]
            
            # Apply the mapping
            mapped = np.array([
                signs[0] * cam_pos[axes[0]],
                signs[1] * cam_pos[axes[1]],
                signs[2] * cam_pos[axes[2]]
            ])
            
            # Calculate t
            optimal_t = target - mapped
            optimal_ts.append(optimal_t)
        
        # Average t
        avg_t = np.mean(optimal_ts, axis=0)
        
        # Calculate total error
        total_error = 0
        for frame in [90, 93, 96, 99]:
            gt_row = df[df['frame'] == frame].iloc[0]
            target = np.array([gt_row['x'], gt_row['y'], gt_row['z']])
            
            cam_pos = camera_coords[frame]
            mapped = np.array([
                signs[0] * cam_pos[axes[0]],
                signs[1] * cam_pos[axes[1]],
                signs[2] * cam_pos[axes[2]]
            ])
            
            world_pos = mapped + avg_t
            error = np.linalg.norm(world_pos - target)
            total_error += error
        
        avg_error = total_error / 4
        
        if avg_error < best_error:
            best_error = avg_error
            best_mapping = (axes, signs)
            best_t = avg_t

# Print best result
axes, signs = best_mapping
axis_names = ['X', 'Y', 'Z']
mapping_str = [f"{'+' if signs[i] > 0 else '-'}{axis_names[axes[i]]}" for i in range(3)]

print(f"\n🎯 BEST MAPPING FOUND:")
print(f"   Mapping: [{', '.join(mapping_str)}]")
print(f"   Average Error: {best_error:.3f}m")
print(f"   Optimal t: [{best_t[0]:.4f}, {best_t[1]:.4f}, {best_t[2]:.4f}]")
print(f"\n   Translation matrix:")
print(f"   [[{signs[0] if axes[0]==0 else 0}, {signs[0] if axes[0]==1 else 0}, {signs[0] if axes[0]==2 else 0}],")
print(f"    [{signs[1] if axes[1]==0 else 0}, {signs[1] if axes[1]==1 else 0}, {signs[1] if axes[1]==2 else 0}],")
print(f"    [{signs[2] if axes[2]==0 else 0}, {signs[2] if axes[2]==1 else 0}, {signs[2] if axes[2]==2 else 0}]]")

print(f"\n   Python code:")
print(f"   ball_pos_blender = np.array([{mapping_str[0].replace('+', '').replace('X', 'ball_pos[0,0]').replace('Y', 'ball_pos[1,0]').replace('Z', 'ball_pos[2,0]')}, {mapping_str[1].replace('+', '').replace('X', 'ball_pos[0,0]').replace('Y', 'ball_pos[1,0]').replace('Z', 'ball_pos[2,0]')}, {mapping_str[2].replace('+', '').replace('X', 'ball_pos[0,0]').replace('Y', 'ball_pos[1,0]').replace('Z', 'ball_pos[2,0]')}]).reshape(3,1)")

# Test the best mapping
print(f"\n{'='*70}")
print("TESTING BEST MAPPING:")
print(f"{'='*70}")

for frame in [90, 93, 96, 99]:
    gt_row = df[df['frame'] == frame].iloc[0]
    target = np.array([gt_row['x'], gt_row['y'], gt_row['z']])
    
    cam_pos = camera_coords[frame]
    mapped = np.array([
        signs[0] * cam_pos[axes[0]],
        signs[1] * cam_pos[axes[1]],
        signs[2] * cam_pos[axes[2]]
    ])
    
    world_pos = mapped + best_t
    error_vec = world_pos - target
    error = np.linalg.norm(error_vec)
    
    print(f"\nFrame {frame}:")
    print(f"  Ground truth: X={target[0]:7.3f}, Y={target[1]:7.3f}, Z={target[2]:7.3f}")
    print(f"  Hawkeye:      X={world_pos[0]:7.3f}, Y={world_pos[1]:7.3f}, Z={world_pos[2]:7.3f}")
    print(f"  Error:        X={error_vec[0]:7.3f}, Y={error_vec[1]:7.3f}, Z={error_vec[2]:7.3f}  (3D: {error:.3f}m)")
