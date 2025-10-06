"""
Test the new configuration ([+Y, -X, -Z] mapping with baseline 2.115m) on our test frames.
"""

import numpy as np
import pandas as pd
import sys
import os

# Add src to path
sys.path.append('src')

from camera_config import load_camera_config

# Load config
config = load_camera_config()

# Camera coords with baseline 2.115m
camera_coords = {
    90: np.array([7.5375, 2.1150, 18.0000]),
    93: np.array([7.0500, 1.9785, 18.1935]),
    96: np.array([6.4019, 1.8421, 18.1935]),
    99: np.array([5.7556, 1.7336, 18.4918])
}

# Transform parameters
R = np.eye(3)
t = np.array([-1.9443, 0.8225, 20.4775])

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

print("="*70)
print("TESTING NEW CONFIGURATION")
print("="*70)
print(f"Baseline: {config['baseline_m']}m")
print(f"Mapping: [+Y, -X, -Z]")
print(f"Transform t: {t}")
print("="*70)

errors = []

for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
        target = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
        
        cam_pos = camera_coords[frame]
        
        # Apply mapping: [+Y, -X, -Z]
        cam_blender = np.array([cam_pos[1], -cam_pos[0], -cam_pos[2]])
        
        # Transform to world
        hawkeye_pos = R @ cam_blender + t
        
        error_vec = hawkeye_pos - target
        error_3d = np.linalg.norm(error_vec)
        errors.append(error_3d)
        
        print(f"\nFrame {frame}:")
        print(f"  Camera coords:   [{cam_pos[0]:7.3f}, {cam_pos[1]:7.3f}, {cam_pos[2]:7.3f}]")
        print(f"  After mapping:   [{cam_blender[0]:7.3f}, {cam_blender[1]:7.3f}, {cam_blender[2]:7.3f}]")
        print(f"  World (Hawkeye): [{hawkeye_pos[0]:7.3f}, {hawkeye_pos[1]:7.3f}, {hawkeye_pos[2]:7.3f}]")
        print(f"  Ground Truth:    [{target[0]:7.3f}, {target[1]:7.3f}, {target[2]:7.3f}]")
        print(f"  Error:           [{error_vec[0]:7.3f}, {error_vec[1]:7.3f}, {error_vec[2]:7.3f}]")
        print(f"  3D Error: {error_3d:.3f}m")

print("\n" + "="*70)
print(f"AVERAGE ERROR: {np.mean(errors):.3f}m")
print(f"MAX ERROR: {np.max(errors):.3f}m")
print(f"MIN ERROR: {np.min(errors):.3f}m")
print("="*70)
