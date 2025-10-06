"""
Test if there's a SCALE factor missing in our coordinate transform.
Maybe the coordinate system has different units or scaling.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Camera coords with baseline 2.115m
camera_coords = {
    90: np.array([7.5375, 2.1150, 18.0000]),
    93: np.array([7.0500, 1.9785, 18.1935]),
    96: np.array([6.4019, 1.8421, 18.1935]),
    99: np.array([5.7556, 1.7336, 18.4918])
}

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Test mapping [+Y, -X, -Z] with SCALE factors
def objective(params):
    """params = [scale_x, scale_y, scale_z, tx, ty, tz]"""
    scale_x, scale_y, scale_z, tx, ty, tz = params
    
    t = np.array([tx, ty, tz])
    scale = np.array([scale_x, scale_y, scale_z])
    
    total_error = 0
    for frame in [90, 93, 96, 99]:
        if frame in blender_df['frame'].values:
            row = blender_df[blender_df['frame'] == frame].iloc[0]
            target = np.array([row['x'], row['y'], row['z']])
            
            cam_pos = camera_coords[frame]
            
            # Apply mapping [+Y, -X, -Z]
            cam_blender = np.array([cam_pos[1], -cam_pos[0], -cam_pos[2]])
            
            # Apply scaling
            cam_blender_scaled = cam_blender * scale
            
            # Apply translation
            hawkeye_pos = cam_blender_scaled + t
            
            error = np.sum((hawkeye_pos - target)**2)
            total_error += error
    
    return total_error

print("="*70)
print("OPTIMIZING WITH SCALE FACTORS + TRANSLATION")
print("="*70)

# Initial guess: scale=1, t=0
x0 = [1.0, 1.0, 1.0, 0, 0, 0]

result = minimize(objective, x0, method='Powell', options={'maxiter': 10000})

if result.success:
    scale_x, scale_y, scale_z, tx, ty, tz = result.x
    
    print(f"\n✅ Optimization successful!")
    print(f"   Scale factors: [{scale_x:.4f}, {scale_y:.4f}, {scale_z:.4f}]")
    print(f"   Translation: [{tx:.3f}, {ty:.3f}, {tz:.3f}]")
    
    # Verify on each frame
    print("\n" + "="*70)
    print("FRAME-BY-FRAME VERIFICATION:")
    print("="*70)
    
    scale = np.array([scale_x, scale_y, scale_z])
    t = np.array([tx, ty, tz])
    
    errors = []
    for frame in [90, 93, 96, 99]:
        if frame in blender_df['frame'].values:
            row = blender_df[blender_df['frame'] == frame].iloc[0]
            target = np.array([row['x'], row['y'], row['z']])
            
            cam_pos = camera_coords[frame]
            
            # Apply mapping [+Y, -X, -Z]
            cam_blender = np.array([cam_pos[1], -cam_pos[0], -cam_pos[2]])
            
            # Apply scaling + translation
            hawkeye_pos = cam_blender * scale + t
            
            error_vec = hawkeye_pos - target
            error = np.linalg.norm(error_vec)
            errors.append(error)
            
            print(f"\nFrame {frame}:")
            print(f"  Camera coords: [{cam_pos[0]:7.3f}, {cam_pos[1]:7.3f}, {cam_pos[2]:7.3f}]")
            print(f"  After mapping: [{cam_blender[0]:7.3f}, {cam_blender[1]:7.3f}, {cam_blender[2]:7.3f}]")
            print(f"  After scaling: [{(cam_blender*scale)[0]:7.3f}, {(cam_blender*scale)[1]:7.3f}, {(cam_blender*scale)[2]:7.3f}]")
            print(f"  Hawkeye:       [{hawkeye_pos[0]:7.3f}, {hawkeye_pos[1]:7.3f}, {hawkeye_pos[2]:7.3f}]")
            print(f"  Ground Truth:  [{target[0]:7.3f}, {target[1]:7.3f}, {target[2]:7.3f}]")
            print(f"  Error:         [{error_vec[0]:7.3f}, {error_vec[1]:7.3f}, {error_vec[2]:7.3f}] = {error:.3f}m")
    
    print("\n" + "="*70)
    print(f"AVERAGE ERROR: {np.mean(errors):.3f}m")
    print(f"MAX ERROR: {np.max(errors):.3f}m")
    print(f"MIN ERROR: {np.min(errors):.3f}m")
    print("="*70)
    
    print("\nINTERPRETATION:")
    if abs(scale_x - 1.0) > 0.01:
        print(f"  X-axis needs {scale_x:.2f}x scaling")
    if abs(scale_y - 1.0) > 0.01:
        print(f"  Y-axis needs {scale_y:.2f}x scaling")
    if abs(scale_z - 1.0) > 0.01:
        print(f"  Z-axis needs {scale_z:.2f}x scaling")
else:
    print(f"\n❌ Optimization failed: {result.message}")
