"""
Solve for the actual camera position that makes the projection work.

Given:
- Ball world position (known from Blender)
- Ball image position (detected)
- Focal length: 1600px
- Camera rotation: Some unknown rotation

We can solve for the camera position that makes img_x = focal * cam_X / cam_Z + cx work.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Known values
focal = 1600.0
cx, cy = 960.0, 540.0

# Measurements
measurements = {
    90: {'left': (1630, 728), 'disparity': 188},
    93: {'left': (1580, 714), 'disparity': 186},
    96: {'left': (1523, 702), 'disparity': 186},
    99: {'left': (1458, 690), 'disparity': 183}
}

def rotation_matrix(rx, ry, rz):
    """Create rotation matrix from Euler angles (degrees)."""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    
    # Rotation around X
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    # Rotation around Y
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    # Rotation around Z
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

def project_error(params):
    """Calculate projection error for given camera pose."""
    cam_x, cam_y, cam_z, rot_x, rot_y, rot_z = params
    
    cam_pos = np.array([cam_x, cam_y, cam_z])
    R = rotation_matrix(rot_x, rot_y, rot_z)
    
    total_error = 0
    
    for frame in [90, 93, 96, 99]:
        if frame in blender_df['frame'].values:
            blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
            world_pos = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
            
            meas = measurements[frame]
            left_x, left_y = meas['left']
            
            # Vector from camera to ball in world coords
            ball_from_cam = world_pos - cam_pos
            
            # Transform to camera coords
            cam_coords = R @ ball_from_cam
            
            # Project
            if cam_coords[2] > 0:  # Ball in front of camera
                img_x = focal * cam_coords[0] / cam_coords[2] + cx
                img_y = focal * cam_coords[1] / cam_coords[2] + cy
                
                # Error
                error = np.sqrt((img_x - left_x)**2 + (img_y - left_y)**2)
                total_error += error
            else:
                total_error += 10000  # Penalty for ball behind camera
    
    return total_error

print("="*70)
print("SOLVING FOR CAMERA POSE")
print("="*70)

# Initial guess: camera at (18, -6, 3) with rotation (90, 0, 90)
x0 = [18, -6, 3, 90, 0, 90]

print("\nOptimizing camera pose...")
result = minimize(project_error, x0, method='Powell', options={'maxiter': 10000})

if result.success:
    cam_x, cam_y, cam_z, rot_x, rot_y, rot_z = result.x
    print(f"\n✅ Found optimal camera pose!")
    print(f"   Position: ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})")
    print(f"   Rotation: ({rot_x:.1f}°, {rot_y:.1f}°, {rot_z:.1f}°)")
    print(f"   Total projection error: {result.fun:.1f} pixels")
    
    # Verify projections
    cam_pos = np.array([cam_x, cam_y, cam_z])
    R = rotation_matrix(rot_x, rot_y, rot_z)
    
    print("\n" + "="*70)
    print("VERIFICATION:")
    print("="*70)
    
    for frame in [90, 93, 96, 99]:
        if frame in blender_df['frame'].values:
            blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
            world_pos = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
            
            meas = measurements[frame]
            left_x, left_y = meas['left']
            
            ball_from_cam = world_pos - cam_pos
            cam_coords = R @ ball_from_cam
            
            img_x = focal * cam_coords[0] / cam_coords[2] + cx
            img_y = focal * cam_coords[1] / cam_coords[2] + cy
            
            print(f"\nFrame {frame}:")
            print(f"  Projected: ({img_x:.1f}, {img_y:.1f})")
            print(f"  Actual:    ({left_x}, {left_y})")
            print(f"  Error:     ({img_x - left_x:.1f}, {img_y - left_y:.1f}) = {np.sqrt((img_x - left_x)**2 + (img_y - left_y)**2):.1f}px")
else:
    print("\n❌ Optimization failed")
    print(result.message)

print("\n" + "="*70)
print("COMPARISON TO NOMINAL:")
print("="*70)
print(f"Nominal position: (18.000, -6.000, 3.000)")
print(f"Actual position:  ({cam_x:.3f}, {cam_y:.3f}, {cam_z:.3f})")
print(f"Difference:       ({cam_x-18:.3f}, {cam_y+6:.3f}, {cam_z-3:.3f})")
print(f"\nNominal rotation: (90.0°, 0.0°, 90.0°)")
print(f"Actual rotation:  ({rot_x:.1f}°, {rot_y:.1f}°, {rot_z:.1f}°)")
print(f"Difference:       ({rot_x-90:.1f}°, {rot_y:.1f}°, {rot_z-90:.1f}°)")
