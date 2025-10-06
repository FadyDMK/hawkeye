"""
Camera at (-18, -6, 3) gives correct depth, but wrong projection.
Let's try various rotations to find the right one.
"""

import numpy as np

def blender_rotation_matrix(rx, ry, rz):
    """Blender rotation matrix with XYZ Euler order."""
    rx, ry, rz = np.radians(rx), np.radians(ry), np.radians(rz)
    
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(rx), -np.sin(rx)],
                   [0, np.sin(rx), np.cos(rx)]])
    
    Ry = np.array([[np.cos(ry), 0, np.sin(ry)],
                   [0, 1, 0],
                   [-np.sin(ry), 0, np.cos(ry)]])
    
    Rz = np.array([[np.cos(rz), -np.sin(rz), 0],
                   [np.sin(rz), np.cos(rz), 0],
                   [0, 0, 1]])
    
    return Rz @ Ry @ Rx

ball_blender = np.array([-0.004, -9.347, 2.278])
cam_pos = np.array([-18, -6, 3])  # Flipped X position

focal = 1600.0
cx, cy = 960.0, 540.0
target_x, target_y = 1630, 728

print("="*70)
print("TESTING ROTATIONS WITH CAMERA AT (-18, -6, 3)")
print("="*70)

# Try various rotations
best_error = float('inf')
best_rot = None

for rx in [0, 90, -90, 180]:
    for ry in [0, 90, -90, 180]:
        for rz in [0, 90, -90, 180]:
            R = blender_rotation_matrix(rx, ry, rz)
            R_inv = R.T
            
            ball_from_cam_blender = ball_blender - cam_pos
            ball_from_cam_camera = R_inv @ ball_from_cam_blender
            
            if ball_from_cam_camera[2] > 0:  # Ball in front
                img_x = focal * ball_from_cam_camera[0] / ball_from_cam_camera[2] + cx
                img_y = focal * ball_from_cam_camera[1] / ball_from_cam_camera[2] + cy
                
                error = np.sqrt((img_x - target_x)**2 + (img_y - target_y)**2)
                
                if error < best_error:
                    best_error = error
                    best_rot = (rx, ry, rz)
                    best_proj = (img_x, img_y)
                    best_cam_coords = ball_from_cam_camera

print(f"\n🎯 BEST ROTATION FOUND:")
print(f"   Rotation: {best_rot}")
print(f"   Projected: ({best_proj[0]:.1f}, {best_proj[1]:.1f})")
print(f"   Target:    ({target_x}, {target_y})")
print(f"   Error: {best_error:.1f}px")
print(f"   Camera coords: X={best_cam_coords[0]:.3f}, Y={best_cam_coords[1]:.3f}, Z={best_cam_coords[2]:.3f}")

# Test this rotation on all frames
import pandas as pd

blender_df = pd.read_csv('ball_positions_blender.csv')

measurements = {
    90: {'left': (1630, 728)},
    93: {'left': (1580, 714)},
    96: {'left': (1523, 702)},
    99: {'left': (1458, 690)}
}

print("\n" + "="*70)
print(f"TESTING ROTATION {best_rot} ON ALL FRAMES")
print("="*70)

R = blender_rotation_matrix(*best_rot)
R_inv = R.T

for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
        ball_blender = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
        
        ball_from_cam_blender = ball_blender - cam_pos
        ball_from_cam_camera = R_inv @ ball_from_cam_blender
        
        if ball_from_cam_camera[2] > 0:
            img_x = focal * ball_from_cam_camera[0] / ball_from_cam_camera[2] + cx
            img_y = focal * ball_from_cam_camera[1] / ball_from_cam_camera[2] + cy
            
            target = measurements[frame]['left']
            error = np.sqrt((img_x - target[0])**2 + (img_y - target[1])**2)
            
            print(f"\nFrame {frame}:")
            print(f"  Projected: ({img_x:.1f}, {img_y:.1f})")
            print(f"  Actual:    {target}")
            print(f"  Error: {error:.1f}px")
            print(f"  Camera Z: {ball_from_cam_camera[2]:.3f}m")
