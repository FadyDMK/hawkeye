"""
Let's think about Blender camera coordinates more carefully.

In Blender:
- Right camera at (18, -3, 3)
- Left camera at (18, -6, 3)
- Both have rotation (90°, 0°, 90°)

Rotation (90, 0, 90) in Blender means:
- First rotate 90° around X axis
- Then 0° around Y axis  
- Then 90° around Z axis

Let's figure out what direction the camera is actually pointing.
"""

import numpy as np

def blender_rotation_matrix(rx, ry, rz):
    """
    Blender rotation matrix.
    Blender uses XYZ Euler angles applied in order: X -> Y -> Z
    """
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
    
    # Blender applies: Z * Y * X
    return Rz @ Ry @ Rx

# Camera rotation
R = blender_rotation_matrix(90, 0, 90)

print("="*70)
print("BLENDER CAMERA COORDINATE SYSTEM")
print("="*70)

print("\nRotation matrix for (90°, 0°, 90°):")
print(R)

# In Blender's default camera:
# - Camera looks down -Z axis
# - Camera +X is right
# - Camera +Y is up

# Apply rotation to these vectors
camera_forward = R @ np.array([0, 0, -1])
camera_right = R @ np.array([1, 0, 0])
camera_up = R @ np.array([0, 1, 0])

print("\nAfter rotation (90°, 0°, 90°):")
print(f"  Camera points in Blender direction: {camera_forward}")
print(f"  Camera right in Blender direction:  {camera_right}")
print(f"  Camera up in Blender direction:     {camera_up}")

# So the mapping from camera coords to Blender coords is:
# Blender = R @ Camera

# And from Blender coords to camera coords is:
# Camera = R^T @ Blender (since R is orthogonal)

R_inv = R.T

print("\n" + "="*70)
print("COORDINATE TRANSFORM")
print("="*70)
print("\nBlender to Camera: Camera = R^T @ Blender")
print("Inverse rotation matrix:")
print(R_inv)

# Test with a ball at Blender position (-0.004, -9.347, 2.278)
# and left camera at (18, -6, 3)

ball_blender = np.array([-0.004, -9.347, 2.278])
cam_blender = np.array([18, -6, 3])

# Vector from camera to ball in Blender coords
ball_from_cam_blender = ball_blender - cam_blender

print(f"\nTest: Frame 90")
print(f"  Ball in Blender: {ball_blender}")
print(f"  Camera in Blender: {cam_blender}")
print(f"  Ball from camera (Blender): {ball_from_cam_blender}")

# Transform to camera coords
ball_from_cam_camera = R_inv @ ball_from_cam_blender

print(f"  Ball from camera (Camera frame): {ball_from_cam_camera}")
print(f"    X (right): {ball_from_cam_camera[0]:.3f}m")
print(f"    Y (down):  {ball_from_cam_camera[1]:.3f}m")
print(f"    Z (forward): {ball_from_cam_camera[2]:.3f}m")

# Project to image
focal = 1600.0
cx, cy = 960.0, 540.0

if ball_from_cam_camera[2] > 0:
    img_x = focal * ball_from_cam_camera[0] / ball_from_cam_camera[2] + cx
    img_y = focal * ball_from_cam_camera[1] / ball_from_cam_camera[2] + cy
    
    print(f"\n  Projected to image: ({img_x:.1f}, {img_y:.1f})")
    print(f"  Actual detection:   (1630, 728)")
    print(f"  Error: ({img_x - 1630:.1f}, {img_y - 728:.1f})")
else:
    print(f"\n  Ball is behind camera! (Z={ball_from_cam_camera[2]:.3f})")
