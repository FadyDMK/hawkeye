"""
Test hypothesis: Maybe the camera is actually on the OPPOSITE side of the court,
or the coordinate system is flipped.

Try different camera positions and see which one makes the ball appear in front of the camera.
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

# Frame 90 test case
ball_blender = np.array([-0.004, -9.347, 2.278])

# Test different camera positions
test_cases = [
    ("Original (18, -6, 3)", np.array([18, -6, 3]), (90, 0, 90)),
    ("Flipped X (-18, -6, 3)", np.array([-18, -6, 3]), (90, 0, 90)),
    ("Flipped X (-18, -6, 3), rot opposite", np.array([-18, -6, 3]), (-90, 0, -90)),
    ("Original, rot flipped", np.array([18, -6, 3]), (-90, 0, -90)),
    ("Original, rot (0, 0, 0)", np.array([18, -6, 3]), (0, 0, 0)),
    ("Flipped X, rot (0, 0, 0)", np.array([-18, -6, 3]), (0, 0, 0)),
]

focal = 1600.0
cx, cy = 960.0, 540.0

print("="*70)
print("TESTING DIFFERENT CAMERA CONFIGURATIONS")
print("="*70)

for name, cam_pos, rot in test_cases:
    R = blender_rotation_matrix(*rot)
    R_inv = R.T
    
    # Vector from camera to ball
    ball_from_cam_blender = ball_blender - cam_pos
    
    # Transform to camera coords
    ball_from_cam_camera = R_inv @ ball_from_cam_blender
    
    print(f"\n{name}:")
    print(f"  Rotation: {rot}")
    print(f"  Ball from camera (Blender): [{ball_from_cam_blender[0]:7.3f}, {ball_from_cam_blender[1]:7.3f}, {ball_from_cam_blender[2]:7.3f}]")
    print(f"  Ball from camera (Camera):  [{ball_from_cam_camera[0]:7.3f}, {ball_from_cam_camera[1]:7.3f}, {ball_from_cam_camera[2]:7.3f}]")
    
    if ball_from_cam_camera[2] > 0:
        img_x = focal * ball_from_cam_camera[0] / ball_from_cam_camera[2] + cx
        img_y = focal * ball_from_cam_camera[1] / ball_from_cam_camera[2] + cy
        
        print(f"  ✅ Ball in front! Z={ball_from_cam_camera[2]:.3f}m")
        print(f"  Projected: ({img_x:.1f}, {img_y:.1f})")
        print(f"  Actual:    (1630, 728)")
        error = np.sqrt((img_x - 1630)**2 + (img_y - 728)**2)
        print(f"  Error: {error:.1f}px")
    else:
        print(f"  ❌ Ball behind camera! Z={ball_from_cam_camera[2]:.3f}m")

print("\n" + "="*70)
print("Looking for configuration with ball in front AND low projection error...")
print("="*70)
