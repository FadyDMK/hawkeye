"""
Work backwards: given the disparities and ground truth, what must the camera geometry be?

We know:
- Disparities: 188, 186, 186, 183 pixels
- Ground truth world positions
- Focal length: 1600 pixels

Let's solve for the actual camera positions that would produce these observations.
"""

import numpy as np
import pandas as pd

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Known values
focal = 1600.0
cx, cy = 960.0, 540.0

# Measured disparities and detections
measurements = {
    90: {'left': (1630, 728), 'right': (1442, 726), 'disparity': 188},
    93: {'left': (1580, 714), 'right': (1394, 713), 'disparity': 186},
    96: {'left': (1523, 702), 'right': (1337, 702), 'disparity': 186},
    99: {'left': (1458, 690), 'right': (1275, 687), 'disparity': 183}
}

print("="*70)
print("REVERSE-ENGINEERING CAMERA GEOMETRY")
print("="*70)

for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        blender_row = blender_df[blender_df['frame'] == frame].iloc[0]
        world_pos = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
        
        meas = measurements[frame]
        left_x, left_y = meas['left']
        disp = meas['disparity']
        
        print(f"\nFrame {frame}:")
        print(f"  World position: X={world_pos[0]:.3f}, Y={world_pos[1]:.3f}, Z={world_pos[2]:.3f}")
        print(f"  Left detection: ({left_x}, {left_y})")
        print(f"  Disparity: {disp}px")
        
        # The ball is at world_pos. The left camera sees it at (left_x, left_y).
        # What must the camera position and orientation be?
        
        # If camera is at Blender position (18, -6, 3) with rotation (90, 0, 90):
        # - Camera faces -X direction in Blender
        # - Camera Y axis points up in Blender (+Z)
        # - Camera X axis points right in image, which is +Y in Blender
        
        # Vector from nominal camera to ball in Blender coords
        cam_blender = np.array([18, -6, 3])
        ball_from_cam = world_pos - cam_blender
        
        print(f"  Ball from camera (18,-6,3): X={ball_from_cam[0]:.3f}, Y={ball_from_cam[1]:.3f}, Z={ball_from_cam[2]:.3f}")
        print(f"  Distance: {np.linalg.norm(ball_from_cam):.3f}m")
        
        # With camera rotation (90, 0, 90), the camera coordinate frame is:
        # Camera +Z (forward) = Blender -X
        # Camera +X (right) = Blender +Y  
        # Camera +Y (down) = Blender -Z
        
        # So in camera coords:
        cam_X = ball_from_cam[1]   # Blender Y → Camera X
        cam_Y = -ball_from_cam[2]  # Blender -Z → Camera Y
        cam_Z = -ball_from_cam[0]  # Blender -X → Camera Z
        
        print(f"  In camera frame: X={cam_X:.3f}, Y={cam_Y:.3f}, Z={cam_Z:.3f}")
        
        # Project to image
        img_x = focal * cam_X / cam_Z + cx
        img_y = focal * cam_Y / cam_Z + cy
        
        print(f"  Projected to image: ({img_x:.1f}, {img_y:.1f})")
        print(f"  Actual detection:   ({left_x}, {left_y})")
        print(f"  Difference: ({img_x - left_x:.1f}, {img_y - left_y:.1f})")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("If the projections don't match detections, then either:")
print("1. The camera position (18,-6,3) is not correct")
print("2. The camera rotation (90,0,90) is not correct")
print("3. The coordinate system interpretation is wrong")
print("4. The focal length 1600px is not correct")
