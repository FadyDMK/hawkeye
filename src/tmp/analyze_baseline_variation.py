"""
Check what baseline would be required for each frame individually
to get the EXACT ground truth position.

This will tell us if baseline variation is the issue.
"""

import numpy as np
import pandas as pd

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Measured disparities and detections
measurements = {
    90: {'left': (1630, 728), 'disparity': 188},
    93: {'left': (1580, 714), 'disparity': 186},
    96: {'left': (1523, 702), 'disparity': 186},
    99: {'left': (1458, 690), 'disparity': 183}
}

focal = 1600.0
cx, cy = 960.0, 540.0

print("="*70)
print("WHAT BASELINE + MAPPING WOULD GIVE PERFECT RESULTS?")
print("="*70)

# For each frame, work backwards from ground truth to required camera coords
# Then calculate what baseline would produce those camera coords

for frame in [90, 93, 96, 99]:
    if frame in blender_df['frame'].values:
        row = blender_df[blender_df['frame'] == frame].iloc[0]
        gt = np.array([row['x'], row['y'], row['z']])
        
        meas = measurements[frame]
        left_x, left_y = meas['left']
        disp = meas['disparity']
        
        print(f"\nFrame {frame}:")
        print(f"  Ground truth: [{gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f}]")
        print(f"  Detection: ({left_x}, {left_y}), disparity: {disp}px")
        
        # With current baseline 2.115m, we get Z from disparity
        baseline_current = 2.115
        Z_current = (focal * baseline_current) / disp
        
        print(f"  Current baseline {baseline_current}m → Z = {Z_current:.3f}m")
        
        # What baseline would give us Z equal to distance from camera to ball?
        # Distance from camera at (-18, -6, 3) to ball
        cam_nominal = np.array([-18, -6, 3])  # or (18, -6, 3)?
        
        dist_from_cam_neg18 = np.linalg.norm(gt - np.array([-18, -6, 3]))
        dist_from_cam_pos18 = np.linalg.norm(gt - np.array([18, -6, 3]))
        
        print(f"  Distance from camera at (-18,-6,3): {dist_from_cam_neg18:.3f}m")
        print(f"  Distance from camera at (18,-6,3): {dist_from_cam_pos18:.3f}m")
        
        # What baseline would give Z = dist_from_cam?
        baseline_needed_neg18 = (dist_from_cam_neg18 * disp) / focal
        baseline_needed_pos18 = (dist_from_cam_pos18 * disp) / focal
        
        print(f"  Required baseline for Z={dist_from_cam_neg18:.3f}m: {baseline_needed_neg18:.3f}m")
        print(f"  Required baseline for Z={dist_from_cam_pos18:.3f}m: {baseline_needed_pos18:.3f}m")
        
        # Current vs required
        print(f"  Current baseline: {baseline_current:.3f}m")
        print(f"  Ratio (needed/current): {baseline_needed_neg18/baseline_current:.3f}x")

print("\n" + "="*70)
print("CONCLUSION:")
print("="*70)
print("If required baseline varies significantly between frames,")
print("then a single baseline value cannot work for all frames.")
print("This suggests either:")
print("1. Disparity measurements are systematically wrong")
print("2. Ground truth distances are wrong")
print("3. Focal length is wrong")
print("4. There's lens distortion affecting measurements")
print("="*70)
