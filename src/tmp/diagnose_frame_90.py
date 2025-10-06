"""Detailed diagnostic of frame 90 to see what's happening"""
import sys
sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline
import cv2
import pandas as pd
import numpy as np

# Load ground truth
df = pd.read_csv('ball_positions_blender.csv')
gt = df[df['frame'] == 90].iloc[0]

print("="*70)
print("DETAILED FRAME 90 DIAGNOSTIC")
print("="*70)
print(f"\nGround Truth: X={gt.x:.3f}m, Y={gt.y:.3f}m, Z={gt.z:.3f}m")

# Initialize pipeline
pipeline = HawkeyePipeline()

print(f"\nPipeline config:")
print(f"  Baseline: {pipeline.config['baseline_m']}m")
print(f"  Focal: {pipeline.config['focal_length_px']}px")
print(f"  Transform t: {pipeline.t}")
print(f"  Transform R: {pipeline.R}")

# Load frame 90
left_video = cv2.VideoCapture('data/left3.mp4')
right_video = cv2.VideoCapture('data/right3.mp4')

left_video.set(cv2.CAP_PROP_POS_FRAMES, 90)
right_video.set(cv2.CAP_PROP_POS_FRAMES, 90)

ret_l, left_frame = left_video.read()
ret_r, right_frame = right_video.read()

print(f"\nProcessing frame 90...")

# Get detections
from volleyball_detection import get_ball_xy
left_xy = get_ball_xy(left_frame)
right_xy = get_ball_xy(right_frame)

print(f"\nDetections:")
print(f"  Left: {left_xy}")
print(f"  Right: {right_xy}")
print(f"  X disparity: {left_xy[0] - right_xy[0]:.1f}px")
print(f"  Y disparity: {left_xy[1] - right_xy[1]:.1f}px")

# Expected depth
focal = 1600.0
baseline = 1.442
x_disp = left_xy[0] - right_xy[0]
expected_depth = (focal * baseline) / x_disp
print(f"\nExpected depth from disparity: {expected_depth:.2f}m")

# Process through pipeline
result = pipeline.process_from_pair(left_frame, right_frame, frame_num=90)

if result and result.get('world_coords') is not None:
    cam_coords = result['camera_coords']
    world_coords = result['world_coords']
    
    print(f"\nCamera coordinates: X={cam_coords[0]:.3f}, Y={cam_coords[1]:.3f}, Z={cam_coords[2]:.3f}")
    print(f"World coordinates:  X={world_coords[0]:.3f}, Y={world_coords[1]:.3f}, Z={world_coords[2]:.3f}")
    
    # Manual calculation to verify
    print(f"\n--- Manual verification ---")
    ball_pos = np.array([[cam_coords[0]], [cam_coords[1]], [cam_coords[2]]])
    
    # Apply coordinate mapping: [-Z, X, -Y]
    ball_pos_blender = np.array([-ball_pos[2,0], ball_pos[0,0], -ball_pos[1,0]]).reshape(3,1)
    print(f"After coord mapping [-Z, X, -Y]: {ball_pos_blender.flatten()}")
    
    # Apply transform
    R = np.array(pipeline.R)
    t = np.array(pipeline.t).reshape(3,1)
    world_manual = (R @ ball_pos_blender) + t
    print(f"After transform (R @ pos + t): {world_manual.flatten()}")
    
    # Compare to ground truth
    error = np.array(world_coords) - np.array([gt.x, gt.y, gt.z])
    error_mag = np.linalg.norm(error)
    
    print(f"\nComparison to ground truth:")
    print(f"  Error: X={error[0]:.3f}m, Y={error[1]:.3f}m, Z={error[2]:.3f}m")
    print(f"  3D Error: {error_mag:.3f}m")
else:
    print("\n❌ Pipeline returned None or invalid world coords")

left_video.release()
right_video.release()
