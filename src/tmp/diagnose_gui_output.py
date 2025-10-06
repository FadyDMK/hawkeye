"""
Diagnose what the GUI is actually outputting vs what we expect.
Simulate the exact processing pipeline on frame 90.
"""

import sys
import cv2
import numpy as np
sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline
import pandas as pd

# Initialize pipeline
pipeline = HawkeyePipeline()

print("="*70)
print("PIPELINE CONFIGURATION")
print("="*70)
print(f"Baseline: {pipeline.config['baseline_m']}m")
print(f"Focal length: {pipeline.config['focal_length_px']}px")
print(f"Transform R:\n{np.array(pipeline.R)}")
print(f"Transform t: {pipeline.t}")

# Load frame 90 from videos
left_cap = cv2.VideoCapture('data/left3.mp4')
right_cap = cv2.VideoCapture('data/right3.mp4')

# Seek to frame 90
left_cap.set(cv2.CAP_PROP_POS_FRAMES, 90)
right_cap.set(cv2.CAP_PROP_POS_FRAMES, 90)

ret_left, left_frame = left_cap.read()
ret_right, right_frame = right_cap.read()

left_cap.release()
right_cap.release()

if not ret_left or not ret_right:
    print("ERROR: Could not read frame 90!")
    sys.exit(1)

print("\n" + "="*70)
print("PROCESSING FRAME 90")
print("="*70)

# Process through pipeline
pipeline.clear_previous_results()
result = pipeline.process_from_pair(left_frame, right_frame, frame_num=90, display=False)

if result and len(pipeline.ball_positions_world) > 0:
    world_pos = pipeline.ball_positions_world[-1]
    
    print(f"\nGUI Output:")
    print(f"  X = {world_pos[0]:.3f}m")
    print(f"  Y = {world_pos[1]:.3f}m")
    print(f"  Z = {world_pos[2]:.3f}m")
    
    # Load ground truth
    blender_df = pd.read_csv('ball_positions_blender.csv')
    blender_row = blender_df[blender_df['frame'] == 90].iloc[0]
    gt = np.array([blender_row['x'], blender_row['y'], blender_row['z']])
    
    print(f"\nGround Truth:")
    print(f"  X = {gt[0]:.3f}m")
    print(f"  Y = {gt[1]:.3f}m")
    print(f"  Z = {gt[2]:.3f}m")
    
    error = np.linalg.norm(world_pos - gt)
    print(f"\n3D Error: {error:.3f}m")
    print(f"Error vector: [{world_pos[0] - gt[0]:.3f}, {world_pos[1] - gt[1]:.3f}, {world_pos[2] - gt[2]:.3f}]")
    
    # Also check camera coords
    if len(pipeline.ball_positions_camera) > 0:
        cam_pos = pipeline.ball_positions_camera[-1]
        print(f"\nCamera coordinates:")
        print(f"  X = {cam_pos[0]:.3f}m")
        print(f"  Y = {cam_pos[1]:.3f}m")
        print(f"  Z = {cam_pos[2]:.3f}m")
        
        # Check what the mapping produces
        cam_blender = np.array([cam_pos[1], -cam_pos[0], -cam_pos[2]])
        print(f"\nAfter [+Y, -X, -Z] mapping:")
        print(f"  [{cam_blender[0]:.3f}, {cam_blender[1]:.3f}, {cam_blender[2]:.3f}]")
        
        # Apply transform manually
        R = np.array(pipeline.R)
        t = np.array(pipeline.t)
        world_manual = R @ cam_blender + t
        print(f"\nManual transform (R @ cam_blender + t):")
        print(f"  [{world_manual[0]:.3f}, {world_manual[1]:.3f}, {world_manual[2]:.3f}]")
else:
    print("\nERROR: No ball detected!")

print("\n" + "="*70)
