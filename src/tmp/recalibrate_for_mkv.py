"""
Recalibrate scale factors for the MKV videos (newnewLeft.mkv, newnewRight.mkv)
using the pre-extracted frames in output_frames folder.

This will find the optimal scale factors for YOUR videos.
"""

import sys
import cv2
import numpy as np
from scipy.optimize import minimize
import os

sys.path.append('src')
from hawkeye_pipeline import HawkeyePipeline

print("="*80)
print("RECALIBRATING SCALE FACTORS FOR MKV VIDEOS")
print("="*80)

# Initialize pipeline
pipeline = HawkeyePipeline()

# Test frames to use for calibration
# We'll use frames where you know the ball position or can manually measure it
# For now, let's process a few frames to see what we get
test_frames = [85, 90, 95, 100]

print("\n📊 Current configuration:")
print(f"  Baseline: {pipeline.config['baseline_m']}m")
print(f"  Scale: {pipeline.scale}")
print(f"  Translation: {pipeline.t}")

print("\n🔍 Processing test frames to find camera coordinates...")
print("   (These are the raw coordinates before scaling)")

results = []
for frame_num in test_frames:
    # Load frames from output_frames folder
    root = os.path.dirname(os.path.abspath(__file__))
    left_path = os.path.join(root, "output_frames", "left", f"left3_{frame_num:04d}.jpg")
    right_path = os.path.join(root, "output_frames", "right", f"right3_{frame_num:04d}.jpg")
    
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"  Frame {frame_num}: Not found")
        continue
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"  Frame {frame_num}: Failed to load")
        continue
    
    # Process without display
    pipeline.clear_previous_results()
    result = pipeline.process_from_pair(left_img, right_img, frame_num=frame_num, display=False)
    
    if result and len(pipeline.ball_positions_camera) > 0:
        camera_coords = pipeline.ball_positions_camera[-1]
        world_coords = pipeline.ball_positions_world[-1] if len(pipeline.ball_positions_world) > 0 else None
        
        results.append({
            'frame': frame_num,
            'camera': camera_coords,
            'world': world_coords
        })
        
        print(f"\n  Frame {frame_num}:")
        print(f"    Camera coords: ({camera_coords[0]:.3f}, {camera_coords[1]:.3f}, {camera_coords[2]:.3f})")
        if world_coords is not None and isinstance(world_coords, (list, tuple, np.ndarray)):
            if None not in world_coords:
                print(f"    World coords:  ({world_coords[0]:.3f}, {world_coords[1]:.3f}, {world_coords[2]:.3f})")
            else:
                print(f"    World coords:  REJECTED (out of bounds)")
        else:
            print(f"    World coords:  None")
    else:
        print(f"  Frame {frame_num}: No detection")

print("\n" + "="*80)
print("NEXT STEPS:")
print("="*80)
print("Since we don't have ground truth for the MKV videos, we need to:")
print()
print("OPTION 1: Manual measurement")
print("  1. Pick a few frames where you can see the ball clearly")
print("  2. Manually measure/estimate the ball position in world coordinates")
print("  3. Use those measurements to optimize scale factors")
print()
print("OPTION 2: Use known court landmarks")
print("  1. Find frames where ball touches known locations (baseline, net, etc.)")
print("  2. Use those positions to calibrate")
print()
print("OPTION 3: Extract ground truth from Blender for MKV videos")
print("  1. If these MKV videos also came from Blender, extract the ball positions")
print("  2. Save them to a CSV file (like ball_positions_blender.csv)")
print("  3. Run optimization to find scale factors")
print()
print("Which option would you like to use?")
print("Or provide frame numbers and their expected world coordinates for calibration.")
print("="*80)
