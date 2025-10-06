"""
Optimize scale factors for MKV videos using ground truth data.
Uses frames from output_frames folder with ground truth from Blender.
"""

import sys
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import os

sys.path.append('src')

# Import the transformation function directly
sys.path.append('src/court_detection')
from transforms import ball_camera_to_world

print("="*80)
print("OPTIMIZING SCALE FACTORS FOR MKV VIDEOS")
print("="*80)

# Load ground truth from the MKV video's Blender file
gt_path = "3D-models/Latest volley go brr/ball_positions_blender.csv"
blender_df = pd.read_csv(gt_path)

print(f"\n✅ Loaded ground truth: {len(blender_df)} frames")
print(f"   Position ranges:")
print(f"   X: {blender_df['x'].min():.4f} to {blender_df['x'].max():.4f}")
print(f"   Y: {blender_df['y'].min():.4f} to {blender_df['y'].max():.4f}")
print(f"   Z: {blender_df['z'].min():.4f} to {blender_df['z'].max():.4f}")

# Test frames where we have both detections and ground truth
test_frames = [85, 90, 93, 95, 96, 99]

print(f"\n🔍 Collecting camera coordinates from output_frames...")

# Collect camera coordinates by processing frames
from hawkeye_pipeline import HawkeyePipeline
pipeline = HawkeyePipeline()

camera_coords_list = []
ground_truth_list = []

for frame_num in test_frames:
    # Load frames from output_frames folder
    root = os.path.dirname(os.path.abspath(__file__))
    left_path = os.path.join(root, "output_frames", "left", f"left3_{frame_num:04d}.jpg")
    right_path = os.path.join(root, "output_frames", "right", f"right3_{frame_num:04d}.jpg")
    
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"  ❌ Frame {frame_num}: Not found")
        continue
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"  ❌ Frame {frame_num}: Failed to load")
        continue
    
    # Process to get camera coordinates (before transformation)
    pipeline.clear_previous_results()
    
    # Get detection and triangulation
    from stereo_matching import StereoMatching
    stereo_matcher = StereoMatching(left_img, right_img, config=pipeline.config)
    
    if stereo_matcher.try_detection_triangulation():
        camera_coords = (stereo_matcher.X_ball, stereo_matcher.Y_ball, stereo_matcher.Z_ball)
        
        if None not in camera_coords and frame_num in blender_df['frame'].values:
            row = blender_df[blender_df['frame'] == frame_num].iloc[0]
            gt = np.array([row['x'], row['y'], row['z']])
            
            camera_coords_list.append(camera_coords)
            ground_truth_list.append(gt)
            
            print(f"  ✅ Frame {frame_num}:")
            print(f"     Camera: ({camera_coords[0]:.3f}, {camera_coords[1]:.3f}, {camera_coords[2]:.3f})")
            print(f"     Ground truth: ({gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f})")
        else:
            print(f"  ⚠️  Frame {frame_num}: No detection or no ground truth")
    else:
        print(f"  ❌ Frame {frame_num}: Detection failed")

if len(camera_coords_list) < 3:
    print(f"\n❌ ERROR: Need at least 3 frames with valid detections. Got {len(camera_coords_list)}")
    print("   Try adjusting detection confidence or using different frames.")
    sys.exit(1)

print(f"\n✅ Collected {len(camera_coords_list)} valid frames for optimization")

# Current transformation parameters
t = np.array([-0.170, 18.827, 0.249])
R = np.eye(3)

# Optimization function
def error_function(params):
    scale = params[:3]
    
    total_error = 0
    for cam_coords, gt in zip(camera_coords_list, ground_truth_list):
        # Apply transformation
        world = ball_camera_to_world(cam_coords, t, R, scale)
        
        # Calculate error
        error = np.linalg.norm(world - gt)
        total_error += error ** 2
    
    return total_error / len(camera_coords_list)

# Initial guess (current scale factors)
initial_scale = [0.0748, 3.6928, -0.1101]

print(f"\n🔧 Optimizing scale factors...")
print(f"   Initial scale: {initial_scale}")
print(f"   Initial error: {np.sqrt(error_function(initial_scale)):.3f}m")

# Optimize
result = minimize(
    error_function,
    initial_scale,
    method='Nelder-Mead',
    options={'maxiter': 5000, 'xatol': 1e-6, 'fatol': 1e-6}
)

optimal_scale = result.x
final_error = np.sqrt(result.fun)

print(f"\n✅ OPTIMIZATION COMPLETE!")
print(f"   Optimal scale: [{optimal_scale[0]:.4f}, {optimal_scale[1]:.4f}, {optimal_scale[2]:.4f}]")
print(f"   Final error: {final_error:.3f}m")
print(f"   Improvement: {np.sqrt(error_function(initial_scale)) / final_error:.2f}x better")

# Test the optimized scale on each frame
print(f"\n📊 Per-frame accuracy with optimized scale:")
errors = []
for i, (cam_coords, gt) in enumerate(zip(camera_coords_list, ground_truth_list)):
    world = ball_camera_to_world(cam_coords, t, R, optimal_scale)
    error = np.linalg.norm(world - gt)
    errors.append(error)
    
    frame_num = test_frames[i]
    print(f"  Frame {frame_num}:")
    print(f"    Predicted: ({world[0]:.3f}, {world[1]:.3f}, {world[2]:.3f})")
    print(f"    GT:        ({gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f})")
    print(f"    Error:     {error:.3f}m")

print(f"\n📈 Statistics:")
print(f"   Average error: {np.mean(errors):.3f}m")
print(f"   Min error:     {np.min(errors):.3f}m")
print(f"   Max error:     {np.max(errors):.3f}m")
print(f"   Std dev:       {np.std(errors):.3f}m")

print(f"\n" + "="*80)
print("TO APPLY THESE SCALE FACTORS:")
print("="*80)
print(f"Edit src/hawkeye_pipeline.py line 48:")
print(f"  OLD: self.scale = [0.0748, 3.6928, -0.1101]")
print(f"  NEW: self.scale = [{optimal_scale[0]:.4f}, {optimal_scale[1]:.4f}, {optimal_scale[2]:.4f}]")
print("="*80)
