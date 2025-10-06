"""
Optimize BOTH scale factors AND translation vector for MKV videos.
Previous attempts only optimized scale, keeping t fixed from MP4 calibration.
"""

import sys
import cv2
import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.append('src')

# Load ground truth
gt_path = "3D-models/Latest volley go brr/ball_positions_blender.csv"
blender_df = pd.read_csv(gt_path)

# Test frames
test_frames = [85, 90, 93, 95, 96, 99]

# Fixed rotation (identity)
R = np.eye(3)

def ball_camera_to_world_mkv(ball_pos, t, R, scale):
    """
    MKV-specific coordinate transformation with correct mapping [+Y, -Z, -X]
    """
    # NEW MAPPING FOR MKV: [+Y, -Z, -X]
    ball_pos_blender = np.array([
        ball_pos[1],   # +Y
        -ball_pos[2],  # -Z
        -ball_pos[0]   # -X
    ])
    
    if scale is not None:
        ball_pos_blender = ball_pos_blender * scale
    
    world = (R @ ball_pos_blender) + t
    return world

def objective(params, data_points):
    """
    Objective function: minimize RMS error
    params: [scale_x, scale_y, scale_z, t_x, t_y, t_z]
    """
    scale = params[:3]
    t = params[3:]
    
    errors = []
    for cam_coords, gt in data_points:
        pred = ball_camera_to_world_mkv(cam_coords, t, R, scale)
        error = np.linalg.norm(pred - gt)
        errors.append(error)
    
    return np.sqrt(np.mean(np.array(errors)**2))

# Collect data points
from hawkeye_pipeline import HawkeyePipeline
pipeline = HawkeyePipeline()
pipeline.clear_previous_results()

from stereo_matching import StereoMatching

data_points = []

print("Collecting frames for full optimization (scale + translation)...")
for frame_num in test_frames:
    left_path = f"output_frames/left/left3_{frame_num:04d}.jpg"
    right_path = f"output_frames/right/right3_{frame_num:04d}.jpg"
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        continue
    
    stereo_matcher = StereoMatching(left_img, right_img, config=pipeline.config)
    
    if stereo_matcher.try_detection_triangulation():
        cam_coords = np.array([stereo_matcher.X_ball, stereo_matcher.Y_ball, stereo_matcher.Z_ball])
        
        # Get ground truth
        row = blender_df[blender_df['frame'] == frame_num].iloc[0]
        gt = np.array([row['x'], row['y'], row['z']])
        
        data_points.append((cam_coords, gt))
        print(f"Frame {frame_num}: Camera {cam_coords}, GT {gt}")

print(f"\nCollected {len(data_points)} valid frames")

if len(data_points) < 3:
    print("Not enough data points!")
    sys.exit(1)

# Initial parameters: scale from previous optimization + translation from MP4
initial_params = np.array([
    0.12977, 1.4241, 0.0521,  # scale (from previous)
    -0.170, 18.827, 0.249      # translation (from MP4)
])

print(f"\nInitial params:")
print(f"  Scale: {initial_params[:3]}")
print(f"  Translation: {initial_params[3:]}")
initial_error = objective(initial_params, data_points)
print(f"  Initial RMS error: {initial_error:.3f}m")

# Optimize both scale and translation
print("\nOptimizing scale AND translation together...")
result = minimize(
    objective,
    initial_params,
    args=(data_points,),
    method='Nelder-Mead',
    options={'maxiter': 2000, 'disp': False}
)

optimal_params = result.x
final_error = result.fun

print("="*80)
print(f"Optimal scale: {optimal_params[:3]}")
print(f"Optimal translation: {optimal_params[3:]}")
print(f"Final RMS error: {final_error:.3f}m")
print("="*80)

# Test on each frame
print("\nPer-frame results:")
for i, (cam_coords, gt) in enumerate(data_points):
    pred = ball_camera_to_world_mkv(cam_coords, optimal_params[3:], R, optimal_params[:3])
    error = np.linalg.norm(pred - gt)
    frame_num = test_frames[i]
    print(f"Frame {frame_num}: Error {error:.3f}m")
    print(f"  Predicted: ({pred[0]:.3f}, {pred[1]:.3f}, {pred[2]:.3f})")
    print(f"  GT:        ({gt[0]:.3f}, {gt[1]:.3f}, {gt[2]:.3f})")

print("="*80)
print("UPDATE hawkeye_pipeline.py:")
print(f"  self.scale = {optimal_params[:3].tolist()}")
print(f"  self.t = {optimal_params[3:].tolist()}")
print("="*80)
