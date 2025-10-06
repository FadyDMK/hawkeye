"""Diagnostic script to see why world coords are None"""
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'court_detection'))

from hawkeye_pipeline import HawkeyePipeline
from camera_config import load_camera_config

print("="*60)
print("WORLD COORDS DIAGNOSTIC")
print("="*60)

# Load config
config = load_camera_config()
print(f"\nConfig:")
print(f"  Court: {config.get('court_length_m')}m x {config.get('court_width_m')}m")
print(f"  Max height: {config.get('max_ball_height_m')}m")
print(f"  Baseline: {config['baseline_m']}m")
print(f"  Focal length: {config['focal_length_px']}px")

# Create pipeline
pipeline = HawkeyePipeline(config)

print(f"\nCamera transform matrices:")
print(f"  Translation (t): {pipeline.t.flatten() if hasattr(pipeline.t, 'flatten') else pipeline.t}")
print(f"  Rotation (R): {pipeline.R.shape if hasattr(pipeline.R, 'shape') else type(pipeline.R)}")

# Test frame 99
print(f"\n" + "="*60)
print("Testing frame 99:")
print("="*60)

import cv2
left_img = cv2.imread("output_frames/left/left3_0099.jpg")
right_img = cv2.imread("output_frames/right/right3_0099.jpg")

if left_img is None or right_img is None:
    print("❌ Could not load frames!")
    exit()

# Process with detailed output
from stereo_matching import StereoMatching
from volleyball_detection import get_ball_xy

# Manual processing to see each step
print("\n1. Detection:")
x_left, y_left = get_ball_xy(left_img)
x_right, y_right = get_ball_xy(right_img)
print(f"   Left: ({x_left}, {y_left})")
print(f"   Right: ({x_right}, {y_right})")

print("\n2. Stereo triangulation:")
stereo = StereoMatching(left_img, right_img, config)
success = stereo.try_detection_triangulation()
print(f"   Success: {success}")

if success:
    camera_coords = (stereo.X_ball, stereo.Y_ball, stereo.Z_ball)
    print(f"   Camera coords: {camera_coords}")
    
    print("\n3. Camera to World transform:")
    from transforms import ball_camera_to_world
    world_coords = ball_camera_to_world(camera_coords, pipeline.t, pipeline.R)
    print(f"   World coords (raw): {world_coords}")
    
    print("\n4. Court bounds validation:")
    x, y, z = world_coords
    
    court_width = config.get("court_width_m", 9.0)
    court_length = config.get("court_length_m", 18.0)
    max_height = config.get("max_ball_height_m", 15.0)
    
    width_tolerance = 3.0
    length_tolerance = 3.0
    
    x_valid = abs(x) <= (court_width / 2 + width_tolerance)
    y_valid = abs(y) <= (court_length / 2 + length_tolerance)
    z_valid = 0 <= z <= max_height
    
    print(f"   Court dimensions: {court_length}m x {court_width}m")
    print(f"   Max height: {max_height}m")
    print(f"   Tolerance: ±{length_tolerance}m (length), ±{width_tolerance}m (width)")
    print(f"\n   X check: {x:.2f}m, valid if |x| <= {court_width/2 + width_tolerance:.2f}m → {x_valid} {'✅' if x_valid else '❌'}")
    print(f"   Y check: {y:.2f}m, valid if |y| <= {court_length/2 + length_tolerance:.2f}m → {y_valid} {'✅' if y_valid else '❌'}")
    print(f"   Z check: {z:.2f}m, valid if 0 <= z <= {max_height:.2f}m → {z_valid} {'✅' if z_valid else '❌'}")
    
    overall_valid = x_valid and y_valid and z_valid
    print(f"\n   Overall validation: {overall_valid} {'✅' if overall_valid else '❌'}")
    
    if not overall_valid:
        print(f"\n⚠️  PROBLEM: Ball position is outside court bounds!")
        print(f"   This is why world_coords returns (None, None, None)")
        print(f"\n   SOLUTIONS:")
        print(f"   1. Check camera transformation matrices (t and R)")
        print(f"   2. Adjust court dimensions in config if they're wrong")
        print(f"   3. Increase tolerance values")
        print(f"   4. Verify Blender scene coordinate system matches OpenCV")
else:
    print("❌ Stereo triangulation failed!")

print("\n" + "="*60)
