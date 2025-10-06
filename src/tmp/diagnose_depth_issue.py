"""Analyze why depth is consistently wrong - ball always near camera"""
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from hawkeye_pipeline import HawkeyePipeline
from camera_config import load_camera_config

print("="*60)
print("DEPTH ANALYSIS - Why is ball always near camera edge?")
print("="*60)

config = load_camera_config()
pipeline = HawkeyePipeline(config)

print(f"\n1. CAMERA CALIBRATION PARAMETERS:")
print(f"   Baseline: {config['baseline_m']}m")
print(f"   Focal length: {config['focal_length_px']}px")
print(f"   Resolution: {config['resolution_width']}x{config['resolution_height']}")

print(f"\n2. CAMERA TRANSFORMATION MATRICES:")
print(f"   Translation (t): {pipeline.t}")
print(f"   Rotation (R):")
if isinstance(pipeline.R, np.ndarray):
    print(pipeline.R)
else:
    print(f"   Type: {type(pipeline.R)}")
    print(f"   Value: {pipeline.R}")

print(f"\n3. TESTING MULTIPLE FRAMES TO SEE DEPTH PATTERN:")
print("="*60)

import cv2
from volleyball_detection import get_ball_xy

depths = []
world_positions = []

for frame_num in [90, 93, 96, 99]:
    left_path = f"output_frames/left/left3_{frame_num:04d}.jpg"
    right_path = f"output_frames/right/right3_{frame_num:04d}.jpg"
    
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        continue
    
    # Get detections
    x_left, y_left = get_ball_xy(left_img)
    x_right, y_right = get_ball_xy(right_img)
    
    if x_left is None or x_right is None:
        continue
    
    disparity = x_left - x_right
    
    # Calculate depth using stereo formula: Z = (f * B) / d
    focal_length = config['focal_length_px']
    baseline = config['baseline_m']
    
    if disparity > 0:
        Z_camera = (focal_length * baseline) / disparity
        depths.append(Z_camera)
        
        # Process to get world coords
        result = pipeline.process_single_frame(frame_num)
        if result:
            cam = result['camera_coords']
            world = result['world_coords']
            world_positions.append(world)
            
            print(f"\nFrame {frame_num}:")
            print(f"  Detections: Left=({x_left}, {y_left}), Right=({x_right}, {y_right})")
            print(f"  Disparity: {disparity:.1f} pixels")
            print(f"  Camera space: X={cam[0]:.2f}, Y={cam[1]:.2f}, Z={cam[2]:.2f}m")
            print(f"  World space:  X={world[0]:.2f}, Y={world[1]:.2f}, Z={world[2]:.2f}m")
            print(f"  → Ball depth in WORLD: Z={world[2]:.2f}m")

print(f"\n" + "="*60)
print("ANALYSIS:")
print("="*60)

if depths:
    avg_depth_camera = np.mean(depths)
    print(f"\nAverage depth in CAMERA space: {avg_depth_camera:.2f}m")
    print(f"All depths in camera space: {[f'{d:.2f}' for d in depths]}")

if world_positions:
    world_z = [w[2] for w in world_positions if w[0] is not None]
    if world_z:
        avg_depth_world = np.mean(world_z)
        print(f"\nAverage depth in WORLD space: {avg_depth_world:.2f}m")
        print(f"All depths in world space: {[f'{z:.2f}' for z in world_z]}")
        
        if avg_depth_world < 10:
            print(f"\n⚠️  PROBLEM: Ball is consistently at ~{avg_depth_world:.1f}m depth!")
            print(f"   For synthetic Blender scene with cameras at 18m from court,")
            print(f"   this seems too close.")

print(f"\n" + "="*60)
print("POSSIBLE CAUSES:")
print("="*60)

print(f"\n1. CAMERA TRANSFORMATION (t, R) ISSUE:")
print(f"   The translation vector 't' should place the camera origin")
print(f"   at the actual camera location in the world.")
print(f"   Current t: {pipeline.t}")
print(f"   → Check if this matches your Blender camera positions")

print(f"\n2. COORDINATE SYSTEM MISMATCH:")
print(f"   OpenCV: X=right, Y=down, Z=forward")
print(f"   Blender: X=right, Y=forward, Z=up")
print(f"   Your transform code converts between these systems")
print(f"   → Verify the conversion is correct")

print(f"\n3. BASELINE OR FOCAL LENGTH WRONG:")
print(f"   Current baseline: {config['baseline_m']}m")
print(f"   Current focal length: {config['focal_length_px']}px")
print(f"   → These should match your actual Blender camera setup")

print(f"\n4. WORLD ORIGIN LOCATION:")
print(f"   Where is (0,0,0) in your Blender scene?")
print(f"   - Court center?")
print(f"   - One camera?")
print(f"   - Some other point?")
print(f"   → The 't' and 'R' matrices define this relationship")

print(f"\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)

print(f"\n1. Check your Blender scene:")
print(f"   - Left camera world position")
print(f"   - Right camera world position")
print(f"   - Court center position")
print(f"   - What is defined as world origin (0,0,0)?")

print(f"\n2. Verify in hawkeye_pipeline.py __init__:")
print(f"   - self.t = translation vector")
print(f"   - self.R = rotation matrix")
print(f"   These hardcoded values may not match your actual scene!")

print(f"\n3. Calculate correct t and R from Blender positions:")
print(f"   Would you like me to help calculate these from")
print(f"   your Blender camera positions?")
