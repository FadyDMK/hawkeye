import cv2
import sys
import os
import json

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from volleyball_detection import get_ball_xy
from stereo_matching import StereoMatching

# Load config
with open('config/camera_config.json') as f:
    config = json.load(f)

print("="*60)
print("FULL SYSTEM TEST - NEW SEQUENCE (left3/right3)")
print("="*60)
print(f"Config: baseline={config['baseline_m']}m, z_range=[{config['z_min_m']}, {config['z_max_m']}]m")
print()

# Test frames 95-99
success_count = 0
total_count = 5

for frame_num in range(95, 100):
    left_path = f"output_frames/left/left3_{frame_num:04d}.jpg"
    right_path = f"output_frames/right/right3_{frame_num:04d}.jpg"
    
    # Load images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)
    
    if left_img is None or right_img is None:
        print(f"❌ Frame {frame_num}: Files not found")
        continue
    
    # Detect ball
    x_left, y_left = get_ball_xy(left_img)
    x_right, y_right = get_ball_xy(right_img)
    
    if x_left is None or x_right is None:
        print(f"❌ Frame {frame_num}: Detection failed")
        continue
    
    # Calculate disparity
    disparity = x_left - x_right
    
    # Try to triangulate using StereoMatching
    stereo = StereoMatching(left_img, right_img, config)
    success = stereo.try_detection_triangulation()
    
    if success:
        x, y, z = stereo.X_ball, stereo.Y_ball, stereo.Z_ball
        print(f"✅ Frame {frame_num}: Ball at ({x:.2f}, {y:.2f}, {z:.2f})m, disparity={disparity}px")
        success_count += 1
    else:
        print(f"❌ Frame {frame_num}: Triangulation failed (disparity={disparity}px)")

print()
print("="*60)
print(f"SUMMARY: {success_count}/{total_count} frames processed successfully")
print(f"Success rate: {100*success_count/total_count:.1f}%")

if success_count == total_count:
    print("\n🎉 SYSTEM WORKING PERFECTLY!")
elif success_count > 0:
    print(f"\n⚠️  PARTIAL SUCCESS - {success_count} out of {total_count} frames worked")
else:
    print("\n❌ SYSTEM NOT WORKING - Check configuration")
