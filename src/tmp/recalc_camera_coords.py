"""Calculate actual camera coords with baseline 1.442m for optimization"""
import sys
sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline
from volleyball_detection import get_ball_xy
import cv2
import pandas as pd
import numpy as np

# Load ground truth
df = pd.read_csv('ball_positions_blender.csv')

# Initialize pipeline
pipeline = HawkeyePipeline()

left_video = cv2.VideoCapture('data/left3.mp4')
right_video = cv2.VideoCapture('data/right3.mp4')

print("="*70)
print("RECALCULATING CAMERA COORDS WITH BASELINE 1.442m")
print("="*70)

camera_coords = {}

for frame in [90, 93, 96, 99]:
    left_video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    right_video.set(cv2.CAP_PROP_POS_FRAMES, frame)
    
    ret_l, left_frame = left_video.read()
    ret_r, right_frame = right_video.read()
    
    if ret_l and ret_r:
        # Get detections
        left_xy = get_ball_xy(left_frame)
        right_xy = get_ball_xy(right_frame)
        
        if left_xy != (None, None) and right_xy != (None, None):
            # Calculate camera coords using stereo triangulation
            focal = pipeline.config['focal_length_px']
            baseline = pipeline.config['baseline_m']
            
            disparity = left_xy[0] - right_xy[0]
            Z = (focal * baseline) / disparity
            
            cx, cy = 1920 // 2, 1080 // 2
            X = (left_xy[0] - cx) * Z / focal
            Y = (left_xy[1] - cy) * Z / focal
            
            camera_coords[frame] = np.array([X, Y, Z])
            
            gt = df[df['frame'] == frame].iloc[0]
            
            print(f"\nFrame {frame}:")
            print(f"  Detections: L=({left_xy[0]}, {left_xy[1]}), R=({right_xy[0]}, {right_xy[1]})")
            print(f"  Disparity: {disparity:.1f}px")
            print(f"  Camera coords: X={X:7.3f}, Y={Y:7.3f}, Z={Z:7.3f}")
            print(f"  Ground truth:  X={gt.x:7.3f}, Y={gt.y:7.3f}, Z={gt.z:7.3f}")

left_video.release()
right_video.release()

print("\n" + "="*70)
print("PYTHON CODE FOR optimize_camera_position.py:")
print("="*70)
print("\ncamera_coords = {")
for frame, coords in camera_coords.items():
    print(f"    {frame}: np.array([{coords[0]:.4f}, {coords[1]:.4f}, {coords[2]:.4f}]),")
print("}")
