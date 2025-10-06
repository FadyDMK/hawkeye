"""
Test multiple frames to see if any fail bounds checking.
"""

import sys
import cv2
import numpy as np
sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline

# Initialize pipeline
pipeline = HawkeyePipeline()

# Load videos
left_cap = cv2.VideoCapture('data/left3.mp4')
right_cap = cv2.VideoCapture('data/right3.mp4')

# Test frames 85-100
print("="*70)
print("TESTING FRAMES 1-20")
print("="*70)

successful = []
failed = []

for frame_num in range(1, 21):
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()
    
    if not ret_left or not ret_right:
        continue
    
    # Process through pipeline
    pipeline.clear_previous_results()
    result = pipeline.process_from_pair(left_frame, right_frame, frame_num=frame_num, display=False)
    
    if result and len(pipeline.ball_positions_world) > 0:
        world_pos = pipeline.ball_positions_world[-1]
        if None in world_pos:
            failed.append(frame_num)
            print(f"Frame {frame_num}: FAILED (got None)")
        else:
            successful.append(frame_num)
            print(f"Frame {frame_num}: SUCCESS - ({world_pos[0]:.3f}, {world_pos[1]:.3f}, {world_pos[2]:.3f})")
    else:
        failed.append(frame_num)
        print(f"Frame {frame_num}: FAILED (no detection)")

left_cap.release()
right_cap.release()

print("\n" + "="*70)
print(f"Successful: {len(successful)}/{len(successful) + len(failed)}")
print(f"Failed frames: {failed}")
print("="*70)
