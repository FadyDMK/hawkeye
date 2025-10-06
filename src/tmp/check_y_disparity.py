"""Check Y disparities on frames 15-17"""
import sys
sys.path.append('src')
from volleyball_detection import get_ball_xy
import cv2

left_video = cv2.VideoCapture('data/left3.mp4')
right_video = cv2.VideoCapture('data/right3.mp4')

print("Checking Y disparities on frames where ball is between cameras:\n")

for frame_num in [15, 16, 17]:
    left_video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    right_video.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    ret_l, left_frame = left_video.read()
    ret_r, right_frame = right_video.read()
    
    if ret_l and ret_r:
        left_xy = get_ball_xy(left_frame)
        right_xy = get_ball_xy(right_frame)
        
        if left_xy != (None, None) and right_xy != (None, None):
            y_disp = left_xy[1] - right_xy[1]
            x_disp = left_xy[0] - right_xy[0]
            print(f"Frame {frame_num}:")
            print(f"  Left:  {left_xy}")
            print(f"  Right: {right_xy}")
            print(f"  X disparity: {x_disp:.1f} pixels")
            print(f"  Y disparity: {y_disp:.1f} pixels")
            print()

left_video.release()
right_video.release()
