"""
Check the actual ball positions in left vs right images to understand disparity direction.
"""

import sys
import cv2
sys.path.append('src')

from volleyball_detection import get_ball_xy

# Test frame 90
left_img = cv2.imread("output_frames/left/left3_0090.jpg")
right_img = cv2.imread("output_frames/right/right3_0090.jpg")

print("Frame 90 ball detection:")
print("="*60)

x_left, y_left = get_ball_xy(left_img)
print(f"Left image:  x={x_left}, y={y_left}")

x_right, y_right = get_ball_xy(right_img)
print(f"Right image: x={x_right}, y={y_right}")

print(f"\nDisparities:")
print(f"  X disparity (x_left - x_right): {x_left - x_right if x_left and x_right else 'N/A'}")
print(f"  Y disparity (y_left - y_right): {y_left - y_right if y_left and y_right else 'N/A'}")

print(f"\nCamera positions:")
print(f"  Left camera:  (18, -3, 3) - closer to court center (Y=-3)")
print(f"  Right camera: (18, -6, 3) - further from court center (Y=-6)")

print(f"\nExpected behavior with Y-axis baseline:")
print(f"  - Ball should appear at DIFFERENT Y positions")
print(f"  - Vertical disparity should be significant")

# Try more frames
print("\n" + "="*60)
print("Testing frames 85, 90, 95:")
print("="*60)

for frame_num in [85, 90, 95]:
    left_img = cv2.imread(f"output_frames/left/left3_{frame_num:04d}.jpg")
    right_img = cv2.imread(f"output_frames/right/right3_{frame_num:04d}.jpg")
    
    x_left, y_left = get_ball_xy(left_img)
    x_right, y_right = get_ball_xy(right_img)
    
    if x_left and x_right and y_left and y_right:
        dx = x_left - x_right
        dy = y_left - y_right
        print(f"Frame {frame_num}: left({x_left:4.0f}, {y_left:4.0f}), right({x_right:4.0f}, {y_right:4.0f}), dx={dx:6.1f}, dy={dy:6.1f}")
