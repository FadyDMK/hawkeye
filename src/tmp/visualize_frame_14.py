"""Visualize the left and right detections for frame 14"""

import sys
sys.path.append('src')

from volleyball_detection import get_ball_xy
import cv2
import matplotlib.pyplot as plt

def visualize_frame_14():
    # Load videos
    left_video = cv2.VideoCapture('data/left3.mp4')
    right_video = cv2.VideoCapture('data/right3.mp4')
    
    # Navigate to frame 14
    left_video.set(cv2.CAP_PROP_POS_FRAMES, 14)
    right_video.set(cv2.CAP_PROP_POS_FRAMES, 14)
    
    ret_l, left_frame = left_video.read()
    ret_r, right_frame = right_video.read()
    
    left_x, left_y = get_ball_xy(left_frame)
    right_x, right_y = get_ball_xy(right_frame)
    
    print(f"Left detection: ({left_x}, {left_y})")
    print(f"Right detection: ({right_x}, {right_y})")
    print(f"Disparity: {left_x - right_x} pixels")
    
    # Draw circles on detections
    left_vis = left_frame.copy()
    right_vis = right_frame.copy()
    
    if left_x is not None:
        cv2.circle(left_vis, (int(left_x), int(left_y)), 20, (0, 255, 0), 3)
        cv2.putText(left_vis, f"({left_x}, {left_y})", (int(left_x)+25, int(left_y)), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if right_x is not None:
        cv2.circle(right_vis, (int(right_x), int(right_y)), 20, (0, 255, 0), 3)
        cv2.putText(right_vis, f"({right_x}, {right_y})", (int(right_x)+25, int(right_y)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Create side-by-side visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    ax1.imshow(cv2.cvtColor(left_vis, cv2.COLOR_BGR2RGB))
    ax1.set_title(f'LEFT camera - Ball at ({left_x}, {left_y})')
    ax1.axis('off')
    
    ax2.imshow(cv2.cvtColor(right_vis, cv2.COLOR_BGR2RGB))
    ax2.set_title(f'RIGHT camera - Ball at ({right_x}, {right_y})')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig('frame_14_detections.png', dpi=150, bbox_inches='tight')
    print("\n✅ Saved visualization to frame_14_detections.png")
    
    left_video.release()
    right_video.release()

if __name__ == "__main__":
    visualize_frame_14()
