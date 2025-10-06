"""Test detection and triangulation specifically for frame 14"""

import sys
sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline
import cv2

def test_frame_14():
    print("=" * 60)
    print("TESTING FRAME 14 DETECTION & TRIANGULATION")
    print("=" * 60)
    
    # Initialize pipeline
    pipeline = HawkeyePipeline()
    
    # Load videos
    left_video = cv2.VideoCapture('data/left3.mp4')
    right_video = cv2.VideoCapture('data/right3.mp4')
    
    # Navigate to frame 14
    left_video.set(cv2.CAP_PROP_POS_FRAMES, 14)
    right_video.set(cv2.CAP_PROP_POS_FRAMES, 14)
    
    ret_l, left_frame = left_video.read()
    ret_r, right_frame = right_video.read()
    
    if not ret_l or not ret_r:
        print("❌ Failed to read frame 14")
        return
    
    print(f"\n✅ Successfully loaded frame 14")
    print(f"   Left frame shape: {left_frame.shape}")
    print(f"   Right frame shape: {right_frame.shape}")
    
    # Test detection
    print("\n" + "=" * 60)
    print("DETECTION TEST")
    print("=" * 60)
    
    left_xy = pipeline.get_ball_xy(left_frame)
    right_xy = pipeline.get_ball_xy(right_frame)
    
    print(f"\nLeft detection: {left_xy}")
    print(f"Right detection: {right_xy}")
    
    # Test triangulation
    if left_xy != (None, None) and right_xy != (None, None):
        print("\n" + "=" * 60)
        print("TRIANGULATION TEST")
        print("=" * 60)
        
        print(f"\nLeft center: {left_xy}")
        print(f"Right center: {right_xy}")
        print(f"Disparity: {left_xy[0] - right_xy[0]} pixels")
        
        # Process frame through pipeline
        result = pipeline.process_from_pair(left_frame, right_frame, frame_num=14)
        
        if result and result.get('world_coords') is not None and None not in result['world_coords']:
            world_coords = result['world_coords']
            print(f"\nWorld coords: X={world_coords[0]:.3f}m, Y={world_coords[1]:.3f}m, Z={world_coords[2]:.3f}m")
            print(f"\nGround truth: X=2.694m, Y=-2.067m, Z=2.741m")
            print(f"Error: X={abs(world_coords[0]-2.694):.3f}m, Y={abs(world_coords[1]-(-2.067)):.3f}m, Z={abs(world_coords[2]-2.741):.3f}m")
            
            # Check left/right
            if world_coords[0] > 0:
                print("\n✅ Ball is on RIGHT side (X > 0)")
            else:
                print("\n❌ Ball is on LEFT side (X < 0) - SHOULD BE RIGHT!")
        else:
            print("\n❌ Pipeline returned None world coords")
            print(f"Camera coords: {result.get('camera_coords', 'N/A')}")
    else:
        print("\n❌ Cannot triangulate - missing detections")
        if left_xy == (None, None):
            print("   No detection in LEFT frame")
        if right_xy == (None, None):
            print("   No detection in RIGHT frame")
    
    left_video.release()
    right_video.release()

if __name__ == "__main__":
    test_frame_14()
