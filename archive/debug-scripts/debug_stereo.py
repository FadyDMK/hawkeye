import cv2
import sys
import os
import numpy as np

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from volleyball_detection import get_ball_xy
from stereo_matching import StereoMatching
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from camera_config import load_camera_config

def test_stereo_matching():
    """Test stereo matching on frame pairs from new footage"""
    
    config = load_camera_config()
    
    # Test frame pairs
    test_pairs = [
        ("output_frames/left/left_0050.jpg", "output_frames/right/right_0050.jpg"),
        ("output_frames/left/left_0100.jpg", "output_frames/right/right_0100.jpg"),
        ("output_frames/left/left3_0050.jpg", "output_frames/right/right3_0050.jpg"),
        ("output_frames/left/left3_0100.jpg", "output_frames/right/right3_0100.jpg"),
    ]
    
    for left_path, right_path in test_pairs:
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print(f"❌ Missing frames: {left_path} or {right_path}")
            continue
            
        print(f"\n{'='*60}")
        print(f"Testing stereo pair: {os.path.basename(left_path)}")
        print(f"{'='*60}")
        
        # Load images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        if left_img is None or right_img is None:
            print("❌ Could not load images")
            continue
            
        # Test individual detections
        print("Left camera detection:")
        x_left, y_left = get_ball_xy(left_img)
        if x_left is not None:
            print(f"  ✅ Ball detected at ({x_left}, {y_left})")
        else:
            print(f"  ❌ No ball detected")
            
        print("Right camera detection:")
        x_right, y_right = get_ball_xy(right_img)
        if x_right is not None:
            print(f"  ✅ Ball detected at ({x_right}, {y_right})")
        else:
            print(f"  ❌ No ball detected")
            
        # Calculate disparity if both detected
        if x_left is not None and x_right is not None:
            disparity = x_left - x_right
            print(f"Raw disparity: {disparity}")
            
            # Check if disparity is reasonable
            if disparity > 0.5:
                # Calculate expected depth
                focal_length = config['focal_length_px']
                baseline = config['baseline_m']
                Z = (focal_length * baseline) / disparity
                print(f"Calculated depth: {Z:.2f}m")
                
                # Check if depth is in valid range
                z_min = config.get('z_min_m', 15.0)
                z_max = config.get('z_max_m', 40.0)
                if z_min <= Z <= z_max:
                    print(f"✅ Depth in valid range [{z_min}, {z_max}]")
                else:
                    print(f"❌ Depth {Z:.2f}m outside valid range [{z_min}, {z_max}]")
            else:
                print(f"❌ Disparity too small: {disparity}")
        
        # Test full stereo matching
        print("\nTesting stereo matching:")
        try:
            stereo = StereoMatching(left_img, right_img, config)
            
            # Try detection-based triangulation
            success = stereo.try_detection_triangulation()
            if success:
                print(f"✅ Detection triangulation succeeded")
                print(f"  3D Position: ({stereo.X_ball:.2f}, {stereo.Y_ball:.2f}, {stereo.Z_ball:.2f})")
            else:
                print(f"❌ Detection triangulation failed")
                
                # Try full stereo matching workflow
                print("Trying full stereo matching workflow...")
                # This would normally be called with a disparity map
                # For now, let's just see what the detection results are
                
        except Exception as e:
            print(f"❌ Stereo matching error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_stereo_matching()