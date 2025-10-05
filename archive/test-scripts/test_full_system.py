import cv2
import sys
import os
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from volleyball_detection import get_ball_xy
from stereo_matching import StereoMatching
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))
from camera_config import load_camera_config

def test_full_system_on_latest():
    """Test complete stereo system on latest rendered frames"""
    
    config = load_camera_config()
    
    # Test a few frames from the end of the sequence
    test_frame_numbers = [245, 246, 247, 248, 249]
    
    print("FULL SYSTEM TEST - LATEST FRAMES")
    print("="*60)
    print(f"Config: baseline={config['baseline_m']}m, z_range=[{config['z_min_m']}, {config['z_max_m']}]m")
    print()
    
    success_count = 0
    
    for frame_num in test_frame_numbers:
        left_path = f"output_frames/left/left3_{frame_num:04d}.jpg"
        right_path = f"output_frames/right/right3_{frame_num:04d}.jpg"
        
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print(f"❌ Frame {frame_num}: Files not found")
            continue
            
        print(f"\n{'='*60}")
        print(f"Frame {frame_num}")
        print(f"{'='*60}")
        
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        # Test detection on both
        x_left, y_left = get_ball_xy(left_img)
        x_right, y_right = get_ball_xy(right_img)
        
        print(f"Left detection:  ", end="")
        if x_left is not None:
            print(f"✅ ({x_left}, {y_left})")
        else:
            print(f"❌ FAILED")
            
        print(f"Right detection: ", end="")
        if x_right is not None:
            print(f"✅ ({x_right}, {y_right})")
        else:
            print(f"❌ FAILED")
        
        if x_left is None or x_right is None:
            print(f"Result: ❌ DETECTION FAILED")
            continue
            
        # Calculate disparity
        disparity = x_left - x_right
        print(f"Disparity: {disparity} pixels")
        
        if disparity <= 0:
            print(f"Result: ❌ NEGATIVE/ZERO DISPARITY")
            continue
        
        # Try full stereo matching
        try:
            stereo = StereoMatching(left_img, right_img, config)
            success = stereo.try_detection_triangulation()
            
            if success:
                print(f"✅ STEREO SUCCESS!")
                print(f"   3D Position: X={stereo.X_ball:.2f}m, Y={stereo.Y_ball:.2f}m, Z={stereo.Z_ball:.2f}m")
                success_count += 1
            else:
                print(f"❌ Triangulation failed")
                
                # Calculate manually to see why
                focal_length = config['focal_length_px']
                baseline = config['baseline_m']
                Z = (focal_length * baseline) / disparity
                print(f"   Manual calculation: Z={Z:.2f}m")
                print(f"   Valid range: [{config['z_min_m']}, {config['z_max_m']}]m")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {success_count}/{len(test_frame_numbers)} frames processed successfully")
    print(f"Success rate: {success_count/len(test_frame_numbers)*100:.1f}%")
    
    if success_count == 0:
        print(f"\n⚠️  SYSTEM NOT WORKING - DIAGNOSIS:")
        print(f"   - Detection works ✅")
        print(f"   - But stereo triangulation fails ❌")
        print(f"   - Check: disparity sign, baseline value, depth range")

if __name__ == "__main__":
    test_full_system_on_latest()
