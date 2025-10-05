import cv2
import sys
import os
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from volleyball_detection import get_ball_xy

def analyze_latest_frames():
    """Analyze the latest rendered frames to diagnose detection issues"""
    
    # Get list of all left frames sorted by modification time
    left_dir = "output_frames/left"
    
    if not os.path.exists(left_dir):
        print(f"Directory {left_dir} not found!")
        return
    
    # Get all jpg files
    frames = [f for f in os.listdir(left_dir) if f.endswith('.jpg')]
    
    # Sort by modification time (most recent first)
    frames_with_time = [(f, os.path.getmtime(os.path.join(left_dir, f))) for f in frames]
    frames_with_time.sort(key=lambda x: x[1], reverse=True)
    
    print(f"LATEST FRAME ANALYSIS")
    print(f"="*60)
    print(f"Total frames in directory: {len(frames)}")
    print(f"\nMost recent 5 frames:")
    
    for i, (frame, mtime) in enumerate(frames_with_time[:5]):
        from datetime import datetime
        mod_time = datetime.fromtimestamp(mtime)
        print(f"  {i+1}. {frame} - Modified: {mod_time}")
    
    # Ask user which sequence to test
    print(f"\n🔍 TESTING DETECTION ON LATEST FRAMES...")
    
    # Test the 5 most recent frames
    test_frames = frames_with_time[:5]
    
    for frame_name, _ in test_frames:
        frame_path = os.path.join(left_dir, frame_name)
        print(f"\n{'='*60}")
        print(f"Testing: {frame_name}")
        print(f"{'='*60}")
        
        # Load image
        img = cv2.imread(frame_path)
        if img is None:
            print(f"❌ Could not load image")
            continue
        
        print(f"Image size: {img.shape}")
        
        # Check image properties
        mean_brightness = np.mean(img)
        print(f"Mean brightness: {mean_brightness:.1f}")
        
        # Try detection
        x, y = get_ball_xy(img)
        
        if x is not None and y is not None:
            print(f"✅ Ball detected at ({x}, {y})")
            
            # Visualize
            img_vis = img.copy()
            cv2.circle(img_vis, (int(x), int(y)), 30, (0, 255, 0), 3)
            cv2.putText(img_vis, f"Ball: ({x},{y})", (int(x)+35, int(y)-15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            output_path = frame_path.replace('.jpg', '_DETECTED.jpg')
            cv2.imwrite(output_path, img_vis)
            print(f"Saved visualization: {output_path}")
            
        else:
            print(f"❌ NO BALL DETECTED")
            
            # Save the frame for manual inspection
            print(f"\n🔍 Checking why detection failed...")
            print(f"   Possible reasons:")
            print(f"   1. Ball too blurred from motion")
            print(f"   2. Ball color blends with background")
            print(f"   3. Ball partially out of frame")
            print(f"   4. Ball too small (even at 2x size)")
            print(f"   5. Lighting/contrast issues")
            
            # Try with even lower confidence
            print(f"\n   Trying with VERY LOW confidence (0.1)...")
            x_low, y_low = get_ball_xy(img, conf=0.1)
            
            if x_low is not None:
                print(f"   ✅ Detected with low confidence at ({x_low}, {y_low})")
                print(f"   → Ball exists but confidence is low")
                print(f"   → Need to improve ball visibility/contrast")
            else:
                print(f"   ❌ Still no detection even at 0.1 confidence")
                print(f"   → Ball may not be visible or recognizable")
                
                # Save for manual inspection
                output_path = frame_path.replace('.jpg', '_FAILED.jpg')
                cv2.imwrite(output_path, img)
                print(f"   → Saved for manual inspection: {output_path}")

if __name__ == "__main__":
    analyze_latest_frames()
