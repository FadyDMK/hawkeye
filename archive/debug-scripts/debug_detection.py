import cv2
import sys
import os
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from volleyball_detection import get_ball_xy

def test_detection_on_frames():
    """Test YOLO detection on the new footage frames"""
    
    # Test frames from both sequences
    test_frames = [
        "output_frames/left/left_0050.jpg",    # Middle of first sequence
        "output_frames/left/left_0100.jpg",    # Later in first sequence
        "output_frames/left/left3_0050.jpg",   # Middle of second sequence
        "output_frames/left/left3_0100.jpg",   # Later in second sequence
    ]
    
    results = []
    
    for frame_path in test_frames:
        if os.path.exists(frame_path):
            print(f"\nTesting frame: {frame_path}")
            
            # Load image
            img = cv2.imread(frame_path)
            if img is None:
                print(f"Could not load {frame_path}")
                continue
                
            print(f"Image shape: {img.shape}")
            
            # Test detection
            x, y = get_ball_xy(img)
            
            if x is not None and y is not None:
                print(f"✅ Ball detected at: ({x}, {y})")
                results.append((frame_path, x, y, True))
                
                # Visualize detection
                img_vis = img.copy()
                cv2.circle(img_vis, (int(x), int(y)), 20, (0, 255, 0), 3)
                cv2.putText(img_vis, f"Ball: ({x},{y})", (int(x)+25, int(y)-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Save visualization
                output_path = frame_path.replace('.jpg', '_detected.jpg')
                cv2.imwrite(output_path, img_vis)
                print(f"Saved detection visualization: {output_path}")
                
            else:
                print("❌ No ball detected")
                results.append((frame_path, None, None, False))
        else:
            print(f"Frame not found: {frame_path}")
    
    # Summary
    print(f"\n{'='*50}")
    print("DETECTION SUMMARY:")
    print(f"{'='*50}")
    
    detected_count = sum(1 for r in results if r[3])
    total_count = len(results)
    
    print(f"Detected: {detected_count}/{total_count} frames")
    print(f"Success rate: {detected_count/total_count*100:.1f}%")
    
    for result in results:
        frame, x, y, detected = result
        status = "✅ DETECTED" if detected else "❌ FAILED"
        pos_str = f"at ({x}, {y})" if detected else ""
        print(f"{os.path.basename(frame):20} {status} {pos_str}")
    
    return results

if __name__ == "__main__":
    test_detection_on_frames()