import cv2
import numpy as np

def analyze_camera_views():
    """Analyze the actual camera views to understand the setup"""
    
    # Load same frame from both cameras
    left_img = cv2.imread("output_frames/left/left_0050.jpg")
    right_img = cv2.imread("output_frames/right/right_0050.jpg")
    
    if left_img is None or right_img is None:
        print("Could not load images")
        return
    
    print("CAMERA VIEW ANALYSIS")
    print("="*50)
    
    # Check if images are identical (would indicate wrong setup)
    diff = cv2.absdiff(left_img, right_img)
    total_diff = np.sum(diff)
    
    print(f"Total pixel difference between views: {total_diff}")
    
    if total_diff < 1000000:  # Very similar images
        print("⚠️  WARNING: Images are very similar - possible camera setup issue")
    else:
        print("✅ Images are different - cameras have different viewpoints")
    
    # Analyze ball positions we found earlier
    print(f"\nBALL POSITION ANALYSIS:")
    print(f"Left camera ball:  (704, 557)")
    print(f"Right camera ball: (739, 558)")
    print(f"Disparity: 704 - 739 = -35 pixels")
    
    print(f"\nSTEREO GEOMETRY ANALYSIS:")
    print(f"Negative disparity means:")
    print(f"- Right camera sees ball FURTHER LEFT than left camera")
    print(f"- This happens when cameras are positioned backwards")
    
    # Check which direction the ball appears to move
    ball_x_left = 704
    ball_x_right = 739
    
    if ball_x_right > ball_x_left:
        print(f"\n🔍 DIAGNOSIS:")
        print(f"Right camera sees ball at x={ball_x_right}")
        print(f"Left camera sees ball at x={ball_x_left}")
        print(f"→ Right camera sees ball FURTHER RIGHT")
        print(f"→ This suggests cameras are positioned CORRECTLY")
        print(f"→ But ball might be BEHIND the camera baseline")
        print(f"→ OR ball is VERY FAR AWAY")
    
    # Calculate what the ball distance would be with correct disparity
    disparity_abs = abs(ball_x_left - ball_x_right)
    focal_length = 1386.67
    baseline = 3.0
    
    distance = (focal_length * baseline) / disparity_abs
    print(f"\nDISTANCE CALCULATION:")
    print(f"Using absolute disparity {disparity_abs} pixels:")
    print(f"Distance = {distance:.1f} meters")
    print(f"This is beyond your 65m maximum range!")
    
    print(f"\nPOSSIBLE SOLUTIONS:")
    print(f"1. Move cameras CLOSER to the volleyball court in Blender")
    print(f"2. Reduce baseline distance (cameras closer together)")
    print(f"3. Adjust camera config to match actual Blender setup")
    
    # Save a comparison image
    comparison = np.hstack([left_img, right_img])
    cv2.putText(comparison, "LEFT", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
    cv2.putText(comparison, "RIGHT", (left_img.shape[1] + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
    cv2.imwrite("camera_comparison.jpg", comparison)
    print(f"\nSaved camera comparison: camera_comparison.jpg")

if __name__ == "__main__":
    analyze_camera_views()