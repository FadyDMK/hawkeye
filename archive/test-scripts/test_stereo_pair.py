import cv2
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from volleyball_detection import get_ball_xy

print("="*60)
print("STEREO PAIR TEST - Frame 99")
print("="*60)

# Test frame 99 (newest)
left_path = "output_frames/left/left3_0099.jpg"
right_path = "output_frames/right/right3_0099.jpg"

print(f"\nLoading stereo pair...")
left_img = cv2.imread(left_path)
right_img = cv2.imread(right_path)

if left_img is None or right_img is None:
    print("❌ Could not load images!")
    exit()

print(f"✅ Images loaded: {left_img.shape}")

# Detect ball in both views
print(f"\nLeft camera detection:")
x_left, y_left = get_ball_xy(left_img)
print(f"  Position: ({x_left}, {y_left})")

print(f"\nRight camera detection:")
x_right, y_right = get_ball_xy(right_img)
print(f"  Position: ({x_right}, {y_right})")

if x_left is None or x_right is None:
    print("\n❌ Ball not detected in both views!")
    exit()

# Calculate disparity
disparity = x_left - x_right

print(f"\n" + "="*60)
print(f"DISPARITY CALCULATION:")
print(f"="*60)
print(f"Left X:     {x_left}")
print(f"Right X:    {x_right}")
print(f"Disparity:  {disparity} pixels")

if disparity > 0:
    print(f"\n✅ POSITIVE DISPARITY - Cameras are correctly oriented!")
    print(f"   This is what we expect for parallel stereo cameras.")
else:
    print(f"\n❌ NEGATIVE DISPARITY - Cameras are SWAPPED!")
    print(f"   The 'left' folder contains right camera images")
    print(f"   The 'right' folder contains left camera images")
    print(f"\n   SOLUTION: Swap the folders or fix Blender render output")

# Also check vertical alignment
y_diff = abs(y_left - y_right)
print(f"\nVertical alignment:")
print(f"  Y difference: {y_diff} pixels")
if y_diff < 50:
    print(f"  ✅ Good vertical alignment (cameras are level)")
else:
    print(f"  ⚠️  Significant vertical misalignment")
    print(f"     Cameras might not be parallel")

print(f"\n" + "="*60)
print(f"DIAGNOSIS:")
print(f"="*60)
if disparity < 0:
    print("Problem: Your Blender cameras or render output are swapped.")
    print("\nQuick fixes:")
    print("1. In File Explorer, rename folders:")
    print("   - Rename 'left' to 'temp'")
    print("   - Rename 'right' to 'left'") 
    print("   - Rename 'temp' to 'right'")
    print("\n2. OR in Blender: Check which camera is rendering to which path")
    print("\n3. OR modify camera_config.json to account for swap")
else:
    print("✅ Camera orientation is correct!")
    print("   Stereo matching should work properly now.")
