import pandas as pd
import cv2

# Load validation results
df = pd.read_csv('output/validation_180frames_results.csv')

# Find frame with largest error
worst = df.loc[df['3d_error_cm'].idxmax()]

print(f"Worst frame: {int(worst['frame'])}")
print(f"Error: {worst['3d_error_cm']:.1f} cm")
print(f"\nPredicted:     ({worst['pred_x']:.3f}, {worst['pred_y']:.3f}, {worst['pred_z']:.3f})")
print(f"Ground truth:  ({worst['gt_x']:.3f}, {worst['gt_y']:.3f}, {worst['gt_z']:.3f})")
print(f"\nDifference:    ({worst['pred_x']-worst['gt_x']:.3f}, {worst['pred_y']-worst['gt_y']:.3f}, {worst['pred_z']-worst['gt_z']:.3f})")

# Extract and save the frame from both videos
frame_num = int(worst['frame'])

left_cap = cv2.VideoCapture('test-vids/finalLeft.mkv')
right_cap = cv2.VideoCapture('test-vids/finalRight.mkv')

# Seek to frame (frame_num - 1 because we read frame 1 as first)
left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)
right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num - 1)

ret_left, left_frame = left_cap.read()
ret_right, right_frame = right_cap.read()

if ret_left and ret_right:
    cv2.imwrite(f'output/worst_frame_{frame_num}_left.jpg', left_frame)
    cv2.imwrite(f'output/worst_frame_{frame_num}_right.jpg', right_frame)
    print(f"\n✓ Saved frames to:")
    print(f"  - output/worst_frame_{frame_num}_left.jpg")
    print(f"  - output/worst_frame_{frame_num}_right.jpg")
else:
    print(f"\n✗ Failed to extract frame {frame_num}")

left_cap.release()
right_cap.release()

