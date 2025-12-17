import pandas as pd
import numpy as np

# Load validation results
df = pd.read_csv('output/validation_180frames_results.csv')
valid = df[df['reconstruction_success'] == True]

# Find frame closest to median error
median_error = valid['3d_error_cm'].median()
median_frame_idx = valid.iloc[(valid['3d_error_cm'] - median_error).abs().argsort()[0]]

print("Frame closest to median error:")
print(f"  Frame: {int(median_frame_idx['frame'])}")
print(f"  Error: {median_frame_idx['3d_error_cm']:.1f} cm")
print(f"\n  Predicted:  [{median_frame_idx['pred_x']:10.4f}, {median_frame_idx['pred_y']:10.4f}, {median_frame_idx['pred_z']:10.4f}]")
print(f"  GT (CSV):   [{median_frame_idx['gt_x']:10.4f}, {median_frame_idx['gt_y']:10.4f}, {median_frame_idx['gt_z']:10.4f}]")

# Calculate what the error should be
diff_x = median_frame_idx['pred_x'] - median_frame_idx['gt_x']
diff_y = median_frame_idx['pred_y'] - median_frame_idx['gt_y']
diff_z = median_frame_idx['pred_z'] - median_frame_idx['gt_z']
manual_error = np.sqrt(diff_x**2 + diff_y**2 + diff_z**2) * 100

print(f"\n  Difference: [{diff_x:10.4f}, {diff_y:10.4f}, {diff_z:10.4f}]")
print(f"  Calculated error: {manual_error:.1f} cm")
print(f"  Stored error: {median_frame_idx['3d_error_cm']:.1f} cm")

print("\n" + "="*70)
print("Your manual test coordinates:")
print(f"  System:     [   -1.4077,    -8.4491,     2.21]")
print(f"  Blender GT: [   -1.57082,   -7.9458,    2.3712]")
print(f"  Difference: [    0.163,      -0.503,     -0.161]")
manual_test_error = np.sqrt(0.163**2 + 0.503**2 + 0.161**2) * 100
print(f"  Your error: {manual_test_error:.1f} cm")

print("\n" + "="*70)
print("CRITICAL FINDING:")
print("Your Blender GT has NEGATIVE coordinates (X: -1.57, Y: -7.95)")
print("But the CSV ground truth has POSITIVE coordinates (X~2.8, Y~4.5)")
print("\nThis means the ground truth CSV is from a DIFFERENT Blender scene!")
print("You need to re-run the Blender export script on the SAME scene as finalLeft/finalRight")

