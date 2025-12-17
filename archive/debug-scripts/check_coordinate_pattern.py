import pandas as pd
import numpy as np

df = pd.read_csv('output/comprehensive_validation_results.csv')
valid = df[df['reconstruction_success']].copy()

print("="*70)
print("COORDINATE PATTERN ANALYSIS")
print("="*70)

# Show first 10 frames
print("\nFirst 10 successfully reconstructed frames:\n")
print(f"{'Frame':<8} {'Pred X':<10} {'Pred Y':<10} {'Pred Z':<10} | {'GT X':<10} {'GT Y':<10} {'GT Z':<10}")
print("-" * 80)

for _, row in valid.head(10).iterrows():
    frame = int(row['frame'])
    print(f"{frame:<8} {row['pred_x']:>9.2f} {row['pred_y']:>9.2f} {row['pred_z']:>9.2f} | "
          f"{row['gt_x']:>9.2f} {row['gt_y']:>9.2f} {row['gt_z']:>9.2f}")

# Check if coordinates are changing in same direction
print("\n" + "="*70)
print("TRAJECTORY ANALYSIS")
print("="*70)

pred_x_change = valid['pred_x'].diff().dropna()
pred_y_change = valid['pred_y'].diff().dropna()
pred_z_change = valid['pred_z'].diff().dropna()

gt_x_change = valid['gt_x'].diff().dropna()
gt_y_change = valid['gt_y'].diff().dropna()
gt_z_change = valid['gt_z'].diff().dropna()

print(f"\nAverage frame-to-frame change (Predicted):")
print(f"  X: {pred_x_change.mean():.3f} m/frame")
print(f"  Y: {pred_y_change.mean():.3f} m/frame")
print(f"  Z: {pred_z_change.mean():.3f} m/frame")

print(f"\nAverage frame-to-frame change (Ground Truth):")
print(f"  X: {gt_x_change.mean():.3f} m/frame")
print(f"  Y: {gt_y_change.mean():.3f} m/frame")
print(f"  Z: {gt_z_change.mean():.3f} m/frame")

# Check correlation between predicted and GT trajectories
print("\n" + "="*70)
print("TRAJECTORY CORRELATION")
print("="*70)

from scipy.stats import pearsonr

if len(valid) > 3:
    corr_x, _ = pearsonr(valid['pred_x'], valid['gt_x'])
    corr_y, _ = pearsonr(valid['pred_y'], valid['gt_y'])
    corr_z, _ = pearsonr(valid['pred_z'], valid['gt_z'])
    
    print(f"\nCorrelation between predicted and GT coordinates:")
    print(f"  X-axis: {corr_x:.3f}")
    print(f"  Y-axis: {corr_y:.3f}")
    print(f"  Z-axis: {corr_z:.3f}")
    print(f"\n(1.0 = perfect correlation, -1.0 = inverse, 0.0 = no correlation)")
    
    if abs(corr_x) < 0.3 or abs(corr_y) < 0.3 or abs(corr_z) < 0.3:
        print("\n⚠️  LOW CORRELATION DETECTED!")
        print("This suggests the frames and CSV don't match up correctly.")

# Check if there's a scale factor
print("\n" + "="*70)
print("SCALE ANALYSIS")
print("="*70)

pred_range_x = valid['pred_x'].max() - valid['pred_x'].min()
pred_range_y = valid['pred_y'].max() - valid['pred_y'].min()
pred_range_z = valid['pred_z'].max() - valid['pred_z'].min()

gt_range_x = valid['gt_x'].max() - valid['gt_x'].min()
gt_range_y = valid['gt_y'].max() - valid['gt_y'].min()
gt_range_z = valid['gt_z'].max() - valid['gt_z'].min()

print(f"\nCoordinate ranges (Predicted):")
print(f"  X: {pred_range_x:.2f} m")
print(f"  Y: {pred_range_y:.2f} m")
print(f"  Z: {pred_range_z:.2f} m")

print(f"\nCoordinate ranges (Ground Truth):")
print(f"  X: {gt_range_x:.2f} m")
print(f"  Y: {gt_range_y:.2f} m")
print(f"  Z: {gt_range_z:.2f} m")

if gt_range_x > 0:
    print(f"\nScale ratios (Pred/GT):")
    print(f"  X: {pred_range_x/gt_range_x:.2f}×")
    print(f"  Y: {pred_range_y/gt_range_y:.2f}×")
    print(f"  Z: {pred_range_z/gt_range_z:.2f}×")

