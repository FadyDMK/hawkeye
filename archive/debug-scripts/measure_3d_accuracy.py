"""
Measure 3D reconstruction accuracy by comparing predicted ball positions
against Blender ground truth.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def load_ground_truth():
    """Load Blender ground truth ball positions."""
    gt_path = Path('3D-models/Latest volley go brr/ball_positions_blender.csv')
    
    if not gt_path.exists():
        # Try alternative paths
        alt_paths = [
            Path('src/tmp/ball_positions_blender.csv'),
            Path('src/ball_positions_world.csv')
        ]
        for alt in alt_paths:
            if alt.exists():
                gt_path = alt
                break
    
    if not gt_path.exists():
        print(f"Error: Ground truth not found at {gt_path}")
        return None
    
    df = pd.read_csv(gt_path)
    print(f"Loaded ground truth: {len(df)} frames")
    print(f"Columns: {df.columns.tolist()}")
    return df

def load_predicted_positions():
    """Load predicted ball positions from pipeline."""
    pred_path = Path('output/ball_positions_world.csv')
    
    if not pred_path.exists():
        pred_path = Path('src/ball_positions_world.csv')
    
    if not pred_path.exists():
        print(f"Error: Predicted positions not found at {pred_path}")
        return None
    
    df = pd.read_csv(pred_path)
    print(f"Loaded predictions: {len(df)} frames")
    print(f"Columns: {df.columns.tolist()}")
    return df

def calculate_errors(gt_df, pred_df):
    """Calculate 3D reconstruction errors."""
    
    # Match frames by frame number/index
    # Assuming both have frame identifiers
    
    # Extract coordinates (adjust column names as needed)
    gt_cols = gt_df.columns.tolist()
    pred_cols = pred_df.columns.tolist()
    
    print(f"\nGround truth columns: {gt_cols}")
    print(f"Predicted columns: {pred_cols}")
    
    # Common column name variations
    x_names = ['X', 'x', 'X_ball', 'x_ball', 'ball_x']
    y_names = ['Y', 'y', 'Y_ball', 'y_ball', 'ball_y']
    z_names = ['Z', 'z', 'Z_ball', 'z_ball', 'ball_z']
    
    # Find matching column names
    gt_x = next((col for col in gt_cols if col in x_names), None)
    gt_y = next((col for col in gt_cols if col in y_names), None)
    gt_z = next((col for col in gt_cols if col in z_names), None)
    
    pred_x = next((col for col in pred_cols if col in x_names), None)
    pred_y = next((col for col in pred_cols if col in y_names), None)
    pred_z = next((col for col in pred_cols if col in z_names), None)
    
    if not all([gt_x, gt_y, gt_z, pred_x, pred_y, pred_z]):
        print("Error: Could not find coordinate columns")
        print(f"GT: {gt_x}, {gt_y}, {gt_z}")
        print(f"Pred: {pred_x}, {pred_y}, {pred_z}")
        return None
    
    print(f"\nUsing columns:")
    print(f"GT: {gt_x}, {gt_y}, {gt_z}")
    print(f"Pred: {pred_x}, {pred_y}, {pred_z}")
    
    # Merge dataframes on frame index
    # Use the minimum length to avoid mismatches
    min_len = min(len(gt_df), len(pred_df))
    
    errors = []
    valid_comparisons = 0
    
    for i in range(min_len):
        try:
            gt_pos = np.array([gt_df[gt_x].iloc[i], gt_df[gt_y].iloc[i], gt_df[gt_z].iloc[i]])
            pred_pos = np.array([pred_df[pred_x].iloc[i], pred_df[pred_y].iloc[i], pred_df[pred_z].iloc[i]])
            
            # Skip if any value is NaN or invalid
            if np.any(np.isnan(gt_pos)) or np.any(np.isnan(pred_pos)):
                continue
            
            # Calculate Euclidean distance
            error = np.linalg.norm(pred_pos - gt_pos)
            
            # Convert to cm if values are in meters
            if error < 1.0:  # Likely in meters
                error *= 100
            
            errors.append({
                'frame': i,
                'error_cm': error,
                'gt_x': gt_pos[0],
                'gt_y': gt_pos[1],
                'gt_z': gt_pos[2],
                'pred_x': pred_pos[0],
                'pred_y': pred_pos[1],
                'pred_z': pred_pos[2],
                'distance_m': np.sqrt(gt_pos[0]**2 + gt_pos[2]**2)  # distance from camera
            })
            valid_comparisons += 1
            
        except Exception as e:
            continue
    
    if not errors:
        print("Error: No valid error comparisons could be made")
        return None
    
    errors_df = pd.DataFrame(errors)
    print(f"\nValid comparisons: {valid_comparisons}")
    
    return errors_df

def analyze_errors(errors_df):
    """Analyze and report error statistics."""
    
    print("\n" + "="*70)
    print("3D RECONSTRUCTION ACCURACY ANALYSIS")
    print("="*70)
    
    errors = errors_df['error_cm'].values
    
    # Overall statistics
    print("\n### OVERALL ERROR STATISTICS ###\n")
    print(f"Mean error:       {np.mean(errors):.2f} cm")
    print(f"Median error:     {np.median(errors):.2f} cm")
    print(f"Std deviation:    {np.std(errors):.2f} cm")
    print(f"Min error:        {np.min(errors):.2f} cm")
    print(f"Max error:        {np.max(errors):.2f} cm")
    print(f"95th percentile:  {np.percentile(errors, 95):.2f} cm")
    
    # Percentile breakdown
    print("\n### ERROR DISTRIBUTION BY PERCENTILE ###\n")
    for p in [25, 50, 75, 90, 95, 99]:
        print(f"{p}th percentile: {np.percentile(errors, p):.2f} cm")
    
    # Error by distance ranges
    print("\n### ERROR BY DISTANCE FROM CAMERAS ###\n")
    distance_ranges = [
        (12, 15, "near court"),
        (15, 20, "mid court"),
        (20, 25, "far court"),
        (25, 30, "beyond baseline")
    ]
    
    for min_d, max_d, label in distance_ranges:
        mask = (errors_df['distance_m'] >= min_d) & (errors_df['distance_m'] < max_d)
        range_errors = errors_df[mask]['error_cm']
        if len(range_errors) > 0:
            print(f"{min_d}-{max_d}m ({label:.<20}): {range_errors.mean():.2f} cm (n={len(range_errors)})")
    
    # Axis-wise error analysis
    print("\n### SYSTEMATIC ERROR BY AXIS ###\n")
    errors_df['error_x'] = (errors_df['pred_x'] - errors_df['gt_x']) * 100  # to cm
    errors_df['error_y'] = (errors_df['pred_y'] - errors_df['gt_y']) * 100
    errors_df['error_z'] = (errors_df['pred_z'] - errors_df['gt_z']) * 100
    
    for axis in ['x', 'y', 'z']:
        col = f'error_{axis}'
        bias = errors_df[col].mean()
        std = errors_df[col].std()
        print(f"{axis.upper()}-axis: Bias = {bias:+.2f} cm, σ = {std:.2f} cm")
    
    # Comparison to ball diameter
    ball_diameter_cm = 21.0
    print("\n### ERROR RELATIVE TO BALL SIZE (21 cm diameter) ###\n")
    print(f"Mean error:       {np.mean(errors):.2f} cm = {np.mean(errors)/ball_diameter_cm:.2f}× ball diameter")
    print(f"Median error:     {np.median(errors):.2f} cm = {np.median(errors)/ball_diameter_cm:.2f}× ball diameter")
    print(f"95th percentile:  {np.percentile(errors, 95):.2f} cm = {np.percentile(errors, 95)/ball_diameter_cm:.2f}× ball diameter")
    
    # Temporal consistency
    print("\n### TEMPORAL CONSISTENCY ###\n")
    frame_diffs = np.diff(errors_df['error_cm'].values)
    outlier_threshold = 15  # cm - large jumps indicate detection failures
    outliers = np.abs(frame_diffs) > outlier_threshold
    print(f"Mean inter-frame error change: {np.mean(np.abs(frame_diffs)):.2f} cm")
    print(f"Outlier rate (>{outlier_threshold}cm jumps): {np.sum(outliers)/len(outliers)*100:.1f}%")
    
    # Save results
    errors_df.to_csv('output/3d_reconstruction_errors.csv', index=False)
    print(f"\n✓ Detailed results saved to output/3d_reconstruction_errors.csv")
    
    # Create histogram
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('3D Reconstruction Error (cm)')
    plt.ylabel('Frequency (number of frames)')
    plt.title('Distribution of 3D Reconstruction Errors')
    plt.axvline(np.mean(errors), color='r', linestyle='--', label=f'Mean: {np.mean(errors):.1f} cm')
    plt.axvline(np.median(errors), color='g', linestyle='--', label=f'Median: {np.median(errors):.1f} cm')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('output/error_distribution.png', dpi=150, bbox_inches='tight')
    print(f"✓ Histogram saved to output/error_distribution.png")
    
    return errors_df

def main():
    print("Loading data...")
    gt_df = load_ground_truth()
    pred_df = load_predicted_positions()
    
    if gt_df is None or pred_df is None:
        print("\nCannot proceed without both ground truth and predicted positions.")
        print("Please ensure you have:")
        print("  1. Ground truth: 3D-models/Latest volley go brr/ball_positions_blender.csv")
        print("  2. Predictions: output/ball_positions_world.csv")
        print("\nRun the pipeline first to generate predictions.")
        return
    
    print("\nCalculating errors...")
    errors_df = calculate_errors(gt_df, pred_df)
    
    if errors_df is None:
        print("Failed to calculate errors")
        return
    
    analyze_errors(errors_df)
    
    print("\n" + "="*70)
    print("CHAPTER 5 DATA SUMMARY")
    print("="*70)
    
    errors = errors_df['error_cm'].values
    
    print("\n**For Section 5.4.1 (Overall Reconstruction Error):**")
    print(f"Mean error: {np.mean(errors):.1f} cm")
    print(f"Median error: {np.median(errors):.1f} cm")
    print(f"Std deviation: {np.std(errors):.1f} cm")
    print(f"95th percentile: {np.percentile(errors, 95):.1f} cm")
    
    print("\n**For Section 5.4.3 (Error vs Distance):**")
    for min_d, max_d, label in [(12, 15), (15, 20), (20, 25), (25, 30)]:
        mask = (errors_df['distance_m'] >= min_d) & (errors_df['distance_m'] < max_d)
        range_errors = errors_df[mask]['error_cm']
        if len(range_errors) > 0:
            print(f"{min_d}-{max_d}m: {range_errors.mean():.1f} cm (n={len(range_errors)})")

if __name__ == '__main__':
    main()
