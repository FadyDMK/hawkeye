"""
Compare Hawkeye-calculated ball positions with actual Blender positions.
Run this AFTER you've run the Blender script to extract ground truth positions.
"""

import pandas as pd
import numpy as np
import os

def load_hawkeye_positions():
    """Load positions calculated by Hawkeye"""
    hawkeye_file = "output/ball_positions_world.csv"
    
    if not os.path.exists(hawkeye_file):
        print(f"ERROR: {hawkeye_file} not found!")
        print("Run the Hawkeye pipeline first to generate world coordinates.")
        return None
    
    df = pd.read_csv(hawkeye_file)
    print(f"Loaded {len(df)} frames from Hawkeye")
    return df

def load_blender_positions():
    """Load ground truth positions from Blender"""
    # Check multiple possible locations
    possible_paths = [
        "ball_positions_blender.csv",
        "data/ball_positions_blender.csv",
        "../ball_positions_blender.csv"
    ]
    
    blender_file = None
    for path in possible_paths:
        if os.path.exists(path):
            blender_file = path
            break
    
    if blender_file is None:
        print("ERROR: ball_positions_blender.csv not found!")
        print("Make sure you've run the Blender script and copied the CSV file to this directory.")
        print(f"Checked locations: {possible_paths}")
        return None
    
    df = pd.read_csv(blender_file)
    print(f"Loaded {len(df)} frames from Blender")
    return df

def compare_positions(hawkeye_df, blender_df):
    """Compare Hawkeye vs Blender positions"""
    
    print("\n" + "="*70)
    print("POSITION COMPARISON: Hawkeye vs Blender Ground Truth")
    print("="*70)
    
    # Normalize column names to lowercase
    hawkeye_df.columns = hawkeye_df.columns.str.lower()
    blender_df.columns = blender_df.columns.str.lower()
    
    # Remove rows with NaN values
    hawkeye_df = hawkeye_df.dropna()
    
    # Merge on frame number
    merged = pd.merge(hawkeye_df, blender_df, on='frame', suffixes=('_hawkeye', '_blender'))
    
    if len(merged) == 0:
        print("ERROR: No matching frames between Hawkeye and Blender data!")
        return
    
    print(f"\nComparing {len(merged)} matching frames\n")
    
    # Calculate errors
    merged['error_x'] = merged['x_hawkeye'] - merged['x_blender']
    merged['error_y'] = merged['y_hawkeye'] - merged['y_blender']
    merged['error_z'] = merged['z_hawkeye'] - merged['z_blender']
    merged['error_3d'] = np.sqrt(merged['error_x']**2 + merged['error_y']**2 + merged['error_z']**2)
    
    # Print sample frames
    test_frames = [90, 93, 96, 99]
    print("Sample frame comparison:")
    print("-" * 70)
    
    for frame_num in test_frames:
        if frame_num in merged['frame'].values:
            row = merged[merged['frame'] == frame_num].iloc[0]
            print(f"\nFrame {frame_num}:")
            print(f"  Blender:  X={row['x_blender']:7.3f}, Y={row['y_blender']:7.3f}, Z={row['z_blender']:7.3f}")
            print(f"  Hawkeye:  X={row['x_hawkeye']:7.3f}, Y={row['y_hawkeye']:7.3f}, Z={row['z_hawkeye']:7.3f}")
            print(f"  Error:    X={row['error_x']:7.3f}, Y={row['error_y']:7.3f}, Z={row['error_z']:7.3f}  (3D: {row['error_3d']:.3f}m)")
    
    # Print overall statistics
    print("\n" + "="*70)
    print("OVERALL STATISTICS")
    print("="*70)
    
    print(f"\nMean errors:")
    print(f"  X: {merged['error_x'].mean():7.3f} ± {merged['error_x'].std():.3f} m")
    print(f"  Y: {merged['error_y'].mean():7.3f} ± {merged['error_y'].std():.3f} m")
    print(f"  Z: {merged['error_z'].mean():7.3f} ± {merged['error_z'].std():.3f} m")
    print(f"  3D: {merged['error_3d'].mean():7.3f} ± {merged['error_3d'].std():.3f} m")
    
    print(f"\nMax errors:")
    print(f"  X: {abs(merged['error_x']).max():.3f} m")
    print(f"  Y: {abs(merged['error_y']).max():.3f} m")
    print(f"  Z: {abs(merged['error_z']).max():.3f} m")
    print(f"  3D: {merged['error_3d'].max():.3f} m")
    
    print(f"\nRMS errors:")
    print(f"  X: {np.sqrt((merged['error_x']**2).mean()):.3f} m")
    print(f"  Y: {np.sqrt((merged['error_y']**2).mean()):.3f} m")
    print(f"  Z: {np.sqrt((merged['error_z']**2).mean()):.3f} m")
    print(f"  3D: {np.sqrt((merged['error_3d']**2).mean()):.3f} m")
    
    # Check for systematic bias
    print("\n" + "="*70)
    print("SYSTEMATIC BIAS CHECK")
    print("="*70)
    
    mean_x = merged['error_x'].mean()
    mean_y = merged['error_y'].mean()
    mean_z = merged['error_z'].mean()
    
    if abs(mean_x) > 0.5:
        print(f"⚠️  Large X bias: {mean_x:.3f}m (Hawkeye may be shifted horizontally)")
    else:
        print(f"✓ X bias acceptable: {mean_x:.3f}m")
    
    if abs(mean_y) > 0.5:
        print(f"⚠️  Large Y bias: {mean_y:.3f}m (Hawkeye may be shifted along court length)")
    else:
        print(f"✓ Y bias acceptable: {mean_y:.3f}m")
    
    if abs(mean_z) > 0.5:
        print(f"⚠️  Large Z bias: {mean_z:.3f}m (Hawkeye may be shifted vertically)")
    else:
        print(f"✓ Z bias acceptable: {mean_z:.3f}m")
    
    # Save comparison
    output_file = "ball_positions_comparison.csv"
    merged.to_csv(output_file, index=False)
    print(f"\n✓ Saved detailed comparison to: {output_file}")

if __name__ == "__main__":
    hawkeye_df = load_hawkeye_positions()
    blender_df = load_blender_positions()
    
    if hawkeye_df is not None and blender_df is not None:
        compare_positions(hawkeye_df, blender_df)
    else:
        print("\nCannot proceed with comparison due to missing data files.")
