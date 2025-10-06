"""
Test frame 14 specifically to diagnose the issue
"""

import pandas as pd
import numpy as np

# Load Blender ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

print("="*70)
print("FRAME 14 ANALYSIS")
print("="*70)

# Get ground truth for frame 14
if 14 in blender_df['frame'].values:
    gt = blender_df[blender_df['frame'] == 14].iloc[0]
    print(f"\nBlender Ground Truth (Frame 14):")
    print(f"  Position: X={gt['x']:.3f}m, Y={gt['y']:.3f}m, Z={gt['z']:.3f}m")
    
    # Describe position relative to court
    print(f"\nDescription:")
    if gt['x'] > 1.0:
        print(f"  Ball is on the RIGHT side of court (X={gt['x']:.1f}m)")
    elif gt['x'] < -1.0:
        print(f"  Ball is on the LEFT side of court (X={gt['x']:.1f}m)")
    else:
        print(f"  Ball is near CENTER of court (X={gt['x']:.1f}m)")
    
    if gt['y'] > 0:
        print(f"  Ball is towards FRONT of court (Y={gt['y']:.1f}m)")
    elif gt['y'] < -6:
        print(f"  Ball is towards BACK of court (Y={gt['y']:.1f}m)")
    else:
        print(f"  Ball is in MIDDLE of court (Y={gt['y']:.1f}m)")
    
    print(f"  Ball height: {gt['z']:.1f}m")
else:
    print("\nFrame 14 not found in Blender data")

print("\n" + "="*70)
print("Please check this in the GUI:")
print("="*70)
print("1. Does the 2D detection show ball on the right in the image?")
print("2. Does the 3D visualization also show it on the right?")
print("3. If there's a mismatch, the coordinate transform might be flipped")
print("\nNote: In the 3D view, looking from the cameras:")
print("  • +X (positive X) = RIGHT side of court")
print("  • -X (negative X) = LEFT side of court")
