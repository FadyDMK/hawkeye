"""
Test all 48 coordinate mappings to find the correct one for MKV videos.
The MP4 videos used [+Y, -X, -Z] but MKV videos might need a different mapping.
"""

import sys
import cv2
import numpy as np
import pandas as pd
import os
from itertools import permutations, product

sys.path.append('src')

# Load ground truth
gt_path = "3D-models/Latest volley go brr/ball_positions_blender.csv"
blender_df = pd.read_csv(gt_path)

# Test frame
test_frame = 90

# Load frame
left_path = f"output_frames/left/left3_{test_frame:04d}.jpg"
right_path = f"output_frames/right/right3_{test_frame:04d}.jpg"

left_img = cv2.imread(left_path)
right_img = cv2.imread(right_path)

# Get camera coordinates
from hawkeye_pipeline import HawkeyePipeline
pipeline = HawkeyePipeline()
pipeline.clear_previous_results()

from stereo_matching import StereoMatching
stereo_matcher = StereoMatching(left_img, right_img, config=pipeline.config)

if stereo_matcher.try_detection_triangulation():
    cam_coords = np.array([stereo_matcher.X_ball, stereo_matcher.Y_ball, stereo_matcher.Z_ball])
    
    # Ground truth
    row = blender_df[blender_df['frame'] == test_frame].iloc[0]
    gt = np.array([row['x'], row['y'], row['z']])
    
    print(f"Frame {test_frame}:")
    print(f"  Camera coords: {cam_coords}")
    print(f"  Ground truth:  {gt}")
    print(f"\nTesting all 48 coordinate mappings...")
    print("="*80)
    
    # Translation (keep same for now)
    t = np.array([-0.170, 18.827, 0.249])
    R = np.eye(3)
    
    # Test all permutations and sign combinations
    results = []
    
    axes = [0, 1, 2]  # X, Y, Z
    for perm in permutations(axes):
        for signs in product([-1, 1], repeat=3):
            # Create mapping
            mapped = np.array([
                signs[0] * cam_coords[perm[0]],
                signs[1] * cam_coords[perm[1]],
                signs[2] * cam_coords[perm[2]]
            ])
            
            # Apply transformation (no scaling for now)
            world = (R @ mapped) + t
            
            # Calculate error
            error = np.linalg.norm(world - gt)
            
            # Store result
            axis_names = ['X', 'Y', 'Z']
            mapping_str = f"[{'+' if signs[0] > 0 else '-'}{axis_names[perm[0]]}, " \
                         f"{'+' if signs[1] > 0 else '-'}{axis_names[perm[1]]}, " \
                         f"{'+' if signs[2] > 0 else '-'}{axis_names[perm[2]]}]"
            
            results.append({
                'mapping': mapping_str,
                'world': world,
                'error': error
            })
    
    # Sort by error
    results.sort(key=lambda x: x['error'])
    
    print(f"\nTop 10 best mappings:")
    print("="*80)
    for i, r in enumerate(results[:10]):
        print(f"{i+1}. {r['mapping']:20s} → Error: {r['error']:.3f}m")
        print(f"   World: ({r['world'][0]:7.3f}, {r['world'][1]:7.3f}, {r['world'][2]:7.3f})")
        print(f"   GT:    ({gt[0]:7.3f}, {gt[1]:7.3f}, {gt[2]:7.3f})")
        print()
    
    print("="*80)
    print(f"Best mapping: {results[0]['mapping']}")
    print(f"This mapping gives {results[0]['error']:.3f}m error BEFORE scaling optimization")
    print("Use this mapping instead of [+Y, -X, -Z] in transforms.py")
    print("="*80)
    
else:
    print(f"Failed to detect ball in frame {test_frame}")
