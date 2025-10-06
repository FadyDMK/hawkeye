"""
Test that exactly matches what the GUI does - load from output_frames/ folder.
"""

import sys
import cv2
import glob
import numpy as np
import pandas as pd
sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline

# Initialize pipeline
pipeline = HawkeyePipeline()

print("="*80)
print("GUI-EXACT TEST - Frames from output_frames/ folder")
print("="*80)
print(f"Configuration:")
print(f"  Baseline: {pipeline.config['baseline_m']}m")
print(f"  Scale: {pipeline.scale}")
print(f"  Translation: {pipeline.t}")
print("="*80)

# Load ground truth
blender_df = pd.read_csv('3D-models/Latest volley go brr/ball_positions_blender.csv')

# Get frame files (same as GUI)
left_frames = sorted(glob.glob("output_frames/left/left3_*.jpg"))
right_frames = sorted(glob.glob("output_frames/right/right3_*.jpg"))

print(f"\nTotal frames available: {len(left_frames)}")
print(f"Testing frames 85-100...")
print("="*80)

results = []

for frame_idx in range(85, min(101, len(left_frames))):
    # Load images (same as GUI)
    left_img = cv2.imread(left_frames[frame_idx])
    right_img = cv2.imread(right_frames[frame_idx])
    
    if left_img is None or right_img is None:
        print(f"\nFrame {frame_idx}: Failed to load images")
        continue
    
    # Process
    pipeline.clear_previous_results()
    result = pipeline.process_from_pair(left_img, right_img, frame_num=frame_idx, display=False)
    
    # Get ground truth for this frame number
    if frame_idx in blender_df['frame'].values:
        row = blender_df[blender_df['frame'] == frame_idx].iloc[0]
        gt = np.array([row['x'], row['y'], row['z']])
        
        if result and len(pipeline.ball_positions_world) > 0:
            world_pos = np.array(pipeline.ball_positions_world[-1])
            
            if None not in world_pos and not any(np.isnan(world_pos)):
                error_vec = world_pos - gt
                error = np.linalg.norm(error_vec)
                
                results.append({
                    'frame': frame_idx,
                    'predicted': world_pos,
                    'ground_truth': gt,
                    'error': error
                })
                
                status = "✅ GOOD" if error < 3.0 else "⚠️  HIGH ERROR"
                print(f"\nFrame {frame_idx}: {status}")
                print(f"  Predicted: ({world_pos[0]:7.3f}, {world_pos[1]:7.3f}, {world_pos[2]:7.3f})")
                print(f"  GT:        ({gt[0]:7.3f}, {gt[1]:7.3f}, {gt[2]:7.3f})")
                print(f"  Error:     {error:.3f}m")
            else:
                print(f"\nFrame {frame_idx}: ❌ FAILED - Got None/NaN values")
        else:
            print(f"\nFrame {frame_idx}: ❌ FAILED - No detection")
    else:
        print(f"\nFrame {frame_idx}: ❌ No ground truth available")

print("\n" + "="*80)
if results:
    errors = [r['error'] for r in results]
    avg_error = np.mean(errors)
    max_error = np.max(errors)
    min_error = np.min(errors)
    
    print(f"Results for {len(results)} successful detections:")
    print(f"  Average error: {avg_error:.3f}m")
    print(f"  Min error:     {min_error:.3f}m") 
    print(f"  Max error:     {max_error:.3f}m")
    print("="*80)
else:
    print("❌ NO SUCCESSFUL DETECTIONS!")
    print("="*80)
