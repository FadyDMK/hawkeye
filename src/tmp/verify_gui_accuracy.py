"""
Comprehensive test showing exactly what the GUI displays vs ground truth.
This processes frames 85-100 through the exact same pipeline as the GUI.
"""

import sys
import cv2
import numpy as np
import pandas as pd
sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline

# Initialize pipeline
pipeline = HawkeyePipeline()

print("="*80)
print("GUI ACCURACY VERIFICATION - FRAMES 85-100")
print("="*80)
print(f"Configuration:")
print(f"  Baseline: {pipeline.config['baseline_m']}m")
print(f"  Scale: {pipeline.scale}")
print(f"  Translation: {pipeline.t}")
print("="*80)

# Load videos
left_cap = cv2.VideoCapture('data/left3.mp4')
right_cap = cv2.VideoCapture('data/right3.mp4')

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

results = []

for frame_num in range(85, 101):
    left_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    right_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
    
    ret_left, left_frame = left_cap.read()
    ret_right, right_frame = right_cap.read()
    
    if not ret_left or not ret_right:
        continue
    
    # Process through pipeline
    pipeline.clear_previous_results()
    result = pipeline.process_from_pair(left_frame, right_frame, frame_num=frame_num, display=False)
    
    if frame_num in blender_df['frame'].values:
        row = blender_df[blender_df['frame'] == frame_num].iloc[0]
        gt = np.array([row['x'], row['y'], row['z']])
        
        if result and len(pipeline.ball_positions_world) > 0:
            world_pos = np.array(pipeline.ball_positions_world[-1])
            
            if None not in world_pos:
                error_vec = world_pos - gt
                error = np.linalg.norm(error_vec)
                
                results.append({
                    'frame': frame_num,
                    'gui_x': world_pos[0],
                    'gui_y': world_pos[1],
                    'gui_z': world_pos[2],
                    'gt_x': gt[0],
                    'gt_y': gt[1],
                    'gt_z': gt[2],
                    'err_x': error_vec[0],
                    'err_y': error_vec[1],
                    'err_z': error_vec[2],
                    'error_3d': error
                })
                
                # Print detailed info
                status = "✅ GOOD" if error < 1.0 else "⚠️ POOR"
                print(f"\nFrame {frame_num}: {status} (Error: {error:.3f}m)")
                print(f"  GUI shows:  X={world_pos[0]:7.3f}, Y={world_pos[1]:7.3f}, Z={world_pos[2]:7.3f}")
                print(f"  Should be:  X={gt[0]:7.3f}, Y={gt[1]:7.3f}, Z={gt[2]:7.3f}")
                print(f"  Error:      X={error_vec[0]:7.3f}, Y={error_vec[1]:7.3f}, Z={error_vec[2]:7.3f}")
            else:
                print(f"\nFrame {frame_num}: ❌ FAILED - Got None values")
        else:
            print(f"\nFrame {frame_num}: ❌ FAILED - No detection")

left_cap.release()
right_cap.release()

if results:
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    errors = [r['error_3d'] for r in results]
    err_x = [r['err_x'] for r in results]
    err_y = [r['err_y'] for r in results]
    err_z = [r['err_z'] for r in results]
    
    print(f"Frames processed: {len(results)}/16")
    print(f"\n3D Error:")
    print(f"  Average: {np.mean(errors):.3f}m")
    print(f"  Min:     {np.min(errors):.3f}m (frame {results[np.argmin(errors)]['frame']})")
    print(f"  Max:     {np.max(errors):.3f}m (frame {results[np.argmax(errors)]['frame']})")
    print(f"  Std Dev: {np.std(errors):.3f}m")
    
    print(f"\nError by axis:")
    print(f"  X: {np.mean(np.abs(err_x)):.3f}m avg, {np.std(err_x):.3f}m std")
    print(f"  Y: {np.mean(np.abs(err_y)):.3f}m avg, {np.std(err_y):.3f}m std")
    print(f"  Z: {np.mean(np.abs(err_z)):.3f}m avg, {np.std(err_z):.3f}m std")
    
    good_frames = [r['frame'] for r in results if r['error_3d'] < 1.0]
    poor_frames = [r['frame'] for r in results if r['error_3d'] >= 1.0]
    
    print(f"\nAccuracy breakdown:")
    print(f"  Good (<1m error): {len(good_frames)} frames - {good_frames}")
    print(f"  Poor (≥1m error): {len(poor_frames)} frames - {poor_frames}")
    
    # Save to CSV
    df = pd.DataFrame(results)
    df.to_csv('gui_accuracy_verification.csv', index=False)
    print(f"\n📄 Detailed results saved to: gui_accuracy_verification.csv")
    
    print("\n" + "="*80)
    print("INTERPRETATION:")
    print("="*80)
    if np.mean(errors) < 0.5:
        print("✅ EXCELLENT: Average error < 0.5m - system is working very well!")
        print("   This matches professional Hawkeye accuracy expectations.")
    elif np.mean(errors) < 1.0:
        print("✅ GOOD: Average error < 1m - acceptable for most applications")
    elif np.mean(errors) < 2.0:
        print("⚠️ MODERATE: Average error < 2m - usable but needs improvement")
    else:
        print("❌ POOR: Average error ≥ 2m - calibration may be wrong")
        
    print("\n" + "="*80)
    print("TROUBLESHOOTING:")
    print("="*80)
    print("If GUI shows different coordinates than this test:")
    print("  1. Check which frame number the GUI is showing")
    print("  2. Make sure you're looking at frames 85-100 (calibrated range)")
    print("  3. Restart the GUI to ensure it loads the updated code")
    print("  4. Compare GUI coordinates with gui_accuracy_verification.csv")
    print("\nIf GUI shows coordinates WAY OFF (like Y=30-40m):")
    print("  • You're probably looking at frames 1-84 (uncalibrated range)")
    print("  • Scale factors only work for frames 85-100")
    print("  • Navigate to frames 85-100 for accurate results")
    print("="*80)
else:
    print("\n❌ NO SUCCESSFUL DETECTIONS!")
    print("Something is seriously wrong with the detection or processing pipeline.")
