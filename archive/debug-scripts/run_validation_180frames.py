"""
Validation script for 180-frame finalLeft/finalRight videos.
Uses the ACTUAL HawkeyePipeline to process frames (not simplified version).

BEFORE RUNNING:
1. Run extract_ball_positions_180frames.py in Blender to generate ground truth CSV
2. Copy ball_positions_ground_truth_180.csv to test-vids folder
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
import os
import sys

# Add src to path to import HawkeyePipeline
sys.path.append('src')
from hawkeye_pipeline import HawkeyePipeline

class Validator180:
    def __init__(self):
        # Use the ACTUAL Hawkeye pipeline
        self.pipeline = HawkeyePipeline()
        
        # Disable bounds checking for validation - monkey patch to always return True
        # We want to measure ALL errors, even out-of-bounds ones
        self.pipeline.validate_court_bounds = lambda x, y, z: True
        
        # Storage
        self.results = []
    
    def process_frame_pair(self, left_frame, right_frame, frame_num, gt_pos=None):
        """Process one stereo pair using actual HawkeyePipeline."""
        
        result = {
            'frame': frame_num,
            'detection_left_success': False,
            'detection_right_success': False,
            'stereo_match_success': False,
            'reconstruction_success': False,
            'detection_time_ms': 0,
            'stereo_match_time_ms': 0,
            'triangulation_time_ms': 0,
            '3d_error_cm': None,
            'pred_x': None, 'pred_y': None, 'pred_z': None,
            'gt_x': None, 'gt_y': None, 'gt_z': None
        }
        
        # Ground truth
        if gt_pos is not None:
            result['gt_x'] = gt_pos[0]
            result['gt_y'] = gt_pos[1]
            result['gt_z'] = gt_pos[2]
        
        # Clear previous results for single-frame processing
        self.pipeline.clear_previous_results()
        
        # Process through actual pipeline
        start_total = time.perf_counter()
        success = self.pipeline.process_from_pair(left_frame, right_frame, frame_num=frame_num, display=False)
        total_time = (time.perf_counter() - start_total) * 1000
        
        result['detection_time_ms'] = total_time  # Total processing time
        
        # Check if pipeline produced camera coordinates (detection + stereo succeeded)
        if len(self.pipeline.ball_positions_camera) > 0:
            result['detection_left_success'] = True
            result['detection_right_success'] = True
            result['stereo_match_success'] = True
            
            cam_pos = self.pipeline.ball_positions_camera[-1]
            
            # Check if world coordinate was also computed (might be rejected by bounds check)
            if len(self.pipeline.ball_positions_world) > 0:
                result['reconstruction_success'] = True
                world_pos = self.pipeline.ball_positions_world[-1]
                result['pred_x'] = world_pos[0]
                result['pred_y'] = world_pos[1]
                result['pred_z'] = world_pos[2]
                
                # Calculate error if ground truth available
                if gt_pos is not None:
                    error = np.sqrt(
                        (world_pos[0] - gt_pos[0])**2 +
                        (world_pos[1] - gt_pos[1])**2 +
                        (world_pos[2] - gt_pos[2])**2
                    )
                    result['3d_error_cm'] = error * 100
            else:
                # Detection succeeded but bounds check rejected the result
                # For validation, we still want to know this happened
                result['reconstruction_success'] = False
        
        return result
    
    def validate_videos(self):
        """Process finalLeft.mkv and finalRight.mkv."""
        
        left_video_path = 'test-vids/finalLeft.mkv'
        right_video_path = 'test-vids/finalRight.mkv'
        gt_path = 'test-vids/ball_positions_ground_truth_180.csv'
        
        # Check files exist
        if not os.path.exists(left_video_path):
            print(f"ERROR: {left_video_path} not found!")
            return
        if not os.path.exists(right_video_path):
            print(f"ERROR: {right_video_path} not found!")
            return
        if not os.path.exists(gt_path):
            print(f"ERROR: {gt_path} not found!")
            print("Run extract_ball_positions_180frames.py in Blender first!")
            return
        
        # Load ground truth
        gt_df = pd.read_csv(gt_path)
        print(f"✓ Loaded ground truth: {len(gt_df)} frames")
        
        # Open videos
        left_cap = cv2.VideoCapture(left_video_path)
        right_cap = cv2.VideoCapture(right_video_path)
        
        left_frame_count = int(left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        right_frame_count = int(right_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"✓ Left video: {left_frame_count} frames")
        print(f"✓ Right video: {right_frame_count} frames")
        print(f"\n{'='*70}")
        print("PROCESSING FRAMES")
        print('='*70)
        
        frame_num = 1  # Start from 1 to match Blender ground truth (1-indexed)
        while True:
            ret_left, left_frame = left_cap.read()
            ret_right, right_frame = right_cap.read()
            
            if not ret_left or not ret_right:
                break
            
            # Get ground truth for this frame (GT CSV is 1-indexed from Blender)
            gt_row = gt_df[gt_df['frame'] == frame_num]
            if len(gt_row) > 0:
                gt_pos = (gt_row.iloc[0]['x'], gt_row.iloc[0]['y'], gt_row.iloc[0]['z'])
            else:
                gt_pos = None
            
            # Process frame pair
            result = self.process_frame_pair(left_frame, right_frame, frame_num, gt_pos)
            self.results.append(result)
            
            # Progress
            if frame_num % 20 == 0:
                print(f"  Frame {frame_num}: ", end='')
                if result['reconstruction_success']:
                    if result['3d_error_cm'] is not None:
                        print(f"✓ Success (error: {result['3d_error_cm']:.1f} cm)")
                    else:
                        print(f"✓ Success (no GT)")
                else:
                    print(f"✗ Failed at {'detection' if not result['detection_left_success'] or not result['detection_right_success'] else 'matching'}")
            
            frame_num += 1
        
        left_cap.release()
        right_cap.release()
        
        print(f"\n✓ Processed {frame_num} frames")
        
        # Analyze results
        self.analyze_results()
        self.analyze_relative_accuracy()
    
    def analyze_results(self):
        """Print comprehensive metrics."""
        
        df = pd.DataFrame(self.results)
        
        print("\n" + "="*70)
        print("VALIDATION RESULTS - 180 FRAME DATASET")
        print("="*70)
        
        # Detection Performance
        print("\n### DETECTION PERFORMANCE ###\n")
        total = len(df)
        left_success = df['detection_left_success'].sum()
        right_success = df['detection_right_success'].sum()
        both_detected = (df['detection_left_success'] & df['detection_right_success']).sum()
        
        print(f"Left camera detection:  {left_success}/{total} ({left_success/total*100:.1f}%)")
        print(f"Right camera detection: {right_success}/{total} ({right_success/total*100:.1f}%)")
        print(f"Both cameras detected:  {both_detected}/{total} ({both_detected/total*100:.1f}%)")
        
        # Stereo Matching
        print("\n### STEREO MATCHING PERFORMANCE ###\n")
        match_success = df['stereo_match_success'].sum()
        print(f"Stereo matching success: {match_success}/{both_detected} ({match_success/both_detected*100:.1f}% of detected pairs)")
        
        # 3D Reconstruction Accuracy
        valid_errors = df[df['3d_error_cm'].notna()]['3d_error_cm']
        if len(valid_errors) > 0:
            print("\n### 3D RECONSTRUCTION ACCURACY (RAW) ###\n")
            print(f"Valid 3D reconstructions: {len(valid_errors)}")
            print(f"Mean error:       {valid_errors.mean():.1f} cm")
            print(f"Median error:     {valid_errors.median():.1f} cm")
            print(f"Std deviation:    {valid_errors.std():.1f} cm")
            print(f"95th percentile:  {valid_errors.quantile(0.95):.1f} cm")
            
            ball_d = 21.0
            print(f"\nRelative to ball diameter (21 cm):")
            print(f"Median: {valid_errors.median()/ball_d:.2f}× ball diameter")
            print(f"95th:   {valid_errors.quantile(0.95)/ball_d:.2f}× ball diameter")
        
        # System Performance
        print("\n### SYSTEM PERFORMANCE ###\n")
        det_times = df['detection_time_ms']
        
        print(f"Total per frame:    {det_times.mean():.1f} ms ({1000/det_times.mean():.1f} FPS)")
        print(f"  Min: {det_times.min():.1f} ms")
        print(f"  Max: {det_times.max():.1f} ms")
        
        # Overall Pipeline
        print("\n### COMBINED PIPELINE PERFORMANCE ###\n")
        final_success = df['reconstruction_success'].sum()
        print(f"Overall pipeline success: {final_success}/{total} ({final_success/total*100:.1f}%)")
        
        # Save results
        os.makedirs('output', exist_ok=True)
        df.to_csv('output/validation_180frames_results.csv', index=False)
        print(f"\n✓ Detailed results saved to output/validation_180frames_results.csv")
    
    def umeyama_alignment(self, source, target):
        """Compute similarity transform (scale, rotation, translation) that aligns source to target.
        
        Returns scale, R, t such that: target ≈ scale * R @ source + t
        """
        assert source.shape == target.shape
        n, dim = source.shape

        source_mean = source.mean(axis=0)
        target_mean = target.mean(axis=0)

        source_centered = source - source_mean
        target_centered = target - target_mean

        cov = target_centered.T @ source_centered / n

        U, D, Vt = np.linalg.svd(cov)

        S = np.eye(dim)
        if np.linalg.det(U) * np.linalg.det(Vt) < 0:
            S[-1, -1] = -1

        R = U @ S @ Vt
        var_source = np.sum(source_centered ** 2) / n
        scale = np.trace(np.diag(D) @ S) / var_source

        t = target_mean - scale * R @ source_mean

        return scale, R, t
    
    def analyze_relative_accuracy(self):
        """Calculate accuracy after Umeyama alignment (same as 100-frame validation)."""
        
        df = pd.DataFrame(self.results)
        valid = df[df['reconstruction_success'] & df['3d_error_cm'].notna()].copy()
        
        if len(valid) == 0:
            return
        
        print("\n" + "="*70)
        print("RELATIVE ACCURACY ANALYSIS (Umeyama Alignment)")
        print("="*70)
        
        # Prepare point clouds
        pred_points = valid[['pred_x', 'pred_y', 'pred_z']].values
        gt_points = valid[['gt_x', 'gt_y', 'gt_z']].values
        
        # Compute optimal similarity transform using Umeyama
        scale, R, t = self.umeyama_alignment(pred_points, gt_points)
        
        print(f"\nOptimal similarity transform:")
        print(f"  Scale: {scale:.6f}")
        print(f"  Rotation matrix:")
        for row in R:
            print(f"    [{row[0]:+.6f}, {row[1]:+.6f}, {row[2]:+.6f}]")
        print(f"  Translation: [{t[0]:+.4f}, {t[1]:+.4f}, {t[2]:+.4f}] m")
        
        # Apply transformation to predictions
        aligned_points = (scale * (R @ pred_points.T)).T + t
        
        # Calculate aligned errors
        aligned_errors = np.linalg.norm(aligned_points - gt_points, axis=1) * 100  # cm
        
        print(f"\n### ALIGNED 3D RECONSTRUCTION ACCURACY ###")
        print(f"(After Umeyama alignment: scale + rotation + translation)\n")
        print(f"Valid reconstructions: {len(aligned_errors)}")
        print(f"Mean error:       {np.mean(aligned_errors):.1f} cm")
        print(f"Median error:     {np.median(aligned_errors):.1f} cm")
        print(f"Std deviation:    {np.std(aligned_errors):.1f} cm")
        print(f"95th percentile:  {np.percentile(aligned_errors, 95):.1f} cm")
        
        ball_d = 21.0
        print(f"\nRelative to ball diameter (21 cm):")
        print(f"Median: {np.median(aligned_errors)/ball_d:.2f}× ball diameter")
        print(f"95th:   {np.percentile(aligned_errors, 95)/ball_d:.2f}× ball diameter")
        
        # RMS error
        rms = np.sqrt(np.mean(aligned_errors**2))
        print(f"\nRMS error: {rms:.1f} cm")

def main():
    validator = Validator180()
    validator.validate_videos()

if __name__ == '__main__':
    main()
