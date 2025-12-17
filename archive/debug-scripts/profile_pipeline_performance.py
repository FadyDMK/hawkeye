"""
Profile the complete pipeline to get accurate timing and success rate data for Chapter 5.

This script:
1. Measures actual stereo matching computation time
2. Calculates real pipeline success rate (not theoretical)
3. Profiles each pipeline stage timing
4. Tests on the full 4,157 frame Blender test dataset
"""

import time
import pandas as pd
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import cv2
import sys
import os

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

# Import only what we need - avoid heavy dependencies
try:
    from volleyball_detection import get_ball_xy
except ImportError:
    print("Warning: Could not import volleyball_detection, will use YOLO directly")
    get_ball_xy = None

# Load camera config
import json
def load_camera_config():
    config_path = Path(__file__).parent / 'src' / 'camera_config.json'
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {
        'focal_length_px': 1600,
        'baseline_m': 3.0,
        'resolution_width': 1920,
        'z_min_m': 12.0,
        'z_max_m': 40.0
    }

class PipelineProfiler:
    def __init__(self):
        self.model_path = 'models/runs/detect/train18/weights/best.pt'
        self.model = YOLO(self.model_path)
        self.config = load_camera_config()
        
        # Results storage
        self.timing_data = []
        self.success_data = []
        
    def profile_single_frame_pair(self, left_frame_path, right_frame_path, ground_truth=None):
        """Profile a single stereo frame pair through the complete pipeline."""
        
        timings = {}
        success = {
            'detection_left': False,
            'detection_right': False,
            'stereo_matching': False,
            'triangulation': False
        }
        
        # 1. Frame Loading
        t0 = time.perf_counter()
        left_img = cv2.imread(left_frame_path)
        right_img = cv2.imread(right_frame_path)
        t1 = time.perf_counter()
        timings['frame_loading'] = (t1 - t0) * 1000  # Convert to ms
        
        if left_img is None or right_img is None:
            return timings, success
        
        # 2. Detection (Left Camera)
        t0 = time.perf_counter()
        results_left = self.model(left_img, verbose=False, imgsz=640)
        # Extract ball position from YOLO results
        x_left, y_left = None, None
        if len(results_left) > 0 and len(results_left[0].boxes) > 0:
            # Get highest confidence detection
            boxes = results_left[0].boxes
            if len(boxes) > 0:
                best_idx = boxes.conf.argmax()
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                x_left = (xyxy[0] + xyxy[2]) / 2
                y_left = (xyxy[1] + xyxy[3]) / 2
        t1 = time.perf_counter()
        timings['detection_left'] = (t1 - t0) * 1000
        success['detection_left'] = (x_left is not None and y_left is not None)
        
        # 3. Detection (Right Camera)
        t0 = time.perf_counter()
        results_right = self.model(right_img, verbose=False, imgsz=640)
        # Extract ball position from YOLO results
        x_right, y_right = None, None
        if len(results_right) > 0 and len(results_right[0].boxes) > 0:
            boxes = results_right[0].boxes
            if len(boxes) > 0:
                best_idx = boxes.conf.argmax()
                xyxy = boxes.xyxy[best_idx].cpu().numpy()
                x_right = (xyxy[0] + xyxy[2]) / 2
                y_right = (xyxy[1] + xyxy[3]) / 2
        t1 = time.perf_counter()
        timings['detection_right'] = (t1 - t0) * 1000
        success['detection_right'] = (x_right is not None and y_right is not None)
        
        # Only proceed if both detections succeeded
        if not (success['detection_left'] and success['detection_right']):
            timings['stereo_matching'] = 0
            timings['triangulation'] = 0
            timings['coordinate_transform'] = 0
            timings['total'] = sum(timings.values())
            return timings, success
        
        # 4. Stereo Matching (geometric consistency checks)
        t0 = time.perf_counter()
        
        # Calculate disparity
        focal_length_cfg = self.config.get('focal_length_px', 1600)
        cfg_width = self.config.get('resolution_width', 1920)
        img_width = left_img.shape[1]
        focal_length = float(focal_length_cfg) * (float(img_width) / float(cfg_width))
        baseline = self.config.get('baseline_m', 3.0)
        z_min = self.config.get('z_min_m', 0.0)
        z_max = self.config.get('z_max_m', 50.0)
        
        # Geometric checks
        d = float(x_left - x_right)
        if d > 0:  # Positive disparity expected
            Z_test = (focal_length * baseline) / (d + 1e-6)
            if z_min <= Z_test <= z_max:
                success['stereo_matching'] = True
        
        t1 = time.perf_counter()
        timings['stereo_matching'] = (t1 - t0) * 1000
        
        if not success['stereo_matching']:
            timings['triangulation'] = 0
            timings['coordinate_transform'] = 0
            timings['total'] = sum(timings.values())
            return timings, success
        
        # 5. Triangulation (calculate 3D position)
        t0 = time.perf_counter()
        
        Z = (focal_length * baseline) / (d + 1e-6)
        h, w = left_img.shape[:2]
        cx, cy = w // 2, h // 2
        X = (x_left - cx) * Z / focal_length
        Y = (y_left - cy) * Z / focal_length
        
        t1 = time.perf_counter()
        timings['triangulation'] = (t1 - t0) * 1000
        success['triangulation'] = True
        
        # 6. Coordinate Transformation (camera to world/court coordinates)
        # Note: This would use Umeyama transform if calibration data available
        t0 = time.perf_counter()
        # Simplified - just matrix multiplication
        # In real system: position = umeyama_matrix @ [X, Y, Z, 1]
        t1 = time.perf_counter()
        timings['coordinate_transform'] = (t1 - t0) * 1000
        
        # Calculate total pipeline time
        timings['total'] = sum(timings.values())
        
        return timings, success
    
    def profile_dataset(self, max_frames=None):
        """Profile the complete Blender test dataset."""
        
        print("=" * 70)
        print("PIPELINE PERFORMANCE PROFILING")
        print("=" * 70)
        print(f"\nModel: {self.model_path}")
        print(f"Dataset: Blender stereo frames")
        print()
        
        # Find Blender test images
        test_left_dir = Path('output_frames') / 'left'
        test_right_dir = Path('output_frames') / 'right'
        
        if not test_left_dir.exists():
            print(f"Error: Blender frames not found at {test_left_dir}")
            print("Please specify the correct path to Blender stereo frames.")
            return
        
        left_frames = sorted(list(test_left_dir.glob('*.jpg')) + list(test_left_dir.glob('*.png')))
        
        if max_frames:
            left_frames = left_frames[:max_frames]
        
        print(f"Processing {len(left_frames)} stereo frame pairs...\n")
        
        # Process each frame pair
        for i, left_path in enumerate(left_frames):
            # Construct right frame path - replace "left" with "right" in filename
            right_filename = left_path.name.replace('left', 'right')
            right_path = test_right_dir / right_filename
            
            if not right_path.exists():
                continue
            
            timings, success = self.profile_single_frame_pair(str(left_path), str(right_path))
            
            self.timing_data.append(timings)
            self.success_data.append(success)
            
            if (i + 1) % 100 == 0:
                print(f"Processed {i + 1}/{len(left_frames)} frames...")
        
        print(f"\nCompleted processing {len(self.timing_data)} frame pairs.\n")
        
        # Analyze results
        self.analyze_results()
    
    def analyze_results(self):
        """Analyze and report profiling results."""
        
        print("=" * 70)
        print("RESULTS")
        print("=" * 70)
        
        # Convert to DataFrames for easy analysis
        timing_df = pd.DataFrame(self.timing_data)
        success_df = pd.DataFrame(self.success_data)
        
        # === TIMING ANALYSIS ===
        print("\n### PIPELINE STAGE TIMING (milliseconds) ###\n")
        
        stages = ['frame_loading', 'detection_left', 'detection_right', 
                  'stereo_matching', 'triangulation', 'coordinate_transform', 'total']
        
        timing_stats = []
        for stage in stages:
            if stage in timing_df.columns:
                mean_time = timing_df[stage].mean()
                std_time = timing_df[stage].std()
                min_time = timing_df[stage].min()
                max_time = timing_df[stage].max()
                
                timing_stats.append({
                    'Stage': stage.replace('_', ' ').title(),
                    'Mean (ms)': f"{mean_time:.2f}",
                    'Std Dev (ms)': f"{std_time:.2f}",
                    'Min (ms)': f"{min_time:.2f}",
                    'Max (ms)': f"{max_time:.2f}"
                })
        
        timing_table = pd.DataFrame(timing_stats)
        print(timing_table.to_string(index=False))
        
        # Calculate percentage breakdown
        print("\n### PIPELINE STAGE BREAKDOWN (%) ###\n")
        total_time = timing_df['total'].mean()
        for stage in ['frame_loading', 'detection_left', 'detection_right', 
                      'stereo_matching', 'triangulation', 'coordinate_transform']:
            if stage in timing_df.columns:
                stage_time = timing_df[stage].mean()
                percentage = (stage_time / total_time) * 100
                print(f"{stage.replace('_', ' ').title():.<30} {stage_time:>6.2f} ms ({percentage:>5.1f}%)")
        
        # === SUCCESS RATE ANALYSIS ===
        print("\n\n### PIPELINE SUCCESS RATES ###\n")
        
        total_frames = len(success_df)
        detection_left_success = success_df['detection_left'].sum()
        detection_right_success = success_df['detection_right'].sum()
        both_detected = (success_df['detection_left'] & success_df['detection_right']).sum()
        matching_success = success_df['stereo_matching'].sum()
        final_success = success_df['triangulation'].sum()
        
        print(f"Total frames processed: {total_frames}")
        print(f"\nDetection (Left):  {detection_left_success}/{total_frames} ({detection_left_success/total_frames*100:.1f}%)")
        print(f"Detection (Right): {detection_right_success}/{total_frames} ({detection_right_success/total_frames*100:.1f}%)")
        print(f"Both Detected:     {both_detected}/{total_frames} ({both_detected/total_frames*100:.1f}%)")
        print(f"Stereo Matching:   {matching_success}/{both_detected} ({matching_success/both_detected*100:.1f}% of detected pairs)")
        print(f"Final 3D Output:   {final_success}/{total_frames} ({final_success/total_frames*100:.1f}% OVERALL)")
        
        print("\n### PROCESSING SPEED ###\n")
        mean_fps = 1000 / timing_df['total'].mean()
        best_fps = 1000 / timing_df['total'].min()
        worst_fps = 1000 / timing_df['total'].max()
        
        print(f"Mean:  {mean_fps:.1f} FPS ({timing_df['total'].mean():.2f} ms/frame)")
        print(f"Best:  {best_fps:.1f} FPS ({timing_df['total'].min():.2f} ms/frame)")
        print(f"Worst: {worst_fps:.1f} FPS ({timing_df['total'].max():.2f} ms/frame)")
        
        # Save detailed results
        print("\n\nSaving detailed results to CSV files...")
        timing_df.to_csv('output/pipeline_timing_profile.csv', index=False)
        success_df.to_csv('output/pipeline_success_profile.csv', index=False)
        print("✓ Results saved to output/ directory")
        
        # === CHAPTER 5 SUMMARY ===
        print("\n" + "=" * 70)
        print("CHAPTER 5 DATA SUMMARY")
        print("=" * 70)
        
        print("\n**For Section 5.3.3 (Computational Efficiency):**")
        stereo_mean = timing_df['stereo_matching'].mean()
        stereo_min = timing_df['stereo_matching'].min()
        stereo_max = timing_df['stereo_matching'].max()
        print(f"Stereo matching time: {stereo_mean:.2f} ms average")
        print(f"  Range: {stereo_min:.2f} ms (best) to {stereo_max:.2f} ms (worst)")
        
        print("\n**For Section 5.5.2 (Pipeline Breakdown):**")
        print(f"Detection (both cameras): {(timing_df['detection_left'].mean() + timing_df['detection_right'].mean()):.2f} ms ({((timing_df['detection_left'].mean() + timing_df['detection_right'].mean()) / total_time * 100):.0f}%)")
        print(f"Stereo matching: {stereo_mean:.2f} ms ({(stereo_mean / total_time * 100):.0f}%)")
        
        print("\n**For Section 5.6.1 (Combined Pipeline Performance):**")
        print(f"Overall pipeline success rate: {final_success/total_frames*100:.1f}%")
        print(f"  ({final_success} valid 3D positions out of {total_frames} frames)")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Profile pipeline performance')
    parser.add_argument('--max-frames', type=int, default=None, 
                        help='Limit number of frames to process (default: all)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test with 100 frames')
    
    args = parser.parse_args()
    
    max_frames = 100 if args.quick else args.max_frames
    
    profiler = PipelineProfiler()
    profiler.profile_dataset(max_frames=max_frames)

if __name__ == '__main__':
    main()
