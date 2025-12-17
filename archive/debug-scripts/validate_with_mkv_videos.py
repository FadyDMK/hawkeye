"""
Validate pipeline accuracy by processing the actual MKV videos
and comparing against Blender ground truth.
"""

import sys
import os
import cv2
import pandas as pd
import numpy as np
from pathlib import Path
import json

# Add src to path
sys.path.append(str(Path(__file__).parent / 'src'))

from volleyball_detection import get_ball_xy
from ultralytics import YOLO

def triangulate_ball(x_left, y_left, x_right, y_right, config):
    """Triangulate 3D position from 2D detections."""
    focal_length_cfg = config.get('focal_length_px', 1600)
    cfg_width = config.get('resolution_width', 1920)
    img_width = cfg_width  # Assume standard resolution
    focal_length = float(focal_length_cfg) * (float(img_width) / float(cfg_width))
    baseline = config.get('baseline_m', 3.0)
    
    # Calculate disparity
    d = float(x_left - x_right)
    if d <= 0:
        return None
    
    # Calculate depth
    Z = (focal_length * baseline) / (d + 1e-6)
    
    # Check depth range
    z_min = config.get('z_min_m', 0.0)
    z_max = config.get('z_max_m', 50.0)
    if not (z_min <= Z <= z_max):
        return None
    
    # Calculate X, Y (camera coordinates)
    h, w = 1080, 1920  # Standard resolution
    cx, cy = w // 2, h // 2
    X = (x_left - cx) * Z / focal_length
    Y = (y_left - cy) * Z / focal_length
    
    return (X, Y, Z)

def main():
    print("="*70)
    print("VALIDATION USING ACTUAL MKV VIDEOS")
    print("="*70)
    print()
    
    # Load config
    config_path = Path('src/camera_config.json')
    with open(config_path) as f:
        config = json.load(f)
    
    left_video = config.get('left_video_path', 'C:/tmp/newnewLeft.mkv')
    right_video = config.get('right_video_path', 'C:/tmp/newnewRight.mkv')
    model_path = 'models/runs/detect/train18/weights/best.pt'
    
    print(f"Left video:  {left_video}")
    print(f"Right video: {right_video}")
    print(f"Model:       {model_path}")
    
    # Check if videos exist
    if not os.path.exists(left_video):
        print(f"\nERROR: Left video not found at {left_video}")
        return
    if not os.path.exists(right_video):
        print(f"ERROR: Right video not found at {right_video}")
        return
    
    # Open videos
    cap_left = cv2.VideoCapture(left_video)
    cap_right = cv2.VideoCapture(right_video)
    
    if not cap_left.isOpened():
        print(f"ERROR: Could not open left video")
        return
    if not cap_right.isOpened():
        print(f"ERROR: Could not open right video")
        return
    
    print(f"\n✓ Videos opened successfully")
    
    # Load ground truth
    gt_path = Path('3D-models/Latest volley go brr/ball_positions_blender_correct.csv')
    if not gt_path.exists():
        print(f"\nERROR: Ground truth not found at {gt_path}")
        print("Please export from Blender first using export_ball_positions_camera_relative.py")
        return
    
    gt_df = pd.read_csv(gt_path)
    print(f"✓ Loaded ground truth: {len(gt_df)} frames\n")
    
    # Load detection model
    print("Loading YOLO model...")
    model = YOLO(model_path)
    print("✓ Model loaded\n")
    
    # Process frames 0-104
    results = []
    frame_idx = 0
    
    print("Processing frames...")
    
    while frame_idx < 105:
        ret_left, frame_left = cap_left.read()
        ret_right, frame_right = cap_right.read()
        
        if not (ret_left and ret_right):
            print(f"Warning: Could not read frame {frame_idx}")
            break
        
        # Detect ball in both frames
        results_left = model(frame_left, verbose=False, imgsz=640)
        results_right = model(frame_right, verbose=False, imgsz=640)
        
        # Get detections
        x_left, y_left = None, None
        x_right, y_right = None, None
        
        if len(results_left) > 0 and len(results_left[0].boxes) > 0:
            boxes = results_left[0].boxes
            best_idx = boxes.conf.argmax()
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            x_left = (xyxy[0] + xyxy[2]) / 2
            y_left = (xyxy[1] + xyxy[3]) / 2
        
        if len(results_right) > 0 and len(results_right[0].boxes) > 0:
            boxes = results_right[0].boxes
            best_idx = boxes.conf.argmax()
            xyxy = boxes.xyxy[best_idx].cpu().numpy()
            x_right = (xyxy[0] + xyxy[2]) / 2
            y_right = (xyxy[1] + xyxy[3]) / 2
        
        # Get ground truth
        gt_x = gt_df['x'].iloc[frame_idx]
        gt_y = gt_df['y'].iloc[frame_idx]
        gt_z = gt_df['z'].iloc[frame_idx]
        
        # Extract results
        frame_result = {
            'frame': frame_idx,
            'success': False,
            'pred_x': None,
            'pred_y': None,
            'pred_z': None,
            'gt_x': gt_x,
            'gt_y': gt_y,
            'gt_z': gt_z,
            'error_cm': None
        }
        
        # Triangulate if both detections exist
        if x_left is not None and x_right is not None:
            pos = triangulate_ball(x_left, y_left, x_right, y_right, config)
            if pos is not None:
                frame_result['success'] = True
                frame_result['pred_x'] = pos[0]
                frame_result['pred_y'] = pos[1]
                frame_result['pred_z'] = pos[2]
                
                # Calculate error
                error = np.sqrt((pos[0] - gt_x)**2 + 
                              (pos[1] - gt_y)**2 + 
                              (pos[2] - gt_z)**2)
                frame_result['error_cm'] = error * 100
        
        results.append(frame_result)
        
        if (frame_idx + 1) % 20 == 0:
            print(f"Processed {frame_idx + 1}/105 frames...")
        
        frame_idx += 1
    
    cap_left.release()
    cap_right.release()
    
    # Analyze results
    df = pd.DataFrame(results)
    valid = df[df['success']]
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print("="*70)
    
    print(f"\nPipeline success rate: {len(valid)}/{len(df)} ({len(valid)/len(df)*100:.1f}%)")
    
    if len(valid) > 0:
        errors = valid['error_cm']
        
        print(f"\n3D Reconstruction Accuracy:")
        print(f"  Mean error:       {errors.mean():.1f} cm")
        print(f"  Median error:     {errors.median():.1f} cm")
        print(f"  Std deviation:    {errors.std():.1f} cm")
        print(f"  95th percentile:  {errors.quantile(0.95):.1f} cm")
        
        # Relative to ball diameter
        ball_d = 21.0
        print(f"\n  Relative to ball diameter (21 cm):")
        print(f"    Mean:   {errors.mean()/ball_d:.2f}× ball diameter")
        print(f"    Median: {errors.median()/ball_d:.2f}× ball diameter")
        print(f"    95th:   {errors.quantile(0.95)/ball_d:.2f}× ball diameter")
        
        # Show some examples
        print(f"\n  Best 5 frames:")
        best = valid.nsmallest(5, 'error_cm')
        for _, row in best.iterrows():
            print(f"    Frame {int(row['frame'])}: {row['error_cm']:.1f} cm error")
        
        print(f"\n  Worst 5 frames:")
        worst = valid.nlargest(5, 'error_cm')
        for _, row in worst.iterrows():
            print(f"    Frame {int(row['frame'])}: {row['error_cm']:.1f} cm error")
        
        # Save results
        df.to_csv('output/mkv_validation_results.csv', index=False)
        print(f"\n✓ Results saved to output/mkv_validation_results.csv")
    else:
        print("\nNo successful reconstructions to analyze.")

if __name__ == '__main__':
    main()
