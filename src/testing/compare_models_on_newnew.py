import sys
import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import volleyball_detection
from hawkeye_pipeline import HawkeyePipeline

# Define paths
VIDEO_LEFT = r"f:\hawkeye\test-vids\newnewLeft.mkv"
VIDEO_RIGHT = r"f:\hawkeye\test-vids\newnewRight.mkv"
GROUND_TRUTH_CSV = r"f:\hawkeye\test-vids\ball_positions_ground_truth_180.csv"

MODELS = {
    "Train 13": r"f:\hawkeye\models\runs\detect\train13\weights\best.pt",
    "Train 18": r"f:\hawkeye\models\runs\detect\train18\weights\best.pt",
    "Retrain Hard Frames 2": r"f:\hawkeye\models\runs\detect\retrain_with_hard_frames2\weights\best.pt"
}

def calculate_rmse(predicted, actual):
    """Calculate RMSE between two lists of (x, y, z) tuples."""
    # Align by frame number
    merged = pd.merge(predicted, actual, on="frame", suffixes=('_pred', '_true'))
    if len(merged) == 0:
        return float('inf')
    
    diff_x = merged['x_pred'] - merged['x_true']
    diff_y = merged['y_pred'] - merged['y_true']
    diff_z = merged['z_pred'] - merged['z_true']
    
    squared_error = diff_x**2 + diff_y**2 + diff_z**2
    rmse = np.sqrt(squared_error.mean())
    return rmse

def run_evaluation():
    # Load ground truth
    gt_df = pd.read_csv(GROUND_TRUTH_CSV)
    # Ensure columns are lower case
    gt_df.columns = gt_df.columns.str.lower()
    
    results = []

    for model_name, model_path in MODELS.items():
        print(f"\nEvaluating {model_name}...")
        
        # Force load the specific model
        if os.path.exists(model_path):
            print(f"Loading weights from {model_path}")
            volleyball_detection._MODEL = YOLO(model_path)
        else:
            print(f"Error: Model path not found: {model_path}")
            continue

        # Initialize pipeline
        pipeline = HawkeyePipeline()
        
        # Open videos
        cap_left = cv2.VideoCapture(VIDEO_LEFT)
        cap_right = cv2.VideoCapture(VIDEO_RIGHT)
        
        predicted_positions = []
        
        frame_count = 0
        max_frames = 100 # As per user request
        
        while cap_left.isOpened() and cap_right.isOpened() and frame_count < max_frames:
            ret_l, frame_l = cap_left.read()
            ret_r, frame_r = cap_right.read()
            
            if not ret_l or not ret_r:
                break
                
            frame_count += 1
            
            # Process frame
            pipeline.process_from_pair(frame_l, frame_r, frame_num=frame_count)
            
            # Get the last result
            # The pipeline appends to ball_positions_world. 
            # Since we create a new pipeline for each model, the index should align with frame_count-1
            if len(pipeline.ball_positions_world) >= frame_count:
                pos = pipeline.ball_positions_world[frame_count-1]
                if pos is not None and pos[0] is not None:
                    predicted_positions.append({
                        "frame": frame_count,
                        "x": pos[0],
                        "y": pos[1],
                        "z": pos[2]
                    })
        
        cap_left.release()
        cap_right.release()
        
        pred_df = pd.DataFrame(predicted_positions)
        
        if len(pred_df) > 0:
            rmse = calculate_rmse(pred_df, gt_df)
            detection_rate = (len(pred_df) / max_frames) * 100
            print(f"RMSE: {rmse:.4f} m")
            print(f"Detection Rate: {detection_rate:.1f}%")
            
            results.append({
                "Model": model_name,
                "RMSE (m)": rmse,
                "Detection Rate (%)": detection_rate,
                "Frames Detected": len(pred_df)
            })
        else:
            print("No detections made.")
            results.append({
                "Model": model_name,
                "RMSE (m)": float('inf'),
                "Detection Rate (%)": 0.0,
                "Frames Detected": 0
            })

    # Summary
    print("\n--- NewNew Footage Evaluation Summary ---")
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    output_path = r"f:\hawkeye\output\newnew_model_comparison.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\nSaved results to {output_path}")

if __name__ == "__main__":
    run_evaluation()
