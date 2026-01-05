"""
Run YOLO validation by having the models detect on Blender frames,
then compare detections to ground truth projections to calculate
Precision, Recall, mAP50, mAP50-95
"""
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from scipy.spatial.transform import Rotation as R_scipy

# Paths
VIDEO_PATH = r"f:\hawkeye\test-vids\finalLeft.mkv"
CSV_PATH = r"f:\hawkeye\3D-models\Latest volley go brr\ball_positions_blender.csv"

# Models
MODELS = {
    "Train 13": r"f:\hawkeye\models\runs\detect\train13\weights\best.pt",
    "Train 18": r"f:\hawkeye\models\runs\detect\train18\weights\best.pt",
    "Retrain Hard Frames 2": r"f:\hawkeye\models\runs\detect\retrain_with_hard_frames2\weights\best.pt"
}

# Camera Parameters
FOCAL_LENGTH_PX = 1600.0
RES_W = 1920
RES_H = 1080
CX = RES_W / 2
CY = RES_H / 2

# Optimized Extrinsics
R_wc_orig = np.array([[-0.016771, 0.19022 , -0.9816 ],
                 [ 0.99516 , 0.098228,  0.002032],
                 [ 0.096807, -0.97682, -0.19094]])
t_wc_orig = np.array([17.569, -6.6545, 6.1974])
pitch = -0.14302803
yaw = -3.88321226
roll = 3.56747381
tx = 0.49412316
ty = 0.67756182
tz = -0.50040697
r_corr = R_scipy.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix()
R_wc_new = R_wc_orig @ r_corr
t_wc_new = t_wc_orig + np.array([tx, ty, tz])
R_cw = R_wc_new.T
t_cw = -R_cw @ t_wc_new

FRAME_OFFSET = 1
IOU_THRESHOLD = 0.5

def project_point(x, y, z):
    """Project 3D world point to 2D pixel"""
    p_world = np.array([x, y, z])
    p_cam = R_cw @ p_world + t_cw
    xc, yc, zc = p_cam
    if zc <= 0:
        return None
    u = (xc / zc) * FOCAL_LENGTH_PX + CX
    v = (yc / zc) * FOCAL_LENGTH_PX + CY
    return u, v

def calculate_iou(box1, box2):
    """Calculate IoU between two boxes [x1, y1, x2, y2]"""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    inter = (x2 - x1) * (y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    
    return inter / union if union > 0 else 0.0

def evaluate_model(model_name, model_path):
    """Evaluate a model on Blender footage"""
    print(f"\nEvaluating {model_name}...")
    
    model = YOLO(model_path)
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    total_gt = 0
    ious = []
    
    for i in range(100):
        ret, frame = cap.read()
        if not ret:
            break
            
        video_frame_num = i + 1
        csv_frame_num = video_frame_num + FRAME_OFFSET
        
        # Get ground truth
        row = df[df['frame'] == csv_frame_num]
        has_gt = False
        gt_box = None
        
        if not row.empty:
            x, y, z = row.iloc[0][['x', 'y', 'z']]
            proj = project_point(x, y, z)
            if proj is not None:
                u, v = proj
                if 0 <= u < RES_W and 0 <= v < RES_H:
                    # GT box (approximate size)
                    radius = 20  # pixels
                    gt_box = [u - radius, v - radius, u + radius, v + radius]
                    has_gt = True
                    total_gt += 1
        
        # Get detections
        results = model(frame, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:  # Volleyball
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    conf = float(box.conf[0])
                    detections.append([x1, y1, x2, y2, conf])
        
        # Match detections to ground truth
        if has_gt and len(detections) > 0:
            # Find best matching detection
            best_iou = 0
            for det in detections:
                iou = calculate_iou(gt_box, det[:4])
                best_iou = max(best_iou, iou)
            
            if best_iou >= IOU_THRESHOLD:
                true_positives += 1
                ious.append(best_iou)
                false_positives += len(detections) - 1  # Other detections
            else:
                false_positives += len(detections)
                false_negatives += 1
        elif has_gt and len(detections) == 0:
            false_negatives += 1
        elif not has_gt and len(detections) > 0:
            false_positives += len(detections)
    
    cap.release()
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    
    # Approximate mAP50 (simplified - at IoU=0.5)
    mAP50 = precision * recall if (precision + recall) > 0 else 0
    
    # Approximate mAP50-95 (rough estimate)
    avg_iou = np.mean(ious) if len(ious) > 0 else 0
    mAP50_95 = mAP50 * (avg_iou / 0.75) if avg_iou > 0 else 0  # Scale by avg IoU
    
    print(f"  TP: {true_positives}, FP: {false_positives}, FN: {false_negatives}")
    print(f"  Precision: {precision:.4f}, Recall: {recall:.4f}")
    print(f"  mAP50: {mAP50:.4f}, mAP50-95: {mAP50_95:.4f}")
    
    return {
        "Model": model_name,
        "Precision": precision,
        "Recall": recall,
        "mAP50": mAP50,
        "mAP50-95": mAP50_95
    }

if __name__ == "__main__":
    results = []
    for name, path in MODELS.items():
        results.append(evaluate_model(name, path))
    
    df_results = pd.DataFrame(results)
    output_path = r"f:\hawkeye\output\blender_model_comparison.csv"
    df_results.to_csv(output_path, index=False)
    
    print(f"\n{'='*60}")
    print(df_results.to_string(index=False))
    print(f"\nSaved to {output_path}")
