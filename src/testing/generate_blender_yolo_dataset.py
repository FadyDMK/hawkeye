import cv2
import pandas as pd
import numpy as np
import os
import shutil
from scipy.spatial.transform import Rotation as R_scipy

# Paths
VIDEO_PATH = r"f:\hawkeye\test-vids\finalLeft.mkv"
CSV_PATH = r"f:\hawkeye\3D-models\Latest volley go brr\ball_positions_blender.csv"
OUTPUT_DIR = r"f:\hawkeye\dataset\blender_val"
IMAGES_DIR = os.path.join(OUTPUT_DIR, "images")
LABELS_DIR = os.path.join(OUTPUT_DIR, "labels")
YAML_PATH = os.path.join(OUTPUT_DIR, "data.yaml")

# Camera Parameters (from hawkeye_pipeline.py)
FOCAL_LENGTH_PX = 1600.0
RES_W = 1920
RES_H = 1080
CX = RES_W / 2
CY = RES_H / 2

# Original Extrinsics
R_wc_orig = np.array([
    [-0.016771, 0.19022 , -0.9816 ],
    [ 0.99516 , 0.098228,  0.002032],
    [ 0.096807, -0.97682, -0.19094]
])
t_wc_orig = np.array([17.569, -6.6545, 6.1974])

# Optimized Parameters (from align_blender_data.py)
pitch = -0.14302803
yaw = -3.88321226
roll = 3.56747381
tx = 0.49412316
ty = 0.67756182
tz = -0.50040697

# Apply Correction
r_corr = R_scipy.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix()
R_wc_new = R_wc_orig @ r_corr
t_wc_new = t_wc_orig + np.array([tx, ty, tz])

# Invert for World -> Camera
R_cw = R_wc_new.T
t_cw = -R_cw @ t_wc_new

BALL_RADIUS_M = 0.20
FRAME_OFFSET = 1

def create_dataset():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(IMAGES_DIR)
    os.makedirs(LABELS_DIR)

    # Load CSV
    df = pd.read_csv(CSV_PATH)
    # Ensure columns are lower case
    df.columns = df.columns.str.lower()

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    # Process first 150 frames
    for i in range(150):
        ret, frame = cap.read()
        if not ret:
            break
            
        video_frame_num = i + 1
        csv_frame_num = video_frame_num + FRAME_OFFSET
        
        row = df[df['frame'] == csv_frame_num]
        if row.empty:
            continue
            
        P_world = np.array([row.iloc[0]['x'], row.iloc[0]['y'], row.iloc[0]['z']])
        
        # Transform to Camera Frame
        P_camera = R_cw @ P_world + t_cw
        Xc, Yc, Zc = P_camera
        
        if Zc <= 0:
            print(f"Frame {video_frame_num}: Ball behind camera (Z={Zc})")
            continue
            
        # Project to Pixel
        u = FOCAL_LENGTH_PX * Xc / Zc + CX
        v = FOCAL_LENGTH_PX * Yc / Zc + CY
        
        # Calculate BBox size
        radius_px = FOCAL_LENGTH_PX * BALL_RADIUS_M / Zc
        w_px = 2 * radius_px
        h_px = 2 * radius_px
        
        # YOLO Format: x_center, y_center, w, h (normalized)
        x_center = u / RES_W
        y_center = v / RES_H
        w = w_px / RES_W
        h = h_px / RES_H
        
        # Check if inside image (with some margin)
        if 0 <= x_center <= 1 and 0 <= y_center <= 1:
            # Save Image
            img_filename = f"frame_{i:03d}.jpg"
            cv2.imwrite(os.path.join(IMAGES_DIR, img_filename), frame)
            
            # Save Label
            label_filename = f"frame_{i:03d}.txt"
            with open(os.path.join(LABELS_DIR, label_filename), "w") as f:
                f.write(f"0 {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}\n")
        else:
            # print(f"Frame {video_frame_num}: Ball out of view ({u:.1f}, {v:.1f})")
            pass

    cap.release()
    
    # Create YAML
    with open(YAML_PATH, "w") as f:
        f.write(f"path: {OUTPUT_DIR}\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("names:\n")
        f.write("  0: Volleyball\n")
        
    print(f"Dataset created at {OUTPUT_DIR}")

if __name__ == "__main__":
    create_dataset()
