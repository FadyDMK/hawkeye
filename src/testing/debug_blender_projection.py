import cv2
import pandas as pd
import numpy as np
import os

# Paths
VIDEO_PATH = r"f:\hawkeye\test-vids\finalLeft.mkv"
CSV_PATH = r"f:\hawkeye\3D-models\Latest volley go brr\ball_positions_blender.csv"
OUTPUT_DIR = r"f:\hawkeye\dataset\blender_debug"

# Camera Parameters
FOCAL_LENGTH_PX = 1600.0
RES_W = 1920
RES_H = 1080
CX = RES_W / 2
CY = RES_H / 2

# Extrinsics
R = np.array([
    [-0.016771, 0.19022 , -0.9816 ],
    [ 0.99516 , 0.098228,  0.002032],
    [ 0.096807, -0.97682, -0.19094]
])
t = np.array([17.569, -6.6545, 6.1974])

BALL_RADIUS_M = 0.105

def debug_projection():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.lower()

    cap = cv2.VideoCapture(VIDEO_PATH)
    
    for i in range(20): # Check first 20 frames
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_num = i + 1
        row = df[df['frame'] == frame_num]
        
        if not row.empty:
            P_world = np.array([row.iloc[0]['x'], row.iloc[0]['y'], row.iloc[0]['z']])
            P_camera = R.T @ (P_world - t)
            Xc, Yc, Zc = P_camera
            
            if Zc > 0:
                u = int(FOCAL_LENGTH_PX * Xc / Zc + CX)
                v = int(FOCAL_LENGTH_PX * Yc / Zc + CY)
                
                radius_px = int(FOCAL_LENGTH_PX * BALL_RADIUS_M / Zc)
                
                # Draw circle
                cv2.circle(frame, (u, v), radius_px, (0, 255, 0), 2)
                cv2.putText(frame, f"Frame {frame_num}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                cv2.imwrite(os.path.join(OUTPUT_DIR, f"debug_{i:03d}.jpg"), frame)
                print(f"Saved debug_{i:03d}.jpg: ({u}, {v})")
            else:
                print(f"Frame {frame_num}: Zc <= 0")
        else:
            print(f"Frame {frame_num}: No data")

    cap.release()

if __name__ == "__main__":
    debug_projection()
