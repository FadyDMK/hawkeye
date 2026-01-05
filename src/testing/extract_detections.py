import sys
import os
import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Define paths
VIDEO_LEFT = r"f:\hawkeye\test-vids\finalLeft.mkv"
MODEL_PATH = r"f:\hawkeye\models\runs\detect\train18\weights\best.pt"

def extract_detections():
    model = YOLO(MODEL_PATH)
    cap = cv2.VideoCapture(VIDEO_LEFT)
    
    frame_count = 0
    detections = []
    
    print("Frame,X,Y,Conf")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        frame_count += 1
        
        # Run inference
        results = model(frame, verbose=False)
        
        # Get boxes
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls == 0: # Volleyball
                    x, y, w, h = box.xywh[0].tolist()
                    conf = float(box.conf[0])
                    print(f"{frame_count},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f}")
                    detections.append({'frame': frame_count, 'x': x, 'y': y, 'conf': conf})
                    break # Only take first ball
        
        if frame_count > 150: # Limit to 150 frames
            break
            
    cap.release()

if __name__ == "__main__":
    extract_detections()
