"""Force reload model and test detection"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Clear the cached model
import volleyball_detection
volleyball_detection._MODEL = None

from camera_config import load_camera_config
config = load_camera_config()

print("="*60)
print("MODEL PATH CHECK")
print("="*60)
print(f"\nConfig detection_model_path: {config.get('detection_model_path')}")

# Force load model
from volleyball_detection import _get_model
model = _get_model()

print(f"Model classes: {model.names}")
print(f"\nLooking for 'Volleyball' or 'volleyball'...")
volleyball_classes = [k for k, v in model.names.items() if 'volley' in v.lower()]
print(f"Found: {volleyball_classes}")

if volleyball_classes:
    print(f"✅ Volleyball detection model loaded!")
else:
    print(f"❌ Wrong model - this is the base YOLO model!")
    print(f"   The model at the configured path doesn't exist or failed to load")
    
    # Check what path it tried
    from pathlib import Path
    expected_path = Path(config.get('detection_model_path'))
    if not expected_path.is_absolute():
        expected_path = Path(__file__).parent / "src" / expected_path
    print(f"\n   Expected path: {expected_path}")
    print(f"   Exists: {expected_path.exists()}")

print("\n" + "="*60)
print("Now testing detection on frame 99:")
print("="*60)

import cv2
from volleyball_detection import get_ball_xy

left_img = cv2.imread("output_frames/left/left3_0099.jpg")
if left_img is not None:
    x, y = get_ball_xy(left_img)
    if x is not None:
        print(f"✅ Ball detected at ({x}, {y})")
    else:
        print(f"❌ No ball detected")
