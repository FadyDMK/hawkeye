import cv2

video_path = r"f:\hawkeye\test-vids\finalLeft.mkv"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"Error opening {video_path}")
else:
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Frame count: {length}")
cap.release()
