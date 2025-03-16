import cv2
import os

def extract_frames(video_path, output_folder, frame_interval=10):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    success, image = cap.read()

    while success:
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
            cv2.imwrite(filename, image)
        success, image = cap.read()
        frame_count += 1

    cap.release()

# Example Usage:
extract_frames("./data/testFootage/videos/testFootageL.mp4", "output_frames/left")
extract_frames("./data/testFootage/videos/testFootageR.mp4", "output_frames/right")
