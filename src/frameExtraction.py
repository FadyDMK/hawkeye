import cv2
import os


def extract_frames(video_path, output_folder, frame_interval=1):
    os.makedirs(output_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = 0
    success, image = cap.read()
    print(success)
    video_filename = os.path.basename(video_path)
    
    while success:
        print(f"Extracting frames from {video_filename} at {fps} fps")
        if frame_count % frame_interval == 0:
            filename = os.path.join(output_folder, f"{os.path.splitext(os.path.basename(video_path))[0]}_{frame_count:04d}.jpg")
            cv2.imwrite(filename, image)
        success, image = cap.read()
        frame_count += 1

    cap.release()
root = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(root, "..\\data")

extract_frames(os.path.join(data_folder,"left3.mp4" ), "output_frames/left")
extract_frames(os.path.join(data_folder,"right3.mp4"), "output_frames/right")
