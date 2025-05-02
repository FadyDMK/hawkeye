import cv2
import os

def load_videos(video_path1, video_path2):
    # Load the video
    video1 = cv2.VideoCapture(video_path1)
    video2 = cv2.VideoCapture(video_path2)

    # Check if the video is opened
    if not video1.isOpened():
        print("Error: Could not open the video.")
        print(video_path1)
        exit()
    if not video2.isOpened():
        print("Error: Could not open the video.")
        print(video_path2)
        exit()

    return video1, video2


def get_frame(video):
    # Read the frame
    ret, frame = video.read()

    # Check if the frame is read
    if not ret:
        print("Error: Could not read the frame.")
        exit()

    return frame

def synchronize_videos(video1, video2):
    # Get the frame rate
    fps = video1.get(cv2.CAP_PROP_FPS)

    # Synchronize the videos
    while True:
        # Get the frames
        frame1 = get_frame(video1)
        frame2 = get_frame(video2)

        # Check if the frames are read
        if frame1 is None or frame2 is None:
            break

        # Display the frames
        cv2.imshow('Video 1', frame1)
        cv2.imshow('Video 2', frame2)

        # Synchronize the videos
        if cv2.waitKey(int(1000 / fps)) & 0xFF == ord('q'):
            break

    # Release the video
    video1.release()
    video2.release()
    cv2.destroyAllWindows()
    
script_dir = os.path.dirname(os.path.abspath(__file__))
video1_path = os.path.join(script_dir, '../data/blenderSim/Video/CameraLNew.mp4')
video2_path = os.path.join(script_dir, '../data/blenderSim/Video/cameraREdited.mp4')
    
video1, video2 = load_videos(video1_path, video2_path)
synchronize_videos(video1, video2)