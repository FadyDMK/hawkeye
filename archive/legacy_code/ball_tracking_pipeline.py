from volleyball_detection import get_ball_xy
from stereo_matching import StereoMatching
import cv2
import os


#Frames paths
root = os.path.dirname(os.path.abspath(__file__))
left_frames_dir = os.path.join(root, "..\\output_frames\\left")
right_frames_dir = os.path.join(root, "..\\output_frames\\right")

#output list for 3d ball positions
ball_positions = []

#looping through all the frames
for frame_num in range(0, 146):
    frame_id = f"{frame_num:04d}"
    left_path = os.path.join(left_frames_dir, f"left3_{frame_id}.jpg")
    right_path = os.path.join(right_frames_dir, f"right3_{frame_id}.jpg")
    
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        print(f"Frame {frame_id} not found in one of the directories.")
        continue
    # Read the images
    left_img = cv2.imread(left_path)
    right_img = cv2.imread(right_path)

    if left_img is None or right_img is None:
        print(f"Error reading images for frame {frame_id}.")
        continue

    #stereo matching
    sm = StereoMatching(left_img, right_img)
    raw_disp, disparity = sm.stereo_match_SGBM(display=False)
    sm.calculate_3d_ball_coordinates(raw_disp)

    #store ball positions if detection is successful
    if sm.X_ball is not None and sm.Y_ball is not None and sm.Z_ball is not None:
        ball_positions.append((frame_num, sm.X_ball, sm.Y_ball, sm.Z_ball))
    else:
        ball_positions.append((frame_num, None, None, None))    
        print(f"Ball not detected in frame {frame_id}.")


#export results to csv
with open("ball_positions.csv", "w") as f:
    f.write("Frame,X,Y,Z\n")
    for frame_num, x, y, z in ball_positions:
        f.write(f"{frame_num},{x},{y},{z}\n")
