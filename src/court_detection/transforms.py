import cv2
import numpy as np
def ball_camera_to_world(ball_pos, t, R):
    ball_pos = np.array(ball_pos).reshape(3, 1)
    t = np.array(t).reshape(3, 1)
    R = np.array(R)
    # OpenCV: X right, Y down, Z forward
    # Blender: X right, Y up, Z backward
    ball_pos_blender = np.array([ball_pos[0,0], -ball_pos[1,0], -ball_pos[2,0]]).reshape(3,1)
    world = (R @ ball_pos_blender) + t
    print("Camera to World:", ball_pos_blender.flatten(), "→", world.flatten())
    return world.flatten()