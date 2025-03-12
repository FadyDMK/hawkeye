import cv2
import numpy as np
import os

# Get the absolute path to the image
script_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(script_dir, '../data/frame2.png')



def detect_ball(image_path):
    # Read
    frame = cv2.imread(image_path)
    if frame is None:
        print("Error: Could not read the image.")
        exit()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)

    # Detect circles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=5, maxRadius=50)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            cv2.circle(frame, (i[0], i[1]), i[2], (0, 255, 0), 2)

    return frame
