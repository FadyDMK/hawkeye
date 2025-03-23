import cv2
import numpy as np
import os

def detect_ball(frame):
    """Detects a ball with white, yellow, and blue colors."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define color ranges
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])

    lower_blue = np.array([100, 100, 100])
    upper_blue = np.array([120, 255, 255])

    lower_white = np.array([0, 0, 200]) # Low saturation, high value
    upper_white = np.array([180, 50, 255])

    # Create masks
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    # Combine masks
    mask = cv2.bitwise_or(mask_yellow, mask_blue)
    mask = cv2.bitwise_or(mask, mask_white)

    # Morphological operations (optional)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Identify the ball
    if contours:
        ball_contour = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(ball_contour)

        # Draw circle (optional)
        if radius > 10:
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            return (int(x), int(y))
    return None

root = os.path.dirname(os.path.abspath(__file__))
left_img = cv2.imread(os.path.join(root, "..\\output_frames\\left\\left2_0080.jpg"))
right_img = cv2.imread(os.path.join(root, "..\\output_frames\\right\\right2_0080.jpg"))

ball_center = detect_ball(left_img)
cv2.imshow("Ball Detection", ball_center)
cv2.waitKey(0)
cv2.destroyAllWindows()