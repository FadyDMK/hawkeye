import cv2
import numpy as np
 
def detect_lines(img,
                 canny_thresh1=50, canny_thresh2=150,
                 hough_thresh=100, min_line_len=80, max_line_gap=30):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, canny_thresh1, canny_thresh2, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi/180,
        threshold=hough_thresh,
        minLineLength=min_line_len,
        maxLineGap=max_line_gap
    )
    if lines is None:
        return [], []

    angles = []
    for x1, y1, x2, y2 in lines[:, 0]:
        angle = np.arctan2(y2 - y1, x2 - x1)
        if angle < 0:
            angle += np.pi
        angles.append(angle)

    data = np.vstack([np.cos(2*np.array(angles)), np.sin(2*np.array(angles))]).T
    kmeans = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(data)
    labels = kmeans.labels_

    dirA = [lines[i, 0] for i in range(len(lines)) if labels[i] == 0]
    dirB = [lines[i, 0] for i in range(len(lines)) if labels[i] == 1]
    return dirA, dirB


def line_intersection(l1, l2):
    x1, y1, x2, y2 = l1
    x3, y3, x4, y4 = l2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-5:
        return None
    px = ((x1*y2-y1*x2)*(x3-x4) - (x1-x2)*(x3*y4-y3*x4)) / denom
    py = ((x1*y2-y1*x2)*(y3-y4) - (y1-y2)*(x3*y4-y3*x4)) / denom
    return (px, py)