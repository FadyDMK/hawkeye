import cv2
import numpy as np
from sklearn.cluster import KMeans
from lines import line_intersection


def manual_click_corners(vis):
    pts = []
    labels = ["TL", "TR", "BL", "BR"]
    cv2.namedWindow("Adjust Corners", cv2.WINDOW_NORMAL)
    disp = cv2.resize(vis, (800, 450))
    cv2.imshow("Adjust Corners", disp)
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            # map click back to original vis size
            scale_x = vis.shape[1] / disp.shape[1]
            scale_y = vis.shape[0] / disp.shape[0]
            xo = int(x * scale_x)
            yo = int(y * scale_y)
            pts.append((xo, yo))
            cv2.circle(vis, (xo, yo), 8, (255, 0, 255), -1)
            cv2.putText(vis, labels[len(pts)-1], (xo+5, yo-5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            disp2 = cv2.resize(vis, (800, 450))
            cv2.imshow("Adjust Corners", disp2)
    cv2.setMouseCallback("Adjust Corners", on_click)
    print("Click points in order: TL, TR, BL, BR.")
    while len(pts) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Adjust Corners")
    return np.array(pts, dtype=np.float32)

def compute_court_corners(dirA, dirB):
    pts = []
    for l1 in dirA:
        for l2 in dirB:
            pt = line_intersection(l1, l2)
            if pt: pts.append(pt)
    points = np.unique(np.round(pts).astype(int), axis=0)
    print("Intersections:", len(points))
    # Two-stage clustering (X then Y) simplified
    kmx = KMeans(2, random_state=0).fit(points[:,0].reshape(-1,1))
    labels_x = kmx.labels_
    xs = kmx.cluster_centers_.flatten()
    left_idx, right_idx = np.argsort(xs)
    xl, xr = xs[left_idx], xs[right_idx]
    left_pts = points[labels_x == left_idx]
    kmy = KMeans(2, random_state=0).fit(left_pts[:,1].reshape(-1,1))
    ys = sorted(kmy.cluster_centers_.flatten())
    corners = [(xl, ys[0]), (xr, ys[0]), (xl, ys[1]), (xr, ys[1])]
    img_pts = np.array(corners, dtype=np.float32)
    return img_pts;