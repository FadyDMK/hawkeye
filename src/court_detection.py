import os
import cv2
import numpy as np
from sklearn.cluster import KMeans

# --- Helper Functions ---
def select_roi(image):
    r = cv2.selectROI("Select Court ROI", image, showCrosshair=True)
    cv2.destroyWindow("Select Court ROI")
    x, y, w, h = r
    if w == 0 or h == 0:
        return image, (0, 0)
    roi = image[y:y+h, x:x+w]
    return roi, (x, y)


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


def manual_click_corners(vis):
    pts = []
    labels = ["TL", "TR", "BL", "BR"]
    cv2.imshow("Adjust Corners", vis)
    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))
            cv2.circle(vis, (x, y), 8, (255, 0, 255), -1)
            cv2.putText(vis, labels[len(pts)-1], (x+5, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,255), 1)
            cv2.imshow("Adjust Corners", vis)
    cv2.setMouseCallback("Adjust Corners", on_click)
    print("Click points in order: TL, TR, BL, BR.")
    while len(pts) < 4:
        cv2.waitKey(1)
    cv2.destroyWindow("Adjust Corners")
    return np.array(pts, dtype=np.float32)

# === Main ===
root = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(root, "..", "output_frames", "left", "left3_0001.jpg")
orig = cv2.imread(img_path)

# 0. Crop to court region
court_img, offset = select_roi(orig)
x_off, y_off = offset
vis = court_img.copy()

# 1. Detect and draw lines
dirA, dirB = detect_lines(court_img)
for x1, y1, x2, y2 in dirA:
    cv2.line(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
for x1, y1, x2, y2 in dirB:
    cv2.line(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)

# 2. Compute intersections
intersections = []
for l1 in dirA:
    for l2 in dirB:
        pt = line_intersection(l1, l2)
        if pt:
            intersections.append(pt)
points = np.array([pt for pt in intersections if 0 <= pt[0] < court_img.shape[1] and 0 <= pt[1] < court_img.shape[0]])
print("Raw intersections:", len(points))

# 3. Two-stage clustering: X then Y
pts_uniq = np.unique(np.round(points).astype(int), axis=0)
two_kmeans_x = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(pts_uniq[:,0].reshape(-1,1))
x_centers = two_kmeans_x.cluster_centers_.flatten()
order_x = np.argsort(x_centers)
x_left, x_right = x_centers[order_x]
labels_x = two_kmeans_x.labels_
pts_left = pts_uniq[labels_x == order_x[0]]
if len(pts_left) < 2:
    print("Warning: not enough left intersections, switching to manual mode")
k2y_left = KMeans(n_clusters=2, random_state=0, n_init='auto').fit(pts_left[:,1].reshape(-1,1))
cy_left = sorted(k2y_left.cluster_centers_.flatten())
# Build corners
corners = [(x_left, cy_left[0]), (x_right, cy_left[0]), (x_left, cy_left[1]), (x_right, cy_left[1])]
img_pts = np.array(corners, dtype=np.float32)

# Draw auto corners
for i, (x, y) in enumerate(img_pts):
    cv2.circle(vis, (int(x), int(y)), 8, (0, 0, 255), -1)
    cv2.putText(vis, str(i), (int(x)+5, int(y)-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 1)

# 4. Ask user to adjust
cv2.imshow("Court Corners & Lines", cv2.resize(vis, (960,540)))
print("Press 'a' to adjust corner points manually, any other key to accept.")
key = cv2.waitKey(0)
cv2.destroyWindow("Court Corners & Lines")
if key == ord('a'):
    img_pts = manual_click_corners(vis.copy())

# 5. Compute homography
img_pts_orig = img_pts + np.array([x_off, y_off], dtype=np.float32)
world_pts = np.array([[-3.17,-7.55],[-3.17,7.55],[3.17,-7.55],[3.17,7.55]], dtype=np.float32)
H, _ = cv2.findHomography(img_pts_orig, world_pts)
print("Final image points:", img_pts_orig)
print("Homography Matrix:\n", H)

# 6. Show final overlay
disp = orig.copy()
for (x,y) in img_pts_orig:
    cv2.circle(disp, (int(x), int(y)), 8, (0,255,255), -1)
cv2.imshow("Final Corners on Original", cv2.resize(disp, (960,540)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Plot 3D ball world positions onto top-down court image
# Load your top-down reference court image
ref_path = os.path.join(root, "..", "images", "court-top.png")
ref_img = cv2.imread(ref_path)
if ref_img is None:
    print(f"Failed to load top-down image at {ref_path}")
else:
    # Example: your ball positions in world coordinates (X, Y, Z in meters)
    # Replace with your actual 3D output list
    ball_world_3d = [(-2.028302248428291,0.8173755329487142,20.98944685692581)]
    print("Ball positions (X, Y, Z):", ball_world_3d)

    # Map meter X,Y into reference image pixels
    h_ref, w_ref = ref_img.shape[:2]
    Sx = w_ref / 31.2  # pixels per meter in X
    Sy = h_ref / 15.1  # pixels per meter in Y
    overlay = ref_img.copy()
    for X, Y, Z in ball_world_3d:
        # Convert from court coords (centered at net midpoint) to image coords
        u = int((X + 15.6) * Sx)
        v = int(h_ref - (Y + 7.55) * Sy) # flip Y axis
        # Color-code by height Z (optional): higher balls brighter
        intensity = int(np.clip((Z / 3.0) * 255, 0, 255))
        cv2.circle(overlay, (u, v), 6, (0, intensity, 255-intensity), -1)

    cv2.imshow("3D Ball Trajectory on Top-Down", cv2.resize(overlay, (960, 540)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()
