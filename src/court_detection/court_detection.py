import os
import cv2
from matplotlib import cm
import numpy as np
import open3d as o3d
import pyvista as pv
import cv2
from scipy.__config__ import show
from roi import select_roi
from lines import detect_lines
from corners import compute_court_corners
from corners import manual_click_corners
from transforms import ball_camera_to_world
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#=== Constants ===

world_pts = np.array([[-3.17,-7.55],[-3.17,7.55],[3.17,-7.55],[3.17,7.55]], dtype=np.float32)
R = [[-4.37113883e-08  ,4.37113883e-08 , 1.00000000e+00],                                                                     
         [ 1.00000000e+00 , 1.91068568e-15 , 4.37113883e-08],                                                                      
    [ 0.00000000e+00 , 1.00000000e+00 ,-4.37113883e-08]]
t =  [25. , -1.5 , 5. ]  

court_length = 31.2
court_width = 15.1


# Define the pixel coordinates of the court corners in the reference image
ref_img_corners = np.array([(90.0, 116.0), (1818.0, 120.0), (92.0, 950.0), (1820.0, 952.0)], dtype=np.float32)

# Define the corresponding world coordinates (meters)
world_court_corners = np.array([
    [-7.55,  -15.6],  # Top-left
    [ -7.55,  15.6],  # Top-right
    [7.55, -15.6],  # Bottom-left
    [ 7.55, 15.6],  # Bottom-right
], dtype=np.float32)

DISPLAY_WIDTH, DISPLAY_HEIGHT = 960, 540

# === Main ===

def show_ball_top_down(ball_positions,overlay):
    for (X, Y, Z) in ball_positions:
        ball_pt = np.array([[X, Y, 1]], dtype=np.float32).T  # shape (3, 1)
        img_pt = H_world2img @ ball_pt
        img_pt = img_pt / img_pt[2]
        u, v = int(img_pt[0]), int(img_pt[1])
        intensity = int(np.clip((Z / 3.0) * 255, 0, 255))
        cv2.circle(overlay, (u, v), 20, (0, intensity, 255-intensity), -1)

    cv2.imshow("3D Ball Trajectory on Top-Down", cv2.resize(overlay, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_court_and_ball(ball_positions):
    # === Court parameters
    court_length = 31.2
    court_width = 15.1
    court_thickness = 0.7  # meters

    z0 = 0  # bottom of court
    z1 = court_thickness  # top of court

    court_verts = np.array([
        [-7.55, -15.6, z0],  # 0: Bottom-left-bottom
        [ 7.55, -15.6, z0],  # 1: Bottom-right-bottom
        [ 7.55,  15.6, z0],  # 2: Top-right-bottom
        [-7.55,  15.6, z0],  # 3: Top-left-bottom
        [-7.55, -15.6, z1],  # 4: Bottom-left-top
        [ 7.55, -15.6, z1],  # 5: Bottom-right-top
        [ 7.55,  15.6, z1],  # 6: Top-right-top
        [-7.55,  15.6, z1],  # 7: Top-left-top
    ])

    court_faces = [
        [4, 0, 1, 2, 3],  # bottom
        [4, 4, 5, 6, 7],  # top
        [4, 0, 1, 5, 4],  # front
        [4, 2, 3, 7, 6],  # back
        [4, 1, 2, 6, 5],  # right
        [4, 0, 3, 7, 4],  # left
    ]
    court_faces = np.hstack(court_faces)
    court = pv.PolyData(court_verts, faces=court_faces)

    # === Net parameters (unchanged)
    net_height = 3.32  # meters
    net_thickness = 0.05  # meters (visual only)
    net_width = 15.1  # meters (court width, sideline to sideline)
    net_verts = np.array([
        [-7.55, -net_thickness/2, 0],      # Bottom-left
        [ 7.55, -net_thickness/2, 0],      # Bottom-right
        [ 7.55,  net_thickness/2, 0],      # Top-right
        [-7.55,  net_thickness/2, 0],      # Top-left
        [-7.55, -net_thickness/2, net_height],  # Upper-bottom-left
        [ 7.55, -net_thickness/2, net_height],  # Upper-bottom-right
        [ 7.55,  net_thickness/2, net_height],  # Upper-top-right
        [-7.55,  net_thickness/2, net_height],  # Upper-top-left
    ])
    net_faces = [
        [4, 0, 1, 2, 3],  # bottom
        [4, 4, 5, 6, 7],  # top
        [4, 0, 1, 5, 4],  # front
        [4, 2, 3, 7, 6],  # back
        [4, 1, 2, 6, 5],  # right
        [4, 0, 3, 7, 4],  # left
    ]
    net_faces = np.hstack(net_faces)
    net = pv.PolyData(net_verts, faces=net_faces)

    BALL_SIZE = 0.72
    plotter = pv.Plotter()
    for (X, Y, Z) in ball_positions:
        sphere = pv.Sphere(center=(X, Y, Z), radius=0.3)
        plotter.add_mesh(sphere, color='red')
    plotter.add_mesh(court, color='green', opacity=1.0)
    plotter.add_mesh(net, color='black', opacity=0.7)
    # Optionally, add a lightgray plane for visual reference
    plane = pv.Plane(center=(0,0,0),
                    direction=(0,0,1),
                    i_size=court_width,
                    j_size=court_length,
                    i_resolution=1,
                    j_resolution=1)
    plotter.add_mesh(plane, color='lightgray', opacity=0.5)
    plotter.add_axes(
        line_width=4,
        labels_off=False,
        x_color='r', y_color='g', z_color='b'
    )
    plotter.show_grid()
    plotter.camera_position = [
        (25. , -1.5 , 5. ),
        (0. , 0. , 1. ), 
        (0. , 0. , 1. )
    ]
    plotter.show()



def load_and_crop_image(image_path):
    orig = cv2.imread(image_path)
    if orig is None:
        raise FileNotFoundError(f"Failed to load image at {image_path}")
    court_img, offset = select_roi(orig)
    return orig, court_img, offset

def draw_detected_lines(image, dirA, dirB):
    for x1, y1, x2, y2 in dirA:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    for x1, y1, x2, y2 in dirB:
        cv2.line(image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

# Set this flag to True if you want to use automatic line detection and corner computation
USE_LINE_DETECTION = False

# Define paths
root = os.path.dirname(os.path.abspath(__file__))
img_path = os.path.join(root, "..\\..\\", "output_frames", "left", "left3_0001.jpg")

# Load and crop image
orig, court_img, offset = load_and_crop_image(img_path)

x_off, y_off = offset
vis = court_img.copy()

# Conditional line detection and corner computation
if USE_LINE_DETECTION:
    # Detect and draw court lines
    dirA, dirB = detect_lines(court_img)
    draw_detected_lines(vis, dirA, dirB)
    
    # Compute intersections and corners
    img_pts = compute_court_corners(dirA, dirB)
    # Draw auto corners at reduced window size
    disp = cv2.resize(vis, (960, 540))
    for i, (x, y) in enumerate(img_pts):
        sx = int((x / vis.shape[1]) * 960)
        sy = int((y / vis.shape[0]) * 540)
        cv2.circle(disp, (sx, sy), 8, (0, 0, 255), -1)
        cv2.putText(disp, str(i), (sx+5, sy-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255),1)
    cv2.namedWindow("Corners", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Corners", DISPLAY_WIDTH, DISPLAY_HEIGHT)
    cv2.imshow("Corners", disp)
    print("Press 'a' to adjust")
    k = cv2.waitKey(0)
    cv2.destroyWindow("Corners")
    if k == ord('a'):
        img_pts = manual_click_corners(vis.copy())

else:
    # If not using automatic line detection, manually select corners
    img_pts = manual_click_corners(vis.copy())

# Adjust to original coords
img_pts_orig = img_pts + [x_off, y_off]

# Compute Homography
H, _ = cv2.findHomography(img_pts_orig, world_pts)
print("Final image points:", img_pts_orig)
print("Homography Matrix:\n", H)

# 6. Show final overlay
disp = orig.copy()
for (x, y) in img_pts_orig:
    cv2.circle(disp, (int(x), int(y)), 8, (0,255,255), -1)
cv2.imshow("Final Corners on Original", cv2.resize(disp, (DISPLAY_WIDTH, DISPLAY_HEIGHT)))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 7. Plot 3D ball world positions onto top-down court image
# Load your top-down reference court image
ref_path = os.path.join(root, "..\\..", "images", "court-top.png")
ref_img = cv2.imread(ref_path)
if ref_img is None:
    print(f"Failed to load top-down image at {ref_path}")
else:
    # Example: your ball positions in world coordinates (X, Y, Z in meters)
    # Replace with your actual 3D output list
    ball_camera = [(12.236497531256097, 3.3046337281863605, 21.615266282378208),(-7.841110260101454,2.5932838620648035,28.315188448892926),(3.3785905101156053,1.4408694822551844,22.96563775814709)]
    ball_world_3d = []
    
    for pos in ball_camera:
        ball_world_3d.append(ball_camera_to_world(pos, t, R))
    
    print("Ball positions (X, Y, Z):", ball_world_3d)

    # === Homography-based mapping ===
    # Compute homography from world to image
    H_world2img, _ = cv2.findHomography(world_court_corners, ref_img_corners)

    overlay = ref_img.copy()
    show_ball_top_down(ball_world_3d, overlay)
    show_court_and_ball(ball_world_3d)


    # === 8. Plot 3D ball world positions in 3D space
    # Court dimensions (adjust if needed)


