import cv2
import numpy as np
import os
import sys
from matplotlib import pyplot as plt
import open3d as o3d

from volleyball_detection import get_ball_xy
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from camera_config import load_camera_config

class StereoMatching:
    def __init__(self, left_img, right_img, config=None, displayImages=False):
        self.left_img = left_img
        self.right_img = right_img
        self.config = config if config else load_camera_config()
        self.X_ball = None
        self.Y_ball = None
        self.Z_ball = None
        
        if displayImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.left_img, 'gray')
            plt.subplot(122)
            plt.imshow(self.right_img, 'gray')
            plt.show()

    def try_detection_triangulation(self) -> bool:
        """Fast path: compute 3D from left/right detections only; return True if valid and set X/Y/Z."""
        leftImg = self.left_img
        rightImg = self.right_img

        x_left, y_left = get_ball_xy(leftImg)
        if x_left is None or y_left is None:
            return False
        x_right, y_right = get_ball_xy(rightImg)
        if x_right is None or y_right is None:
            return False

        # Focal scaling to actual image width
        focal_length_cfg = self.config['focal_length_px']
        cfg_width = self.config.get('resolution_width', self.left_img.shape[1])
        img_width = self.left_img.shape[1]
        focal_length = float(focal_length_cfg) * (float(img_width) / float(cfg_width))
        baseline = self.config['baseline_m']
        z_min = self.config.get('z_min_m', 15.0)
        z_max = self.config.get('z_max_m', 40.0)

        # Disparity calculation: Even though Blender cameras have Y-axis baseline,
        # the rendered stereo images show HORIZONTAL disparity (X-axis)
        # This is because cameras point at the scene from different Y positions,
        # creating horizontal parallax in the image plane.
        d = float(x_left - x_right)
        if d <= 0.5:
            return False
        Z = (focal_length * baseline) / (d + 1e-6)
        if not (z_min <= Z <= z_max):
            return False
        h, w = self.left_img.shape[:2]
        cx, cy = w // 2, h // 2
        X = (x_left - cx) * Z / focal_length
        Y = (y_left - cy) * Z / focal_length
        print(f"Ball coordinates in left image: {x_left}, {y_left}")
        print(f"Ball coordinates in 3D (det-based): {X}, {Y}, {Z}")
        self.X_ball, self.Y_ball, self.Z_ball = X, Y, Z
        return True

    def calculate_3d_ball_coordinates(self, disparity):
        """Compute 3D ball coordinates: detection-based disparity, then SGBM ROI, then NCC, then local high-res SGBM."""
        leftImg = self.left_img
        rightImg = self.right_img

        x_left, y_left = get_ball_xy(leftImg)
        if x_left is None or y_left is None:
            print("Ball not detected in the left image.")
            return
        print(f"Ball coordinates in left image: {x_left}, {y_left}")

        # Scale focal length to actual image width if different from configured resolution
        focal_length_cfg = self.config['focal_length_px']
        cfg_width = self.config.get('resolution_width', self.left_img.shape[1])
        img_width = self.left_img.shape[1]
        focal_length = float(focal_length_cfg) * (float(img_width) / float(cfg_width))
        baseline = self.config['baseline_m']
        z_min = self.config.get('z_min_m', 15.0)
        z_max = self.config.get('z_max_m', 40.0)

        # 1) Try detection-based disparity using right image
        x_right, y_right = get_ball_xy(rightImg)
        if x_right is not None and y_right is not None:
            # Use horizontal disparity (X-axis) - cameras create horizontal parallax
            d = float(x_left - x_right)
            if d > 0.5:
                Z = (focal_length * baseline) / (d + 1e-6)
                if z_min <= Z <= z_max:
                    h, w = disparity.shape
                    cx, cy = w // 2, h // 2
                    X = (x_left - cx) * Z / focal_length
                    Y = (y_left - cy) * Z / focal_length
                    print(f"Ball coordinates in 3D (det-based): {X}, {Y}, {Z}")
                    self.X_ball, self.Y_ball, self.Z_ball = X, Y, Z
                    return
                else:
                    print(f"Det-based Z {Z:.3f}m out of range [{z_min},{z_max}], falling back to SGBM.")
            else:
                print("Det-based disparity too small or negative, falling back to SGBM.")

        # 2) Fallback: progressively expand ROI around left detection and use median disparity
        h, w = disparity.shape
        disp_roi = None
        for window in (9, 13, 17, 21, 25, 31):
            half = window // 2
            y0 = max(0, y_left - half)
            y1 = min(h, y_left + half)
            x0 = max(0, x_left - half)
            x1 = min(w, x_left + half)
            roi = disparity[y0:y1, x0:x1]
            if roi.size == 0:
                continue
            valid = roi > 0
            if np.any(valid):
                disp_roi = float(np.median(roi[valid]))
                break

        # 3) NCC fallback if needed
        if disp_roi is None:
            left_gray = cv2.cvtColor(leftImg, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(rightImg, cv2.COLOR_BGR2GRAY)
            H, W = left_gray.shape
            patch_half = 8
            v_margin = 2
            disp_min = 1
            disp_max = min(64, max(2, x_left - patch_half - 1))
            best = None  # (corr, disparity)
            for dy in range(-v_margin, v_margin + 1):
                yc = y_left + dy
                y0 = yc - patch_half
                y1 = yc + patch_half + 1
                xL0 = x_left - patch_half
                xL1 = x_left + patch_half + 1
                if y0 < 0 or y1 > H or xL0 < 0 or xL1 > W:
                    continue
                left_patch = left_gray[y0:y1, xL0:xL1]
                xS0 = max(0, x_left - disp_max - patch_half)
                xS1 = min(W, x_left - disp_min + patch_half + 1)
                if xS1 - xS0 <= left_patch.shape[1]:
                    continue
                right_strip = right_gray[y0:y1, xS0:xS1]
                res = cv2.matchTemplate(right_strip, left_patch, cv2.TM_CCOEFF_NORMED)
                _, maxVal, _, maxLoc = cv2.minMaxLoc(res)
                k = maxLoc[0]
                if 1 <= k < res.shape[1] - 1:
                    s0, s1, s2 = float(res[0, k - 1]), float(res[0, k]), float(res[0, k + 1])
                    denom = (s0 - 2 * s1 + s2)
                    delta = 0.0
                    if abs(denom) > 1e-6:
                        delta = 0.5 * (s0 - s2) / denom
                        delta = float(np.clip(delta, -0.5, 0.5))
                else:
                    delta = 0.0
                x_right_center = (xS0 + k + delta) + patch_half
                d_ncc = float(x_left - x_right_center)
                if best is None or maxVal > best[0]:
                    best = (maxVal, d_ncc)
            if best is not None and best[1] > 0.3 and best[0] >= 0.5:
                disp_roi = best[1]
            else:
                # 4) Local high-res stereo fallback
                patch = 64
                scale = 3
                y0 = max(0, y_left - patch // 2)
                y1 = min(H, y_left + patch // 2)
                x0 = max(0, x_left - patch // 2)
                x1 = min(W, x_left + patch // 2)
                if y1 - y0 < 10 or x1 - x0 < 10:
                    print("No valid disparity values found around the detected point.")
                    return
                l_crop = left_gray[y0:y1, x0:x1]
                r_crop = right_gray[y0:y1, x0:x1]
                l_big = cv2.resize(l_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                r_big = cv2.resize(r_crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                win = 5
                blk = 5
                min_disp_lr = 0
                num_disp_lr = 64
                stereo_lr = cv2.StereoSGBM_create(
                    minDisparity=min_disp_lr,
                    numDisparities=num_disp_lr,
                    blockSize=blk,
                    P1=8 * 1 * win ** 2,
                    P2=32 * 1 * win ** 2,
                    disp12MaxDiff=1,
                    uniquenessRatio=5,
                    speckleWindowSize=50,
                    speckleRange=1,
                    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
                )
                disp_lr = stereo_lr.compute(l_big, r_big).astype(np.float32) / 16.0
                H2, W2 = disp_lr.shape
                cy2 = H2 // 2
                cx2 = W2 // 2
                m = 17
                mh = m // 2
                ry0 = max(0, cy2 - mh)
                ry1 = min(H2, cy2 + mh + 1)
                rx0 = max(0, cx2 - mh)
                rx1 = min(W2, cx2 + mh + 1)
                roi2 = disp_lr[ry0:ry1, rx0:rx1]
                valid2 = roi2 > 0
                if np.any(valid2):
                    disp_roi = float(np.median(roi2[valid2])) / scale
                else:
                    print("No valid disparity values found around the detected point.")
                    return

        # Convert disparity to depth and compute coordinates
        Z = (focal_length * baseline) / (disp_roi + 1e-6)
        if not (z_min <= Z <= z_max):
            print(f"Computed Z {Z:.3f}m out of range [{z_min},{z_max}], skipping.")
            self.X_ball = self.Y_ball = self.Z_ball = None
            return
        cx, cy = w // 2, h // 2
        X = (x_left - cx) * Z / focal_length
        Y = (y_left - cy) * Z / focal_length
        print(f"Ball coordinates in 3D (SGBM): {X}, {Y}, {Z}")
        self.X_ball, self.Y_ball, self.Z_ball = X, Y, Z
        return

    
    # trying other stereo matching algorithms
    def stereo_match_BM(self):
        nDisparitiesFactor = 1
        stereo = cv2.StereoBM.create(numDisparities = 16 * nDisparitiesFactor, blockSize = 15)
        disparity = stereo.compute(self.left_img, self.right_img)
        plt.imshow(disparity, 'gray')
        plt.show()



    # Function to read the images and perform stereo matching using SGBM algorithm 
    def stereo_match_SGBM(self, display = False):
        left_gray = cv2.cvtColor(self.left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(self.right_img, cv2.COLOR_BGR2GRAY)
        
        # left_gray = cv2.equalizeHist(left_gray)
        # right_gray = cv2.equalizeHist(right_gray)

        left_gray = cv2.GaussianBlur(left_gray, (5,5), 0)
        right_gray = cv2.GaussianBlur(right_gray, (5,5), 0)

        # #clahe
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # left_gray = clahe.apply(left_gray)
        # right_gray = clahe.apply(right_gray)
        
        ## STEREO MATCHING
        # Use configured parameters
        window_size = self.config['sgbm_window_size']
        block_size = self.config['sgbm_block_size']
        min_disp = self.config['sgbm_min_disp']
        nDispFactor = self.config['sgbm_num_disp_factor']
        num_disp = nDispFactor * 16 - min_disp

        print("working on disparity map...")

        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = block_size,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 0,
            uniquenessRatio = self.config['sgbm_uniqueness_ratio'],
            speckleWindowSize = self.config['sgbm_speckle_window_size'],
            preFilterCap = 63,
            speckleRange = self.config['sgbm_speckle_range'],
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )

        
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # disparity_cleaned = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)


        # # Set invalid disparities to 0
        # min_valid_disp = 10  # Minimum valid disparity value
        # disparity_cleaned[disparity_cleaned < min_valid_disp] = 0

        # print("disparity map done")

        # # Normalize and display the cleaned disparity map
        # disparity_normalized = cv2.normalize(disparity_cleaned, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        
        # Use configured WLS filter parameters
        sigma = self.config['wls_sigma']
        lmbda = self.config['wls_lambda']
        
        #create WSL filter
        left_matcher = stereo
        
        try:
            right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)
            left_disp = left_matcher.compute(left_gray, right_gray)
            right_disp = right_matcher.compute(right_gray, left_gray)
            #applying WSL filter
            wls_filter = cv2.ximgproc.createDisparityWLSFilter(left_matcher)
            wls_filter.setLambda(lmbda)
            wls_filter.setSigmaColor(sigma)
            filtered_disp = wls_filter.filter(left_disp, left_gray, disparity_map_right=right_disp) 
            filtered_disp = cv2.normalize(src=filtered_disp, dst=filtered_disp, beta=0, alpha=255, norm_type=cv2.NORM_MINMAX)
            filtered_disp = np.uint8(filtered_disp)
        except AttributeError:
            print("Warning: ximgproc not available. Using basic disparity map.")
            filtered_disp = cv2.normalize(disparity, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
        if display:
            plt.imshow(filtered_disp, 'gray')
            plt.colorbar()
            plt.show()

        return disparity, filtered_disp
    
    def disparity2depth(self, disparity, display = False):
        # Scale focal length to actual image width if different from configured resolution
        focal_length_cfg = self.config['focal_length_px']
        cfg_width = self.config.get('resolution_width', disparity.shape[1])
        img_width = disparity.shape[1]
        focal_length = float(focal_length_cfg) * (float(img_width) / float(cfg_width))
        baseline = self.config['baseline_m']
        z_min = self.config['z_min_m']
        z_max = self.config['z_max_m']

        print("working on depth map...")

        # convert disparity to depth
        depth = np.zeros(disparity.shape, dtype=np.float32)
        valid_pixels = disparity > 0
        depth[valid_pixels] = (focal_length * baseline) / (disparity[valid_pixels] + 1e-6)

        depth = np.clip(depth, z_min, z_max)

        # #median filter to remove the noise
        # depth = cv2.medianBlur(depth, 5)

        #bilaterla filter to smooth the depth map while preserving the edges
        depth = cv2.bilateralFilter(depth, 5, 50, 50)

        
        print("depth map done")

        #display depth map
        if display:
            plt.imshow(depth) 
            plt.colorbar()
            plt.show()

        return depth
    
    def depth2pointcloud(self, depth, display = False):
        # Use configured parameters
        focal_length = self.config['focal_length_px']
        z_max = self.config['z_max_m']
        h, w = depth.shape

        print("working on point cloud...")

        # Create a point cloud
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x  = (u - w/2) * depth / focal_length
        y  = (v - h/2) * depth / focal_length
        z = depth

        points = np.stack((x,y,z), axis=-1).reshape(-1, 3)

        #filter unwanted points
        valid_mask = (z>0 ) & (z<z_max)
        points = points[valid_mask.reshape(-1)]

        #create open3d point cloud object for the full scene
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # Create a point cloud for the ball
        ball_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.4)
        ball_sphere.translate((self.X_ball, self.Y_ball, self.Z_ball))
        ball_sphere.paint_uniform_color([1, 0, 0])  # Red color for the ball

        if display:
            # Visualize the point cloud
            o3d.visualization.draw_geometries([pcd, ball_sphere],
                window_name="Point Cloud",
                width=800,
                height=600,
                left=50,
                top=50,
                mesh_show_back_face=True)
        return pcd


        
 

if __name__ == "__main__":
    root = os.path.dirname(os.path.abspath(__file__))
    left_img = cv2.imread(os.path.join(root, "..\\output_frames\\left\\left3_0111.jpg"))
    right_img = cv2.imread(os.path.join(root, "..\\output_frames\\right\\right3_0111.jpg"))
    sm = StereoMatching(left_img,right_img, displayImages=False)
    raw_disp, disparity = sm.stereo_match_SGBM(display=False)
    sm.calculate_3d_ball_coordinates(raw_disp)
    depth = sm.disparity2depth(disparity, display=False)

    plt.figure(figsize=(12,6))

    #display Disarity map
    plt.subplot(1,2,1)
    plt.title("Disparity Map")
    plt.imshow(disparity)
    plt.colorbar(label="Disparity (px)")

    #display Depth map
    plt.subplot(1,2,2)
    plt.title("Depth Map")
    plt.imshow(depth)
    plt.colorbar(label="Depth (m)")
    

    plt.tight_layout()
    plt.show()


    pcd = sm.depth2pointcloud(depth, display=True)
    # sm.depth2pointcloud(depth)

    




    

    
    




    