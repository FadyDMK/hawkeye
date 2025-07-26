import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import open3d as o3d

from volleyball_detection import get_ball_xy
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

    def calculate_3d_ball_coordinates(self, disparity):
        ## GETTING BALL 3D COORDINATES
        leftImg = self.left_img

        (x_center, y_center) = get_ball_xy(leftImg)
        if x_center is None or y_center is None:
            print("Ball not detected in the image.")
            return
        print(f"Ball coordinates in left image: {x_center}, {y_center}")
        disparity_value =  disparity[y_center, x_center]
        window = 5
        half = window // 2
        roi = disparity[y_center-half:y_center+half, x_center-half:x_center+half]
        if roi.size == 0 or np.all(roi <= 0):
            print("No valid disparity values in the region of interest.")
            return
        disparity_value = np.mean(roi[roi > 0]) 

        # Use configured parameters
        focal_length = self.config['focal_length_px']
        baseline = self.config['baseline_m']
        Z = (focal_length * baseline) / (disparity_value + 1e-6)
        
        h,w =  disparity.shape
        cx, cy = w//2, h//2 

        X = (x_center - cx) * Z / focal_length
        Y = (y_center - cy) * Z / focal_length
        print(f"Ball coordinates in 3D space: {X}, {Y}, {Z}")
        self.X_ball = X
        self.Y_ball = Y
        self.Z_ball = Z
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
        # Use configured parameters
        focal_length = self.config['focal_length_px']
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

    




    

    
    




    