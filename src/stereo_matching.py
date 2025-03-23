import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import open3d as o3d

##Math formulas

## focal length = 26mm
## resolution = 1920x1080
## sensor width = 36 mm
## focal length in px = (focal length * resolution) / sensor width
## focal length in px = 1386.67 px 
## baseline = 3m
## Z_min = 15m
## Z_max = 40m

#max_disp = (baseline * focal_length) / Z_min = 554
#min_disp = (baseline * focal_length) / Z_max = 208 
#numDisparities = round_to_multiple_of_16(d_max - d_min) = 22

class StereoMatching:
    

    def __init__(self, displayImages):
        
        root = os.path.dirname(os.path.abspath(__file__))
        self.left_img = cv2.imread(os.path.join(root, "..\\output_frames\\left\\left3_0010.jpg"),cv2.IMREAD_GRAYSCALE)
        self.right_img = cv2.imread(os.path.join(root, "..\\output_frames\\right\\right3_0010.jpg"),cv2.IMREAD_GRAYSCALE)

        if displayImages:
            plt.figure()
            plt.subplot(121)
            plt.imshow(self.left_img)
            plt.subplot(122)
            plt.imshow(self.right_img)
            plt.show()


    def stereo_match_BM(self):
        nDisparitiesFactor = 1
        stereo = cv2.StereoBM.create(numDisparities = 16 * nDisparitiesFactor, blockSize = 15)
        disparity = stereo.compute(self.left_img, self.right_img)
        plt.imshow(disparity, 'gray')
        plt.show()



    # Function to read the images and perform stereo matching using SGBM algorithm 
    def stereo_match_SGBM(self, display = False):
        left_gray = self.left_img
        right_gray = self.right_img
        
        # left_gray = cv2.equalizeHist(left_gray)
        # right_gray = cv2.equalizeHist(right_gray)

        # left_gray = cv2.GaussianBlur(left_gray, (5,5), 0)
        # right_gray = cv2.GaussianBlur(right_gray, (5,5), 0)

        # #clahe
        # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        # left_gray = clahe.apply(left_gray)
        # right_gray = clahe.apply(right_gray)
        
        

        ## STEREO MATCHING
        #stereo matching settings
        window_size = 5
        block_size = 5
        min_disp = 0
        nDispFactor = 25
        num_disp = nDispFactor * 16 - min_disp

        print("working on disparity map...")

        stereo = cv2.StereoSGBM_create(minDisparity = min_disp,
            numDisparities = num_disp,
            blockSize = block_size,
            P1 = 8*3*window_size**2,
            P2 = 32*3*window_size**2,
            disp12MaxDiff = 1,
            uniquenessRatio = 15,
            speckleWindowSize = 50,
            preFilterCap = 63,
            speckleRange = 10,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
        )
        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        disparity_cleaned = cv2.morphologyEx(disparity, cv2.MORPH_OPEN, kernel)

        # Set invalid disparities to 0
        min_valid_disp = 10  # Minimum valid disparity value
        disparity_cleaned[disparity_cleaned < min_valid_disp] = 0

        print("disparity map done")

        # Normalize and display the cleaned disparity map
        disparity_normalized = cv2.normalize(disparity_cleaned, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        if display:
            plt.imshow(disparity_normalized, 'gray')
            plt.colorbar()
            plt.show()
        return disparity_normalized
    
    def disparity2depth(self, disparity, display = False):
        # Parameters
        focal_length = 1386.67
        baseline = 3.0

        print("working on depth map...")

        # convert disparity to depth
        depth = np.zeros(disparity.shape, dtype=np.float32)
        valid_pixels = disparity > 0
        depth[valid_pixels] = (focal_length * baseline) / disparity[valid_pixels]
        
        #median filter to remove the noise
        depth = cv2.medianBlur(depth, 5)

        #bilaterla filter to smooth the depth map while preserving the edges
        depth = cv2.bilateralFilter(depth, 5, 50, 50)

        
        print("depth map done")

        #display depth map
        depth = cv2.normalize(depth, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if display:
            plt.imshow(depth, cmap='jet')
            plt.colorbar()
            plt.show()


        return depth
    
    def depth2pointcloud(self, depth, display = False):
        # Parameters
        focal_length = 1386.67
        baseline = 3.0
        h, w = depth.shape

        print("working on point cloud...")

        # Create a point cloud
        u, v = np.meshgrid(np.arange(w), np.arange(h))
        x  = (u - w/2) * depth / focal_length
        y  = (v - h/2) * depth / focal_length
        z = depth

        #combine x y z to make the cloud map
        cloud = np.dstack((x, y, z))


        # Display the point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cloud.reshape(-1, 3))
        # Remove outliers
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        pcd = pcd.select_by_index(ind)

        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)
        mesh.compute_vertex_normals()
        mesh.paint_uniform_color([0.7, 0.7, 0.7])

        print("point cloud done")

        # Visualize the mesh
        o3d.visualization.draw_geometries([mesh])

        
        
 

if __name__ == "__main__":
    sm = StereoMatching(False)
    disparity = sm.stereo_match_SGBM(display=True)
    depth = sm.disparity2depth(disparity, display=True)
    # sm.depth2pointcloud(depth)

    

    
    




    