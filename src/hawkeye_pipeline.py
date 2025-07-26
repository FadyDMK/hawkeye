from cProfile import label
import os
from turtle import color
import cv2
import numpy as py
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any, Optional
from camera_config import load_camera_config


class HawkeyePipeline:
    def __init__(self, config=None):
        self.config = config if config else load_camera_config()
        self.__init__components()
    
    def __init__components(self):
        from volleyball_detection import get_ball_xy
        from stereo_matching import StereoMatching
        import sys  # Make sure sys is imported
        
        # Fix the path append by joining the paths first
        court_detection_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "court_detection")
        sys.path.append(court_detection_path)
        
        from transforms import ball_camera_to_world

        self.get_ball_xy = get_ball_xy
        self.ball_camera_to_world = ball_camera_to_world

        self.R = [[-4.37113883e-08, 4.37113883e-08, 1.00000000e+00],
                 [1.00000000e+00, 1.91068568e-15, 4.37113883e-08],
                 [0.00000000e+00, 1.00000000e+00, -4.37113883e-08]]
        self.t = [25.0, -1.5, 5.0]

        self.ball_positions_camera = []
        self.ball_positions_world = []

    def process_from_pair(self, left_frame, right_frame, frame_num=None, display = False):
        # 1: Detect ball in left frame
        left_ball_xy = self.get_ball_xy(left_frame)
        if left_ball_xy[0] is None:
            print("No ball detected in left frame")
            return None, None
        
        # 2: Stereo matching
        from stereo_matching import StereoMatching
        stereo_matcher = StereoMatching(left_frame, right_frame)
        raw_disp, filtered_disp = stereo_matcher.stereo_match_SGBM(display=display)
        stereo_matcher.calculate_3d_ball_coordinates(raw_disp)

        # 3: Get ball coordinates in camera space
        camera_coords = (stereo_matcher.X_ball, stereo_matcher.Y_ball, stereo_matcher.Z_ball)
        if None in camera_coords:
            print("No ball detected in stereo matching")
            return None

        # 4: Convert to world coordinates
        world_coords = self.ball_camera_to_world(camera_coords, self.t, self.R)

        # 5: Store results
        if frame_num is not None:
            while len(self.ball_positions_camera) <= frame_num:
                self.ball_positions_camera.append((None, None, None))
            while len(self.ball_positions_world) <= frame_num:
                self.ball_positions_world.append((None, None, None))
            
            self.ball_positions_camera[frame_num] = camera_coords
            self.ball_positions_world[frame_num] = world_coords

        return {
            'frame_num': frame_num,
            'camera_coords': camera_coords,
            'world_coords': world_coords,
        }
    
    def process_video(self, start_frame = 0, end_frame = None):
        #paths - use default paths since they're not in camera config
        root = os.path.dirname(os.path.abspath(__file__))
        left_frames_dir = os.path.join(root, "..", "output_frames", "left")
        right_frames_dir = os.path.join(root, "..", "output_frames", "right")

        if end_frame is None:
            import glob
            left_files = glob.glob(os.path.join(left_frames_dir, "left3_*.jpg"))
            end_frame = len(left_files)

        for frame_num in range(start_frame, end_frame):
            frame_id = f"{frame_num:04d}"
            left_path = os.path.join(left_frames_dir, f"left3_{frame_id}.jpg")
            right_path = os.path.join(right_frames_dir, f"right3_{frame_id}.jpg")

            if not os.path.exists(left_path) or not os.path.exists(right_path):
                print(f"Frame {frame_num} not found: {left_path} or {right_path}")
                continue
            left_img = cv2.imread(left_path)
            right_img = cv2.imread(right_path)
            if left_img is None or right_img is None:
                print(f"Failed to load images for frame {frame_num}")
                continue

            result = self.process_from_pair(left_img, right_img, frame_num)
            if result and isinstance(result, dict):  # Check if result is a dictionary
                print(f"Frame {frame_num}: Camera coords: {result['camera_coords']}, World coords: {result['world_coords']}")
            else:
                print(f"Frame {frame_num}: No valid result")
    def export_results(self, output_path = None):
        """ Export ball position results to a CSV file. """
        if output_path is None:
            root = os.path.dirname(os.path.abspath(__file__))
            output_path = os.path.join(root, "..", "output")
            os.makedirs(output_path, exist_ok=True)
        
        #   Export camera coordinates
        camera_path = os.path.join(output_path, "ball_positions_camera.csv")
        with open(camera_path, 'w') as f:
            f.write("Frame,X,Y,Z\n")
            for frame_num, (x,y,z) in enumerate(self.ball_positions_camera):
                f.write(f"{frame_num},{x},{y},{z}\n")
        
        #   Export world coordinates
        world_path = os.path.join(output_path, "ball_positions_world.csv")
        with open(world_path, 'w') as f:
            f.write("Frame,X,Y,Z\n")
            for frame_num, (x,y,z) in enumerate(self.ball_positions_world):
                f.write(f"{frame_num},{x},{y},{z}\n")
        
        print(f"Results exported to {output_path}")

    def visualize_results(self, type="3d"):
        """ Visualize the ball positions in 3D or 2D. """
        if type == "3d":
            self._visualize_3d()
        elif type == "2d":
            self._visualize_2d_topdown()
        else:
            print("Invalid visualization type. Use '3d' or '2d'.")
    def _visualize_3d(self):
        """3D visualization of court and ball"""
        import pyvista as pv
        import numpy as np

        # Court parameters - use values from configuration
        court_length = self.config.get("court_length_m", 18.0)
        court_width = self.config.get("court_width_m", 9.0)
        court_thickness = 0.7 # in meters

        # Create a court mesh
        court_verts =  court_verts = np.array([
            [-7.55, -15.6, 0],    # Bottom corners
            [ 7.55, -15.6, 0],
            [ 7.55,  15.6, 0],
            [-7.55,  15.6, 0],
            [-7.55, -15.6, court_thickness],  # Top corners
            [ 7.55, -15.6, court_thickness],
            [ 7.55,  15.6, court_thickness],
            [-7.55,  15.6, court_thickness],
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

        # Net parameters
        net_height = self.config.get("net_height_m", 2.43)
        net_thickness = 0.05

        # Create a net mesh
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

        # Create a plotter
        plotter = pv.Plotter()

        # Add ball positions
        for positions in self.ball_positions_world:
            if positions[0] is not None:
                sphere = pv.Sphere(radius=0.3, center=positions)
                plotter.add_mesh(sphere, color='red', show_edges=True)
        
        plotter.add_mesh(court, color='green', opacity=0.5, show_edges=True)
        plotter.add_mesh(net, color='black', opacity=0.7, show_edges=True)

        plane = pv.Plane(
            center=(0, 0, 0),
            direction=(0, 0, 1),
            i_size= court_width,
            j_size= court_length,
            )
        plotter.add_mesh(plane, color='lightgray', opacity=0.5)

        # Setup view
        plotter.add_axes()
        plotter.show_grid()
        plotter.camera_position = [
            (25.0, -1.5, 5.0),
            (0.0, 0.0, 1.0),
            (0.0, 0.0, 1.0)
        ]
        plotter.show()

    def process_single_frame(self, frame_num):
        """ for processing a single frame """
        #paths - use default paths since they're not in camera config
        root = os.path.dirname(os.path.abspath(__file__))
        left_frames_dir = os.path.join(root, "..", "output_frames", "left")
        right_frames_dir = os.path.join(root, "..", "output_frames", "right")

        #   Format frame number
        frame_id = f"{frame_num:04d}"
        left_path = os.path.join(left_frames_dir, f"left3_{frame_id}.jpg")
        right_path = os.path.join(right_frames_dir, f"right3_{frame_id}.jpg")

        #   Check if the frame exists
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print(f"Frame {frame_id} not found: {left_path} or {right_path}")
            return None
        
        #   Read the images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            print(f"Failed to load images for frame {frame_id}")
            return None
        
        result = self.process_from_pair(left_img, right_img, frame_num)
        if result and isinstance(result, dict):
            print(f"frrame {frame_num} processed successfully")
            print(f"Camera coords: {result['camera_coords']},\n World coords: {result['world_coords']}")
            return result
        else:
            print(f"Frame {frame_num}: No valid result")
            return None



    def _visualize_2d_topdown(self):
        """2D top-down visualization of court and ball"""
        import numpy as np

        # Court parameterscreate figure and axis
        fig, ax = plt.subplots(figsize=(10, 8))

        # Court dimensions
        court_length = self.config.get("court_length_m", 18.0)
        court_width = self.config.get("court_width_m", 9.0)

        # Draw Court Boundaries
        court_x = [-court_width/2, court_width/2, court_width/2, -court_width/2, -court_width/2]
        court_y = [-court_length/2, -court_length/2, court_length/2, court_length/2, -court_length/2]
        ax.plot(court_x, court_y, 'k-', color='green', linewidth=2)


        #   extract ball positions
        x_coords = []
        y_coords = []
        frame_nums = []

        for i, (x,y,z) in enumerate(self.ball_positions_world):
            if x is not None and y is not None:
                x_coords.append(x)
                y_coords.append(y)
                frame_nums.append(i)
        
        if not x_coords:
            ax.text(0, 0, "No ball detected", fontsize=12, ha='center', va='center', color='red')
        else:
            ax.plot(x_coords, y_coords, 'b-', alpha=0.5, linewidth = 1)

            # plot individual ball positions
            scatter = ax.scatter(x_coords, y_coords, c=frame_nums, cmap='viridis', s=50, alpha=0.8, edgecolors='k')

            # Add colorbar
            cbar = plt.colorbar(scatter, ax=ax, label='Frame Number')

            # Mark start and end points
            ax.plot(x_coords[0], y_coords[0], 'ro', markersize=8, label='Start')
            ax.plot(x_coords[-1], y_coords[-1], 'go', markersize=8, label='End')
        
        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

        # Set labels and title
        ax.set_xlabel('X Coordinate (m)')
        ax.set_ylabel('Y Coordinate (m)')
        ax.set_title('2D Top-Down View of Ball Positions')

        # Show grid and legend
        ax.grid(True, alpha = 0.3)
        ax.legend()

        # Set limits with some padding
        padding = max(court_length, court_width) * 0.1
        ax.set_xlim(-court_width/2 - padding, court_width/2 + padding)
        ax.set_ylim(-court_length/2 - padding, court_length/2 + padding)

        plt.tight_layout()
        plt.show()



        

                

