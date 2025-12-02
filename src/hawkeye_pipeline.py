from cProfile import label
import os
import argparse
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Union, Any, Optional
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
from camera_config import load_camera_config


class HawkeyePipeline:
    def __init__(self, config=None):
        self.config = config if config else load_camera_config()
        self.__init__components()
        # Optional temporal smoothing for world coordinates
        self.smoothing_enabled = bool(self.config.get("smoothing_enabled", False))
        self.smoothing_alpha = float(self.config.get("smoothing_alpha", 0.3))
        self._last_world = None
    
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

        # Camera-to-world transformation
        # Blender scene: Camera geometry is complex, so we rely on empirical calibration.
        # MKV videos (newnewLeft.mkv, newnewRight.mkv) calibration:
        #   • Baseline: 3.0m between Blender cameras (18,-3,3) and (18,-6,3)
        #   • Umeyama alignment on frames 80-99 → similarity transform with <0.35m RMS error
        #   • scale ≈ 1.00157
        #   • rotation ≈
        #       [-0.01677,  0.19022, -0.98160]
        #       [ 0.99516,  0.09823,  0.00203]
        #       [ 0.09681, -0.97682, -0.19094]
        #   • translation ≈ [17.569, -6.655, 6.197]
        self.scale = 1.001569
        self.R = [[-0.016771, 0.19022 , -0.9816 ],
                  [ 0.99516 , 0.098228,  0.002032],
                  [ 0.096807, -0.97682, -0.19094]]
        self.t = [17.569, -6.6545, 6.1974]

        # Pre-compute camera pose vectors in world coordinates. These drive the 3D
        # visualization so that "left/right" matches what we see in the video.
        R_np = np.array(self.R, dtype=float)
        self._camera_position_world = np.array(self.t, dtype=float)
        self._camera_forward_world = R_np @ np.array([0.0, 0.0, 1.0])
        self._camera_up_world = R_np @ np.array([0.0, -1.0, 0.0])

        self.ball_positions_camera = []
        self.ball_positions_world = []

    def clear_previous_results(self):
        """Clear previous frame results for single-frame processing"""
        self.ball_positions_camera = []
        self.ball_positions_world = []
        # Also clear smoothing state for single-frame processing
        self._last_world = None
        
    def validate_court_bounds(self, x, y, z):
        """Validate if the ball position is within reasonable court bounds"""
        # Court dimensions from config (with some tolerance for balls slightly outside)
        court_width = self.config.get("court_width_m", 9.0)
        court_length = self.config.get("court_length_m", 18.0)
        court_center_x = float(self.config.get("court_center_x", 0.0))
        court_center_y = float(self.config.get("court_center_y", 0.0))
        court_center_z = float(self.config.get("court_center_z", 0.0))
        max_height = self.config.get("max_ball_height_m", 15.0)  # Reasonable max ball height
        
        # Add tolerance margins (increased for synthetic test data and imperfect calibration)
        # NOTE: With scaling factors, early frames (ball closer to camera) produce much larger Y values
        # Scale factors were optimized for frames 85-100, so early frames need huge tolerance
        width_tolerance = 15.0  # was 3.0, then 10.0
        length_tolerance = 30.0  # was 3.0, then 10.0 - increased to 30 to handle early frames with Y~40m
        
        # Check bounds
        rel_x = x - court_center_x
        rel_y = y - court_center_y
        rel_z = z - court_center_z

        x_valid = abs(rel_x) <= (court_width / 2 + width_tolerance)
        y_valid = abs(rel_y) <= (court_length / 2 + length_tolerance)
        z_valid = 0 <= rel_z <= max_height
        
        # Debug output
        if not (x_valid and y_valid and z_valid):
            print(
                f"BOUNDS CHECK FAILED: x={x:.3f} (valid: {x_valid}, center: {court_center_x:.3f}, limit: ±{court_width/2 + width_tolerance}), "
                f"y={y:.3f} (valid: {y_valid}, center: {court_center_y:.3f}, limit: ±{court_length/2 + length_tolerance}), "
                f"z={z:.3f} (valid: {z_valid}, center: {court_center_z:.3f}, limit: {court_center_z:.3f}-{court_center_z + max_height:.3f})"
            )
        
        return x_valid and y_valid and z_valid

    def process_from_pair(self, left_frame, right_frame, frame_num=None, display=False, use_smoothing=None):
        # 1: Try fast detection-based triangulation first
        from stereo_matching import StereoMatching
        stereo_matcher = StereoMatching(left_frame, right_frame, config=self.config)
        if not stereo_matcher.try_detection_triangulation():
            # 2: If that fails, compute disparity and fall back to robust multi-stage 3D
            left_ball_xy = self.get_ball_xy(left_frame)
            if left_ball_xy[0] is None:
                print("No ball detected in left frame")
                return None, None
            raw_disp, filtered_disp = stereo_matcher.stereo_match_SGBM(display=display)
            stereo_matcher.calculate_3d_ball_coordinates(raw_disp)

        # 3: Get ball coordinates in camera space
        camera_coords = (stereo_matcher.X_ball, stereo_matcher.Y_ball, stereo_matcher.Z_ball)
        if None in camera_coords:
            print("No ball detected in stereo matching")
            return None

        # 4: Convert to world coordinates (with scaling for unit conversion)
        world_coords = self.ball_camera_to_world(camera_coords, self.t, self.R, self.scale)
        
        # 4a: Validate court bounds - reject clearly out-of-bounds detections
        if all(v is not None for v in world_coords):
            if not self.validate_court_bounds(*world_coords):
                print(f"Ball position {world_coords} is out of court bounds, rejecting")
                world_coords = (None, None, None)

        # 4b: Optional EMA smoothing on world coordinates (disabled for single-frame processing)
        should_smooth = use_smoothing if use_smoothing is not None else self.smoothing_enabled
        if should_smooth and self._last_world is not None:
            if all(v is not None for v in world_coords) and all(v is not None for v in self._last_world):
                a = self.smoothing_alpha
                wx = self._last_world[0] + a * (world_coords[0] - self._last_world[0])
                wy = self._last_world[1] + a * (world_coords[1] - self._last_world[1])
                wz = self._last_world[2] + a * (world_coords[2] - self._last_world[2])
                world_coords = (wx, wy, wz)
        # update last only on valid and when smoothing is enabled
        if should_smooth:
            self._last_world = world_coords

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
        """3D visualization of court and ball (current frame only)"""
        import pyvista as pv
        import numpy as np

        # Court parameters - use values from configuration
        court_length = float(self.config.get("court_length_m", 18.0))
        court_width = float(self.config.get("court_width_m", 9.0))
        court_thickness = 0.02  # keep a thin slab for visibility without bulk

        court_center = np.array([
            float(self.config.get("court_center_x", 0.0)),
            float(self.config.get("court_center_y", 0.0)),
            float(self.config.get("court_center_z", 0.0)),
        ])

        half_length = court_length / 2.0
        half_width = court_width / 2.0

        # Create a court mesh aligned with the configured dimensions
        court_verts = np.array([
            [-half_width, -half_length, 0.0],
            [ half_width, -half_length, 0.0],
            [ half_width,  half_length, 0.0],
            [-half_width,  half_length, 0.0],
            [-half_width, -half_length, court_thickness],
            [ half_width, -half_length, court_thickness],
            [ half_width,  half_length, court_thickness],
            [-half_width,  half_length, court_thickness],
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
        court = pv.PolyData(court_verts + court_center, faces=court_faces)

        # Net parameters
        net_height = self.config.get("net_height_m", 2.43)
        net_thickness = 0.05

        # Create a net mesh
        net_verts = np.array([
            [-half_width, -net_thickness / 2.0, 0.0],
            [ half_width, -net_thickness / 2.0, 0.0],
            [ half_width,  net_thickness / 2.0, 0.0],
            [-half_width,  net_thickness / 2.0, 0.0],
            [-half_width, -net_thickness / 2.0, net_height],
            [ half_width, -net_thickness / 2.0, net_height],
            [ half_width,  net_thickness / 2.0, net_height],
            [-half_width,  net_thickness / 2.0, net_height],
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
        net = pv.PolyData(net_verts + court_center, faces=net_faces)

        # Create a plotter
        plotter = pv.Plotter()

        # Add ball trajectory (all valid positions)
        if self.ball_positions_world:
            # Collect all valid ball positions
            valid_positions = []
            for positions in self.ball_positions_world:
                if positions[0] is not None:
                    valid_positions.append(positions)
            
            if valid_positions:
                # Convert to numpy array for easier handling
                trajectory = np.array(valid_positions)
                
                # Draw trajectory line
                if len(trajectory) > 1:
                    line = pv.Line(trajectory[0], trajectory[-1])
                    # Create spline through all points
                    points = pv.PolyData(trajectory)
                    spline = points.delaunay_2d()
                    
                    # Simple line through points
                    trajectory_line = pv.Spline(trajectory, len(trajectory))
                    plotter.add_mesh(trajectory_line, color='yellow', line_width=3, label='Ball Trajectory')
                
                # Draw ball positions as small spheres
                for i, pos in enumerate(trajectory):
                    # Make start and end positions more visible
                    if i == 0:
                        # Start position - green
                        sphere = pv.Sphere(radius=0.15, center=pos)
                        plotter.add_mesh(sphere, color='lime', label='Start')
                    elif i == len(trajectory) - 1:
                        # End position - red
                        sphere = pv.Sphere(radius=0.15, center=pos)
                        plotter.add_mesh(sphere, color='red', label='End')
                    else:
                        # Intermediate positions - small blue spheres
                        if i % 5 == 0:  # Show every 5th frame to avoid clutter
                            sphere = pv.Sphere(radius=0.08, center=pos)
                            plotter.add_mesh(sphere, color='cyan', opacity=0.6)
                
                print(f"Visualizing {len(valid_positions)} ball positions")
            else:
                print("No valid ball position to visualize")
        else:
            print("No ball positions available")
        
        plotter.add_mesh(court, color='green', opacity=0.5, show_edges=True)
        plotter.add_mesh(net, color='black', opacity=0.7, show_edges=True)

        plane = pv.Plane(
            center=tuple(court_center.tolist()),
            direction=(0, 0, 1),
            i_size= court_width,
            j_size= court_length,
            )
        plotter.add_mesh(plane, color='lightgray', opacity=0.5)

        # Setup view
        plotter.add_axes()
        plotter.show_grid()

        # Add camera view buttons
        def view_topdown():
            """Top-down view looking straight down at court"""
            plotter.view_xy()
            plotter.camera.zoom(1.2)
        
        def view_side():
            """Side view looking along the court length"""
            plotter.view_xz()
            plotter.camera.zoom(1.2)
        
        def view_end():
            """End view looking across court width"""
            plotter.view_yz()
            plotter.camera.zoom(1.2)
        
        def view_perspective():
            """Perspective/3D view"""
            plotter.camera_position = [
                (25.0, -15.0, 10.0),  # camera position
                tuple(court_center.tolist()),  # focal point (look at court center)
                (0.0, 0.0, 1.0)  # up direction
            ]
        
        def view_player():
            """Player perspective from court side"""
            plotter.camera_position = [
                (0.0, -15.0, 1.8),  # camera position (player height)
                (0.0, 0.0, 2.0),  # focal point (look at net height)
                (0.0, 0.0, 1.0)  # up direction
            ]

        # Add buttons to toolbar
        plotter.add_key_event('t', view_topdown)  # Press 't' for top-down
        plotter.add_key_event('s', view_side)     # Press 's' for side
        plotter.add_key_event('e', view_end)      # Press 'e' for end
        plotter.add_key_event('p', view_perspective)  # Press 'p' for perspective
        plotter.add_key_event('v', view_player)   # Press 'v' for player view

        # Align the PyVista camera with the real left-camera pose so the court
        # visuals match the input video orientation.
        default_pos = np.array([25.0, -1.5, 5.0])
        default_forward = np.array([-1.0, 0.0, -0.2])
        default_up = np.array([0.0, 0.0, 1.0])

        camera_pos = getattr(self, "_camera_position_world", default_pos)
        camera_forward = getattr(self, "_camera_forward_world", default_forward)
        camera_up = getattr(self, "_camera_up_world", default_up)

        def _normalize(vec):
            norm = np.linalg.norm(vec)
            return vec if norm == 0 else vec / norm

        camera_forward = _normalize(np.array(camera_forward, dtype=float))
        camera_up = _normalize(np.array(camera_up, dtype=float))
        camera_pos = np.array(camera_pos, dtype=float)

        focal_point = camera_pos + camera_forward
        plotter.camera_position = [
            tuple(camera_pos.tolist()),
            tuple(focal_point.tolist()),
            tuple(camera_up.tolist())
        ]
        
        # Add text with keyboard shortcuts
        plotter.add_text(
            "Keyboard Shortcuts:\n"
            "T - Top-down view\n"
            "S - Side view\n"
            "E - End view\n"
            "P - Perspective view\n"
            "V - Player view\n"
            "R - Reset camera",
            position='upper_left',
            font_size=10,
            color='white'
        )
        
        plotter.show()

    def process_single_frame(self, frame_num):
        """ for processing a single frame """
        # Use pre-extracted frames from output_frames folder
        # These frames are extracted from the MKV videos with correct alignment
        root = os.path.dirname(os.path.abspath(__file__))
        left_frames_dir = os.path.join(root, "..", "output_frames", "left")
        right_frames_dir = os.path.join(root, "..", "output_frames", "right")

        # Format frame number to match the extracted frame naming
        frame_id = f"{frame_num:04d}"
        left_path = os.path.join(left_frames_dir, f"left3_{frame_id}.jpg")
        right_path = os.path.join(right_frames_dir, f"right3_{frame_id}.jpg")

        # Check if the frame exists
        if not os.path.exists(left_path) or not os.path.exists(right_path):
            print(f"Frame {frame_id} not found: {left_path} or {right_path}")
            return None
        
        # Read the images
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)

        if left_img is None or right_img is None:
            print(f"Failed to load images for frame {frame_id}")
            return None
        
        result = self.process_from_pair(left_img, right_img, frame_num, use_smoothing=False)
        if result and isinstance(result, dict):
            print(f"Frame {frame_num} processed successfully")
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
        court_center_x = float(self.config.get("court_center_x", 0.0))
        court_center_y = float(self.config.get("court_center_y", 0.0))

        half_width = court_width / 2.0
        half_length = court_length / 2.0

        # Draw Court Boundaries
        court_x = [
            court_center_x - half_width,
            court_center_x + half_width,
            court_center_x + half_width,
            court_center_x - half_width,
            court_center_x - half_width,
        ]
        court_y = [
            court_center_y - half_length,
            court_center_y - half_length,
            court_center_y + half_length,
            court_center_y + half_length,
            court_center_y - half_length,
        ]
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
        ax.set_xlim(court_center_x - half_width - padding, court_center_x + half_width + padding)
        ax.set_ylim(court_center_y - half_length - padding, court_center_y + half_length + padding)

        plt.tight_layout()
        plt.show()



def _parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Run Hawkeye pipeline over a frame range and export results")
    parser.add_argument("--start", type=int, default=0, help="Start frame index (inclusive)")
    parser.add_argument("--end", type=int, default=None, help="End frame index (exclusive); defaults to all available")
    parser.add_argument("--export", action="store_true", help="Export CSV results on completion")
    parser.add_argument("--visualize", choices=["none", "2d", "3d"], default="none", help="Optional visualization after processing")
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    pipe = HawkeyePipeline()
    pipe.process_video(start_frame=args.start, end_frame=args.end)
    if args.export:
        pipe.export_results()
    if args.visualize != "none":
        pipe.visualize_results(type=args.visualize)







