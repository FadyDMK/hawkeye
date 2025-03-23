import camera_calibrate
import os

def main():
    # Get the absolute path to the image
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, '../data/blenderSim/ExtractedFrames/frameL2new.png')
    cc = camera_calibrate.CameraCalibrate(image_path)


if __name__ == '__main__':
    main()

# import pycolmap

# # Set paths
# image_dir = "output_frames"
# database_path = "db/database.db"
# output_path = "output"

# # Initialize COLMAP database
# pycolmap.extract_features(database_path, image_dir, camera_mode=1)  # Extracts SIFT features
# pycolmap.match_exhaustive(database_path)  # Matches features

# # Run Structure from Motion (SfM)
# sfm_output = pycolmap.incremental_mapping(database_path, image_dir, output_path)

# # Convert model to PLY format
# sfm_output.export_PLY(output_path + "/point_cloud.ply")
