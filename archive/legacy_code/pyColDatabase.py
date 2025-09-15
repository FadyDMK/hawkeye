import pycolmap
import os

# Paths
root = os.path.dirname(os.path.abspath(__file__))

image_folder = os.path.join(root, "..\\output_frames")
database_path = os.path.join(root,"..\\outputs\\database.db")

# Initialize COLMAP database
db = pycolmap.Database(database_path)

print(f"Database created at: {database_path}")

image_folder_left = os.path.join(image_folder, "left")
image_folder_right = os.path.join(image_folder, "right")

pycolmap.extract_features(database_path, image_folder_left)
pycolmap.extract_features(database_path, image_folder_right)



# pycolmap.match_features(database_path)
# # Ensure the output directory exists
# sparse_output_path = os.path.join(root, "..\\outputs\\sparse")
# os.makedirs(sparse_output_path, exist_ok=True)

# pycolmap.reconstruct_sparse(
#     database_path,
#     image_folder,
#     output_path=sparse_output_path
# )