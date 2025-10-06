"""Calculate correct t and R matrices from Blender camera positions"""
import numpy as np

print("="*60)
print("CALCULATING CORRECT TRANSFORMATION MATRICES")
print("="*60)

# Your Blender camera positions (from earlier analysis)
left_cam_blender = np.array([18.0, -6.0, 3.0])
right_cam_blender = np.array([18.0, -3.0, 3.0])
court_center_blender = np.array([-0.03, -4.5, 0.014])

print(f"\nBlender scene setup:")
print(f"  Left camera:  {left_cam_blender}")
print(f"  Right camera: {right_cam_blender}")
print(f"  Court center: {court_center_blender}")

# Calculate baseline
baseline_vector = right_cam_blender - left_cam_blender
baseline_magnitude = np.linalg.norm(baseline_vector)
print(f"\nBaseline vector: {baseline_vector}")
print(f"Baseline magnitude: {baseline_magnitude:.2f}m")

# The stereo rig coordinate system:
# - Origin at LEFT camera
# - X axis along baseline (left to right)
# - Z axis pointing forward (toward the scene)
# - Y axis completing right-hand coordinate system

print(f"\n" + "="*60)
print("OPTION 1: Simple Translation Only (No Rotation)")
print("="*60)
print("If we use the LEFT camera as the origin:")
print("  - Camera space origin = Left camera position in Blender")
print("  - World space origin = Some reference point (e.g., court center)")

# For simple case: just translate from camera origin to world origin
# Translation vector t: position of world origin in camera coordinates
# If camera is at (18, -6, 3) and world origin is at court center (-0.03, -4.5, 0.014)
# Then: t = court_center - left_camera = (-0.03 - 18, -4.5 - (-6), 0.014 - 3)

t_simple = court_center_blender - left_cam_blender
print(f"\nTranslation vector t: {t_simple}")
print(f"  This means: camera is at {-t_simple} in world coordinates")

# For coordinate system alignment:
# Blender: X=right, Y=forward, Z=up
# OpenCV:  X=right, Y=down, Z=forward
# We need to swap and flip axes

print(f"\n" + "="*60)
print("OPTION 2: With Coordinate System Conversion")
print("="*60)

# Rotation to convert Blender -> OpenCV camera coordinates
# Blender (X, Y, Z) -> OpenCV (X', Y', Z')
# X' = X (right stays right)
# Y' = -Z (up becomes down)
# Z' = Y (forward stays forward)

R_blender_to_opencv = np.array([
    [1,  0,  0],  # X' = X
    [0,  0, -1],  # Y' = -Z  
    [0,  1,  0]   # Z' = Y
])

print("Rotation matrix (Blender to OpenCV):")
print(R_blender_to_opencv)

# Apply rotation to translation
t_rotated = R_blender_to_opencv @ t_simple
print(f"\nRotated translation: {t_rotated}")

print(f"\n" + "="*60)
print("OPTION 3: Use Court Center as World Origin (Recommended)")
print("="*60)

# Better approach: Set world origin at court center
# Then camera position in world = left_camera - court_center
camera_pos_in_world = left_cam_blender - court_center_blender
print(f"Left camera position relative to court center: {camera_pos_in_world}")

# In OpenCV camera coordinates, this becomes the translation
# Camera looks at -Z direction, so we need to consider viewing direction
# If cameras look toward negative Y in Blender (toward court from X=18)
# Then in OpenCV (Z forward), the translation is:

# Actually, for stereo where left camera is origin:
# t should be negative of camera world position (to transform world -> camera)
t_recommended = -camera_pos_in_world
print(f"Translation t (world origin to camera): {t_recommended}")

# Apply coordinate conversion
t_final = R_blender_to_opencv @ camera_pos_in_world
print(f"After coordinate conversion: {t_final}")

print(f"\n" + "="*60)
print("RECOMMENDATION FOR hawkeye_pipeline.py:")
print("="*60)

print(f"\nReplace these lines around line 38-41:")
print(f"\n# OLD (WRONG):")
print(f"self.R = [[-4.37113883e-08, 4.37113883e-08, 1.00000000e+00],")
print(f"          [1.00000000e+00, 1.91068568e-15, 4.37113883e-08],")
print(f"          [0.00000000e+00, 1.00000000e+00, -4.37113883e-08]]")
print(f"self.t = [25.0, -1.5, 5.0]")

print(f"\n# NEW (CORRECT for your Blender scene):")
print(f"# Coordinate conversion: Blender (X,Y,Z) -> OpenCV (X,-Z,Y)")
print(f"self.R = [[1, 0, 0],")
print(f"          [0, 0, -1],") 
print(f"          [0, 1, 0]]")
print(f"self.t = [{camera_pos_in_world[0]:.2f}, {camera_pos_in_world[1]:.2f}, {camera_pos_in_world[2]:.2f}]")

print(f"\n" + "="*60)
print("TESTING THE NEW VALUES:")
print("="*60)

# Simulate what would happen with a point at camera coords (0, 0, 18)
# (meaning 18m in front of left camera, centered)
test_point_camera = np.array([0, 0, 18])
print(f"\nTest: Ball at camera coords {test_point_camera}")
print(f"  (0m sideways, 0m vertical, 18m forward from left camera)")

# Transform to world using new values
test_point_world = R_blender_to_opencv @ test_point_camera + camera_pos_in_world
print(f"  World coords: {test_point_world}")
print(f"  Expected: close to court center at {court_center_blender}")

distance_from_court = np.linalg.norm(test_point_world - court_center_blender)
print(f"  Distance from court center: {distance_from_court:.2f}m")

if distance_from_court < 2:
    print(f"  ✅ Looks correct! Ball maps near court center as expected")
else:
    print(f"  ⚠️  Still seems off by {distance_from_court:.2f}m")
