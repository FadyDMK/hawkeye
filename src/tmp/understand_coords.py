"""
Rethink the coordinate transform from scratch.

BLENDER SCENE:
- Court center: (-0.03, -4.5, 0.014)
- Left camera: (18.0, -6.0, 3.0)
- Right camera: (18.0, -3.0, 3.0)
- Blender coords: X right, Y forward, Z up

OPENCV CAMERA COORDS (from stereo triangulation):
- X right (positive to the right in image)
- Y down (positive downward in image)
- Z forward (positive into the scene, away from camera)

STEREO BASELINE:
- Cameras separated by 3m in Y direction (Blender)
- So stereo baseline is along Blender Y axis

CAMERA ORIENTATION:
- Cameras at X=18, looking toward court at X~0
- So cameras are looking in negative X direction (Blender)
- This means OpenCV Z (forward) = Blender -X (negative X)

COORDINATE MAPPING:
If camera is at (18, -6, 3) looking toward (-0.03, -4.5, 0.014):
- OpenCV X (right) = Blender Y (cameras separated in Y)
- OpenCV Y (down) = Blender -Z (up becomes down)
- OpenCV Z (forward) = Blender -X (forward is toward negative X)

TRANSFORM EQUATION:
world = R @ (opencv_to_blender(camera_coords)) + t

where:
- opencv_to_blender([x, y, z]) = [?, ?, ?]  (need to figure this out)
- t = camera position in Blender = [18.0, -6.0, 3.0] for left camera
- R = identity if cameras are aligned with world

Let's test: 
Ball at camera coords (0, 0, 18) should map to:
- Start at camera position: (18, -6, 3)
- Move 18m forward (OpenCV Z) = 18m in Blender -X
- Final position: (18-18, -6, 3) = (0, -6, 3)

This should be approximately at the court center (-0.03, -4.5, 0.014)?
No - there's a 1.5m offset in Y and 3m offset in Z.

Wait, the court center is at Y=-4.5, and camera is at Y=-6, so
camera is 1.5m behind the court center (in Y direction).

Let me reconsider: maybe the cameras are NOT perfectly aligned with world axes.
Or maybe there's a rotation involved.
"""

import numpy as np

# Scene setup
court_center = np.array([-0.03, -4.5, 0.014])
left_camera = np.array([18.0, -6.0, 3.0])
right_camera = np.array([18.0, -3.0, 3.0])

# Camera looking direction (from camera to court)
look_direction = court_center - left_camera
look_direction = look_direction / np.linalg.norm(look_direction)

print("Court center:", court_center)
print("Left camera:", left_camera)
print("Right camera:", right_camera)
print("\nCamera looking direction:", look_direction)
print("Normalized:", look_direction / np.linalg.norm(look_direction))

# If camera looking in direction [-0.995, 0.0829, -0.165]
# This is mostly negative X, slight positive Y, slight negative Z
print("\nDominant direction:", "X" if abs(look_direction[0]) > max(abs(look_direction[1]), abs(look_direction[2])) else "other")

# For stereo with baseline in Y direction, if cameras looking in -X direction:
# - OpenCV X (right in image) = Blender Y (sideways)
# - OpenCV Y (down in image) = Blender -Z (up is down)
# - OpenCV Z (forward/depth) = Blender -X (forward is negative X)

print("\n" + "="*60)
print("COORDINATE SYSTEM MAPPING:")
print("="*60)
print("OpenCV X (right) → Blender +Y")
print("OpenCV Y (down)  → Blender -Z")
print("OpenCV Z (forward) → Blender -X")
print("\nSo: [Blender_X, Blender_Y, Blender_Z] = [-opencv_Z, opencv_X, -opencv_Y]")

# Test this mapping
test_camera_coords = np.array([0, 0, 18.0])  # 18m forward from camera
blender_offset = np.array([-test_camera_coords[2], test_camera_coords[0], -test_camera_coords[1]])
world_pos = left_camera + blender_offset

print(f"\nTest: Camera coords (0, 0, 18) in OpenCV")
print(f"  → Offset in Blender: {blender_offset}")
print(f"  → World position: {world_pos}")
print(f"  → Distance from court center: {np.linalg.norm(world_pos - court_center):.2f}m")
