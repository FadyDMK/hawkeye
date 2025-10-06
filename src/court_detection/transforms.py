import cv2
import numpy as np
def ball_camera_to_world(ball_pos, t, R, scale=None):
    ball_pos = np.array(ball_pos, dtype=float).reshape(3, 1)
    t = np.array(t, dtype=float).reshape(3, 1)
    R = np.array(R, dtype=float)
    # OpenCV camera coords: X=right in image, Y=down, Z=forward (depth)
    # Blender world: X=right in world, Y=forward in world, Z=up
    
    # COORDINATE MAPPING DEPENDS ON VIDEO TYPE:
    # - Historically we relied on manual axis remapping from OpenCV camera to Blender.
    # - The new Umeyama similarity transform incorporates the axis mapping into R.
    # When R is identity (legacy MP4 workflow), we still apply the manual mapping.

    if R is None or np.allclose(R, np.eye(3)):
        base_coords = np.array([ball_pos[1, 0], -ball_pos[2, 0], -ball_pos[0, 0]]).reshape(3, 1)
    else:
        base_coords = ball_pos

    if scale is not None and not np.isscalar(scale):
        scale = np.array(scale, dtype=float).reshape(3, 1)
        base_coords = base_coords * scale

    rotated = R @ base_coords
    if scale is not None and np.isscalar(scale):
        rotated = float(scale) * rotated

    world = rotated + t
    print("Camera to World:", base_coords.flatten(), "→", world.flatten())
    return world.flatten()