"""Compute optimal similarity transform (scale + rotation + translation)
from camera coordinates to Blender world coordinates for MKV frames.
Uses Umeyama algorithm.
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline

FRAMES = [80, 82, 84, 85, 88, 90, 92, 94, 96, 99]

BASE_DIR = Path(__file__).resolve().parent
LEFT_DIR = BASE_DIR / "output_frames" / "left"
RIGHT_DIR = BASE_DIR / "output_frames" / "right"
GT_PATH = BASE_DIR / "3D-models" / "Latest volley go brr" / "ball_positions_blender.csv"

def load_camera_coords(pipeline: HawkeyePipeline):
    camera_pts = []
    world_pts = []

    gt_df = pd.read_csv(GT_PATH)

    for frame in FRAMES:
        left_path = LEFT_DIR / f"left3_{frame:04d}.jpg"
        right_path = RIGHT_DIR / f"right3_{frame:04d}.jpg"
        if not left_path.exists() or not right_path.exists():
            print(f"Skipping frame {frame}: image not found")
            continue

        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        if left_img is None or right_img is None:
            print(f"Skipping frame {frame}: failed to read image")
            continue

        pipeline.clear_previous_results()
        if not pipeline.process_from_pair(left_img, right_img, frame_num=frame, display=False):
            print(f"Skipping frame {frame}: pipeline failed")
            continue

        if not pipeline.ball_positions_camera:
            print(f"Skipping frame {frame}: no camera coords")
            continue

        cam = np.array(pipeline.ball_positions_camera[-1], dtype=float)

        row = gt_df[gt_df['frame'] == frame]
        if row.empty:
            print(f"Skipping frame {frame}: no ground truth")
            continue
        gt = row[['x', 'y', 'z']].values.astype(float).squeeze()

        camera_pts.append(cam)
        world_pts.append(gt)
        print(f"Frame {frame}: camera={cam}, gt={gt}")

    return np.array(camera_pts), np.array(world_pts)

def umeyama_alignment(source: np.ndarray, target: np.ndarray):
    """Compute similarity transform that aligns source to target.
    Returns scale, rotation matrix, translation vector such that:
        target ≈ scale * R @ source + t
    """
    assert source.shape == target.shape
    n, dim = source.shape

    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_centered = source - source_mean
    target_centered = target - target_mean

    cov = target_centered.T @ source_centered / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt
    var_source = np.sum(source_centered ** 2) / n
    scale = np.trace(np.diag(D) @ S) / var_source

    t = target_mean - scale * R @ source_mean

    return scale, R, t


def main():
    pipeline = HawkeyePipeline()
    cam_pts, world_pts = load_camera_coords(pipeline)
    if len(cam_pts) < 3:
        print("Not enough data points for alignment")
        return

    scale, R, t = umeyama_alignment(cam_pts, world_pts)

    print("\n=== Optimal similarity transform (camera -> world) ===")
    print(f"Scale: {scale:.6f}")
    print("Rotation matrix:")
    print(R)
    print("Translation vector:")
    print(t)

    # Evaluate error
    pred = (scale * (R @ cam_pts.T)).T + t
    errors = np.linalg.norm(pred - world_pts, axis=1)
    print("\nPer-frame errors (m):")
    for f, err in zip(FRAMES, errors):
        print(f"Frame {f}: {err:.3f}")
    print(f"\nRMS error: {np.sqrt(np.mean(errors ** 2)):.3f}m")

if __name__ == "__main__":
    main()
