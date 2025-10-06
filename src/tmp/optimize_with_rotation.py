"""
Optimize BOTH rotation R and translation t, not just t.
Maybe the cameras aren't perfectly aligned with the world frame.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation

# Camera coords with baseline 2.115m
camera_coords = {
    90: np.array([7.5375, 2.1150, 18.0000]),
    93: np.array([7.0500, 1.9785, 18.1935]),
    96: np.array([6.4019, 1.8421, 18.1935]),
    99: np.array([5.7556, 1.7336, 18.4918])
}

# Load ground truth
blender_df = pd.read_csv('ball_positions_blender.csv')

# Test all coordinate mappings with optimized R and t
def test_mapping_with_rotation(mapping_idx):
    """
    mapping_idx: 0-47 for the 48 possible mappings
    """
    # Generate mapping from index
    import itertools
    axes = [0, 1, 2]
    signs = [-1, 1]
    
    all_mappings = []
    for perm in itertools.permutations(axes):
        for sign_combo in itertools.product(signs, repeat=3):
            mapping = [perm[i] * sign_combo[i] for i in range(3)]
            all_mappings.append(mapping)
    
    mapping = all_mappings[mapping_idx]
    
    def objective(params):
        """Objective function: sum of squared errors."""
        # params = [rx, ry, rz, tx, ty, tz] (rotation angles in degrees, translation)
        rx, ry, rz, tx, ty, tz = params
        
        # Create rotation matrix
        R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        t = np.array([tx, ty, tz])
        
        total_error = 0
        for frame in [90, 93, 96, 99]:
            if frame in blender_df['frame'].values:
                row = blender_df[blender_df['frame'] == frame].iloc[0]
                target = np.array([row['x'], row['y'], row['z']])
                
                cam_pos = camera_coords[frame]
                
                # Apply mapping
                cam_blender = np.array([
                    cam_pos[abs(mapping[0])] * np.sign(mapping[0]) if mapping[0] != 0 else -cam_pos[abs(mapping[0])],
                    cam_pos[abs(mapping[1])] * np.sign(mapping[1]) if mapping[1] != 0 else -cam_pos[abs(mapping[1])],
                    cam_pos[abs(mapping[2])] * np.sign(mapping[2]) if mapping[2] != 0 else -cam_pos[abs(mapping[2])]
                ])
                
                hawkeye_pos = R @ cam_blender + t
                error = np.sum((hawkeye_pos - target)**2)
                total_error += error
        
        return total_error
    
    # Initial guess: identity rotation, zero translation
    x0 = [0, 0, 0, 0, 0, 0]
    
    # Optimize
    result = minimize(objective, x0, method='Powell', options={'maxiter': 5000})
    
    if result.success:
        rx, ry, rz, tx, ty, tz = result.x
        R = Rotation.from_euler('xyz', [rx, ry, rz], degrees=True).as_matrix()
        t = np.array([tx, ty, tz])
        
        # Calculate average error
        errors = []
        for frame in [90, 93, 96, 99]:
            if frame in blender_df['frame'].values:
                row = blender_df[blender_df['frame'] == frame].iloc[0]
                target = np.array([row['x'], row['y'], row['z']])
                
                cam_pos = camera_coords[frame]
                
                # Apply mapping
                cam_blender = np.array([
                    cam_pos[abs(mapping[0])] * np.sign(mapping[0]) if mapping[0] != 0 else -cam_pos[abs(mapping[0])],
                    cam_pos[abs(mapping[1])] * np.sign(mapping[1]) if mapping[1] != 0 else -cam_pos[abs(mapping[1])],
                    cam_pos[abs(mapping[2])] * np.sign(mapping[2]) if mapping[2] != 0 else -cam_pos[abs(mapping[2])]
                ])
                
                hawkeye_pos = R @ cam_blender + t
                error = np.linalg.norm(hawkeye_pos - target)
                errors.append(error)
        
        avg_error = np.mean(errors)
        
        return {
            'mapping': mapping,
            'rotation': [rx, ry, rz],
            'translation': [tx, ty, tz],
            'R': R,
            't': t,
            'avg_error': avg_error
        }
    else:
        return None

print("="*70)
print("OPTIMIZING WITH ROTATION + TRANSLATION")
print("Testing top 10 coordinate mappings...")
print("="*70)

# Test the most promising mappings (based on previous results)
promising_mappings = [
    10,  # [+Y, -X, -Z]
    11,  # Duplicate
    12,  # [-Y, -X, -Z]
    20,  # [-Z, -X, +Y]
    28,  # [-Z, -X, -Y]
]

results = []
for idx in promising_mappings:
    print(f"\nTesting mapping {idx}...")
    result = test_mapping_with_rotation(idx)
    if result:
        results.append(result)
        print(f"  Average error: {result['avg_error']:.3f}m")

# Find best
if results:
    results.sort(key=lambda x: x['avg_error'])
    best = results[0]
    
    print("\n" + "="*70)
    print("🎯 BEST RESULT WITH ROTATION:")
    print("="*70)
    print(f"Mapping: {best['mapping']}")
    print(f"Rotation (deg): [{best['rotation'][0]:.2f}, {best['rotation'][1]:.2f}, {best['rotation'][2]:.2f}]")
    print(f"Translation: [{best['translation'][0]:.3f}, {best['translation'][1]:.3f}, {best['translation'][2]:.3f}]")
    print(f"Average Error: {best['avg_error']:.3f}m")
    
    print("\nRotation matrix R:")
    print(best['R'])
    
    # Verify on each frame
    print("\n" + "="*70)
    print("FRAME-BY-FRAME VERIFICATION:")
    print("="*70)
    
    mapping = best['mapping']
    R = best['R']
    t = best['t']
    
    for frame in [90, 93, 96, 99]:
        if frame in blender_df['frame'].values:
            row = blender_df[blender_df['frame'] == frame].iloc[0]
            target = np.array([row['x'], row['y'], row['z']])
            
            cam_pos = camera_coords[frame]
            
            # Apply mapping
            cam_blender = np.array([
                cam_pos[abs(mapping[0])] * np.sign(mapping[0]) if mapping[0] != 0 else -cam_pos[abs(mapping[0])],
                cam_pos[abs(mapping[1])] * np.sign(mapping[1]) if mapping[1] != 0 else -cam_pos[abs(mapping[1])],
                cam_pos[abs(mapping[2])] * np.sign(mapping[2]) if mapping[2] != 0 else -cam_pos[abs(mapping[2])]
            ])
            
            hawkeye_pos = R @ cam_blender + t
            error_vec = hawkeye_pos - target
            error = np.linalg.norm(error_vec)
            
            print(f"\nFrame {frame}:")
            print(f"  Ground Truth: [{target[0]:7.3f}, {target[1]:7.3f}, {target[2]:7.3f}]")
            print(f"  Hawkeye:      [{hawkeye_pos[0]:7.3f}, {hawkeye_pos[1]:7.3f}, {hawkeye_pos[2]:7.3f}]")
            print(f"  Error:        [{error_vec[0]:7.3f}, {error_vec[1]:7.3f}, {error_vec[2]:7.3f}] = {error:.3f}m")
