import pandas as pd
import numpy as np
from scipy.spatial.transform import Rotation as R_scipy
from scipy.optimize import minimize

# Camera Parameters
FOCAL_LENGTH = 1600
CX = 960
CY = 540

# Extrinsics from hawkeye_pipeline.py
R_wc_orig = np.array([[-0.016771, 0.19022 , -0.9816 ],
                 [ 0.99516 , 0.098228,  0.002032],
                 [ 0.096807, -0.97682, -0.19094]])
t_wc_orig = np.array([17.569, -6.6545, 6.1974])

# Detections (Frame: (u, v))
detections = {
    9: (1899.70, 587.74),
    10: (1801.17, 578.95),
    11: (1692.50, 571.12),
    12: (1585.78, 561.12),
    13: (1477.83, 553.59),
    14: (1371.37, 542.80),
    15: (1263.97, 536.69),
    16: (1154.33, 524.88),
    17: (1049.16, 546.91),
    18: (944.08, 571.09),
    19: (838.12, 592.29),
    20: (729.82, 611.70)
}

# Load CSV
csv_path = r"f:\hawkeye\3D-models\Latest volley go brr\ball_positions_blender.csv"
df = pd.read_csv(csv_path)

def project(params):
    # params: [pitch_deg, yaw_deg, roll_deg, tx, ty, tz]
    pitch, yaw, roll, tx, ty, tz = params
    
    # Apply rotation correction
    r_corr = R_scipy.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix()
    R_wc_new = R_wc_orig @ r_corr
    t_wc_new = t_wc_orig + np.array([tx, ty, tz])
    
    # Invert
    R_cw = R_wc_new.T
    t_cw = -R_cw @ t_wc_new
    
    error = 0
    count = 0
    
    # Use fixed offset 1 (found previously)
    offset = 1
    
    for frame_idx, (det_u, det_v) in detections.items():
        csv_frame = frame_idx + offset
        row = df[df['frame'] == csv_frame]
        if len(row) == 0:
            continue
            
        x, y, z = row.iloc[0][['x', 'y', 'z']]
        
        p_world = np.array([x, y, z])
        p_cam = R_cw @ p_world + t_cw
        
        xc, yc, zc = p_cam
        if zc <= 0:
            continue
            
        u = (xc / zc) * FOCAL_LENGTH + CX
        v = (yc / zc) * FOCAL_LENGTH + CY
        
        dist = (det_u - u)**2 + (det_v - v)**2
        error += dist
        count += 1
        
    return np.sqrt(error / count) if count > 0 else float('inf')

# Optimize
initial_guess = [0, 0, 0, 0, 0, 0]
res = minimize(project, initial_guess, method='Nelder-Mead', tol=1e-4)

print("Optimization Result:")
print(res)
print(f"\nBest Params: {res.x}")

# Print comparison
params = res.x
pitch, yaw, roll, tx, ty, tz = params
r_corr = R_scipy.from_euler('xyz', [pitch, yaw, roll], degrees=True).as_matrix()
R_wc_new = R_wc_orig @ r_corr
t_wc_new = t_wc_orig + np.array([tx, ty, tz])
R_cw = R_wc_new.T
t_cw = -R_cw @ t_wc_new

print(f"\nComparison with Optimized Extrinsics:")
print(f"{'Frame':<6} {'Det U':<8} {'Det V':<8} {'Proj U':<8} {'Proj V':<8} {'Diff':<8}")
for frame_idx, (det_u, det_v) in detections.items():
    csv_frame = frame_idx + 1
    row = df[df['frame'] == csv_frame]
    if len(row) > 0:
        x, y, z = row.iloc[0][['x', 'y', 'z']]
        p_world = np.array([x, y, z])
        p_cam = R_cw @ p_world + t_cw
        xc, yc, zc = p_cam
        if zc > 0:
            u = (xc / zc) * FOCAL_LENGTH + CX
            v = (yc / zc) * FOCAL_LENGTH + CY
            diff = np.sqrt((det_u - u)**2 + (det_v - v)**2)
            print(f"{frame_idx:<6} {det_u:<8.1f} {det_v:<8.1f} {u:<8.1f} {v:<8.1f} {diff:<8.1f}")
