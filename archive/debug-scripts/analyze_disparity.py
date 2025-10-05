import json

# Calculate expected disparity ranges for your setup
config = {
    "focal_length_px": 1386.6666666666667,
    "baseline_m": 3.0,
    "z_min_m": 15.0,
    "z_max_m": 65.0
}

print("EXPECTED DISPARITY ANALYSIS")
print("="*50)

focal_length = config["focal_length_px"]
baseline = config["baseline_m"]
z_min = config["z_min_m"]
z_max = config["z_max_m"]

print(f"Focal length: {focal_length} pixels")
print(f"Baseline: {baseline} meters")
print(f"Depth range: {z_min} - {z_max} meters")

# Calculate expected disparity range
max_disparity = (focal_length * baseline) / z_min  # At closest distance
min_disparity = (focal_length * baseline) / z_max  # At farthest distance

print(f"\nExpected disparity range:")
print(f"At {z_min}m (closest): {max_disparity:.1f} pixels")
print(f"At {z_max}m (farthest): {min_disparity:.1f} pixels")

print(f"\nYour observed disparities:")
print(f"Frame 1: -35 pixels (WRONG SIGN!)")
print(f"Frame 2: -17 pixels (WRONG SIGN!)")

# What distances would these correspond to?
obs_disparities = [35, 17]  # Taking absolute values
print(f"\nIf signs were correct, these would mean:")
for d in obs_disparities:
    if d > 0:
        z = (focal_length * baseline) / d
        print(f"Disparity {d} pixels → Distance {z:.1f}m")

print(f"\nDIAGNOSIS:")
print(f"1. Camera setup is BACKWARDS (negative disparity)")
print(f"2. Ball distances {90-150}m are way beyond your {z_max}m limit")
print(f"3. Either camera setup or config needs major correction")