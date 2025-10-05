import numpy as np

# Blender positions
right_camera = np.array([18, -3, 3])
left_camera = np.array([18, -6, 3])
court_center = np.array([-0.03, -4.5, 0.014])

print("BLENDER SETUP ANALYSIS")
print("="*60)

# Calculate baseline
baseline_vector = left_camera - right_camera
baseline_distance = np.linalg.norm(baseline_vector)

print(f"Camera Positions:")
print(f"  Right: {right_camera}")
print(f"  Left:  {left_camera}")
print(f"  Court: {court_center}")

print(f"\nBaseline Analysis:")
print(f"  Vector: {baseline_vector}")
print(f"  Distance: {baseline_distance:.2f}m")

# Calculate camera to court distances
dist_right_to_court = np.linalg.norm(right_camera - court_center)
dist_left_to_court = np.linalg.norm(left_camera - court_center)

print(f"\nCamera to Court Distances:")
print(f"  Right camera to court: {dist_right_to_court:.2f}m")
print(f"  Left camera to court: {dist_left_to_court:.2f}m")

# Check camera orientation
# Cameras are along Y-axis (left camera more negative Y)
# Court is also along Y-axis

print(f"\nCamera Orientation:")
if baseline_vector[1] < 0:
    print(f"  ✅ Left camera is at more negative Y (-6 vs -3)")
    print(f"  ✅ Cameras aligned along Y-axis")
    print(f"  → Baseline direction: {baseline_vector}")
else:
    print(f"  ❌ Camera orientation incorrect")

# The issue: cameras are looking DOWN the Y-axis
# But ball moves along X-axis or Z-axis
# This creates minimal disparity!

print(f"\n🔍 PROBLEM IDENTIFIED:")
print(f"  Cameras at X=18, Court at X≈0")
print(f"  Cameras separated along Y-axis (parallel to court length)")
print(f"  This means:")
print(f"    - For ball moving along court (Y direction): MINIMAL disparity")
print(f"    - For ball moving across court (X direction): GOOD disparity")
print(f"    - Ball depth is primarily in X direction (~18m)")

# Calculate expected disparity for a ball at court center
ball_pos = court_center.copy()
ball_pos[2] = 2.0  # Ball at 2m height

# Project to left camera view
left_to_ball = ball_pos - left_camera
right_to_ball = ball_pos - right_camera

print(f"\nFor ball at court center (height 2m):")
print(f"  Ball position: {ball_pos}")
print(f"  Distance from left camera: {np.linalg.norm(left_to_ball):.2f}m")
print(f"  Distance from right camera: {np.linalg.norm(right_to_ball):.2f}m")

# The actual baseline effective for stereo is the perpendicular component
# to the viewing direction
viewing_direction = court_center - right_camera
viewing_direction = viewing_direction / np.linalg.norm(viewing_direction)

baseline_perpendicular = baseline_distance * np.abs(np.dot(baseline_vector/baseline_distance, 
                                                             np.array([0, 1, 0])))

print(f"\n⚠️  STEREO GEOMETRY ISSUE:")
print(f"  Baseline: {baseline_distance:.2f}m along Y-axis")
print(f"  Viewing direction: Primarily X-axis (~18m difference)")
print(f"  Effective stereo baseline for X-depth: ~{baseline_perpendicular:.2f}m")
print(f"  This creates small disparity!")

print(f"\n💡 SOLUTION:")
print(f"  1. Your baseline should be perpendicular to viewing direction")
print(f"  2. Currently: baseline is PARALLEL to court length (Y-axis)")
print(f"  3. Better: baseline along X-axis (perpendicular to line of sight)")
print(f"  4. Try: Right at (18, -4.5, 3), Left at (15, -4.5, 3)")
print(f"     or: Right at (21, -4.5, 3), Left at (18, -4.5, 3)")
