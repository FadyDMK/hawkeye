import numpy as np
import math

# Blender positions
right_camera = np.array([18, -3, 3])
left_camera = np.array([18, -6, 3])
court_center = np.array([-0.03, -4.5, 0.014])

print("REVISED BLENDER SETUP ANALYSIS")
print("="*60)

# Calculate baseline
baseline_vector = left_camera - right_camera
baseline_distance = np.linalg.norm(baseline_vector)

print(f"Camera Positions:")
print(f"  Right: {right_camera}")
print(f"  Left:  {left_camera}")
print(f"  Baseline: {baseline_distance:.2f}m along Y-axis")
print(f"  Cameras viewing from SIDE of court (near net)")
print(f"  Each camera rotated 5° toward center")

# Calculate viewing angles
# Right camera at Y=-3, Court at Y=-4.5
# Left camera at Y=-6, Court at Y=-4.5

right_to_court = court_center - right_camera
left_to_court = court_center - left_camera

print(f"\nViewing Vectors:")
print(f"  Right camera → court: {right_to_court}")
print(f"  Left camera → court: {left_to_court}")

# Primary viewing direction is along negative X (from X=18 to X≈0)
print(f"\n📐 CAMERA SETUP:")
print(f"  Cameras at X=18, viewing toward X≈0")
print(f"  This is ~18m viewing distance")
print(f"  Baseline is 3m along Y-axis (perpendicular to main viewing direction)")
print(f"  With 5° convergence toward center")

# This is actually a GOOD stereo setup!
# The problem must be elsewhere...

print(f"\n🤔 WAIT - THIS SETUP SHOULD WORK!")
print(f"  Baseline perpendicular to viewing direction: ✅")
print(f"  Reasonable distance (~18m): ✅")
print(f"  Convergence for overlap: ✅")

# So why negative disparity?
print(f"\n🔍 INVESTIGATING NEGATIVE DISPARITY...")
print(f"  For a point at court center:")
print(f"    - Both cameras look from same X position (18)")
print(f"    - Right camera (Y=-3) looks DOWN (toward Y=-4.5)")
print(f"    - Left camera (Y=-6) looks UP (toward Y=-4.5)")
print(f"    - Objects appear in DIFFERENT vertical positions")

# The issue: with this geometry, objects closer to Y=-3 appear 
# further RIGHT in the right camera
# and objects closer to Y=-6 appear further RIGHT in the left camera

# For a ball at Y=-4.5 (between cameras):
# Right camera sees ball slightly to the RIGHT of center
# Left camera sees ball slightly to the LEFT of center
# This SHOULD give positive disparity...

# Unless... the camera ROTATIONS are the issue!

print(f"\n💡 HYPOTHESIS:")
print(f"  Your camera rotations might be inverted!")
print(f"  Expected: Right camera rotated LEFT, Left camera rotated RIGHT")
print(f"  Reality: Check if rotations are opposite?")

print(f"\n🎯 DIAGNOSTIC QUESTIONS:")
print(f"  1. Which camera is rotated which direction?")
print(f"     - Right camera (Y=-3): Should rotate TOWARD Y=-4.5 (DOWN)")
print(f"     - Left camera (Y=-6): Should rotate TOWARD Y=-4.5 (UP)")
print(f"  2. Are both cameras using same focal length?")
print(f"  3. Are camera sensors oriented the same way?")

# Calculate what we'd expect for a ball at court center
ball_at_center = np.array([-0.03, -4.5, 2.0])  # 2m height

# Simplified projection (ignoring exact camera matrix)
# Just looking at relative Y positions
y_offset_right = ball_at_center[1] - right_camera[1]  # -4.5 - (-3) = -1.5
y_offset_left = ball_at_center[1] - left_camera[1]    # -4.5 - (-6) = 1.5

print(f"\nFor ball at court center (Y=-4.5, height 2m):")
print(f"  Y offset from right camera: {y_offset_right:.2f}m (ball is BELOW camera)")
print(f"  Y offset from left camera: {y_offset_left:.2f}m (ball is ABOVE camera)")

# In image coordinates:
# Right camera: ball appears in UPPER part of image (negative Y offset in world)
# Left camera: ball appears in LOWER part of image (positive Y offset in world)

# But we're measuring HORIZONTAL (X) disparity!
# With 5° rotation, the Y offset projects to X offset

angle_deg = 5
angle_rad = math.radians(angle_deg)

# Rough projection of Y offset to X disparity
# Right camera rotated toward center (positive rotation)
# Left camera rotated toward center (negative rotation)

print(f"\n📊 With 5° convergence:")
print(f"  The Y-axis offsets project to horizontal disparity")
print(f"  This SHOULD create positive disparity")
print(f"  But you're seeing negative disparity...")

print(f"\n🎯 MOST LIKELY ISSUE:")
print(f"  Camera rotation directions are SWAPPED!")
print(f"  Check in Blender:")
print(f"    - Right camera rotation: Should be NEGATIVE around Z-axis")
print(f"    - Left camera rotation: Should be POSITIVE around Z-axis")
print(f"  Or vice versa depending on your coordinate system")
