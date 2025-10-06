"""
Check if the ball position is actually within court bounds
when accounting for court center not being at origin
"""

# Blender scene setup
court_center = (-0.03, -4.5, 0.014)
ball_world = (15.492, -1.9624, -15.059)

# Court dimensions from config
court_width = 15.9  # user's custom value
court_length = 7.79  # user's custom value

# Calculate position relative to court center
rel_x = ball_world[0] - court_center[0]
rel_y = ball_world[1] - court_center[1]
rel_z = ball_world[2] - court_center[2]

print("Court dimensions: {}m x {}m".format(court_width, court_length))
print("Court center at: {}".format(court_center))
print("\nBall position in world: {}".format(ball_world))
print("Position relative to court center:")
print("  X: {:.2f}m (width direction, limit ±{:.2f}m)".format(rel_x, court_width/2))
print("  Y: {:.2f}m (length direction, limit ±{:.2f}m)".format(rel_y, court_length/2))
print("  Z: {:.2f}m (height, limit 0-15m)".format(rel_z))

# Check if within bounds
x_in_bounds = abs(rel_x) <= court_width/2
y_in_bounds = abs(rel_y) <= court_length/2
z_in_bounds = 0 <= rel_z <= 15.0

print("\nBounds check:")
print("  X: {} (|{:.2f}| {} {:.2f})".format(
    "✓" if x_in_bounds else "✗", rel_x, "≤" if x_in_bounds else ">", court_width/2))
print("  Y: {} (|{:.2f}| {} {:.2f})".format(
    "✓" if y_in_bounds else "✗", rel_y, "≤" if y_in_bounds else ">", court_length/2))
print("  Z: {} ({:.2f} in [0, 15])".format(
    "✓" if z_in_bounds else "✗", rel_z))
