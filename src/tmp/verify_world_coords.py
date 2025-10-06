"""
Verify the world coordinates make sense relative to court center
"""
import numpy as np

court_center = np.array([-0.03, -4.5, 0.014])
court_length = 7.79
court_width = 15.9

# Sample ball positions from diagnostic
ball_positions = [
    np.array([-0.045, -8.54, 2.54]),
    np.array([0.022, -6.09, 2.72]),
    np.array([-0.18, -3.88, 2.53]),
    np.array([-0.18, -2.11, 1.66])
]

print("Court center:", court_center)
print("Court dimensions: {:.2f}m (width) x {:.2f}m (length)\n".format(court_width, court_length))

for i, ball in enumerate(ball_positions, 90):
    rel = ball - court_center
    print(f"Frame {i}: Ball at {ball}")
    print(f"  Relative to court center: X={rel[0]:.2f}m, Y={rel[1]:.2f}m, Z={rel[2]:.2f}m")
    
    # Check bounds
    x_ok = abs(rel[0]) <= court_width/2
    y_ok = abs(rel[1]) <= court_length/2
    z_ok = 0 <= rel[2] <= 15
    
    status = "✓" if (x_ok and y_ok and z_ok) else "✗"
    print(f"  In bounds: {status} (X:{x_ok}, Y:{y_ok}, Z:{z_ok})")
    print()
