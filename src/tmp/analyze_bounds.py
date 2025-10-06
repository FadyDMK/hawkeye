"""Simple test without extra imports"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from hawkeye_pipeline import HawkeyePipeline
from camera_config import load_camera_config

config = load_camera_config()
pipeline = HawkeyePipeline(config)

print("="*60)
print("COURT BOUNDS ANALYSIS")
print("="*60)

court_width = config.get("court_width_m", 9.0)
court_length = config.get("court_length_m", 18.0)
max_height = config.get("max_ball_height_m", 15.0)

print(f"\nCourt dimensions from config:")
print(f"  Width:  {court_width}m")
print(f"  Length: {court_length}m")
print(f"  Max height: {max_height}m")

width_tolerance = 3.0
length_tolerance = 3.0

print(f"\nValidation bounds:")
print(f"  X (width):  ±{court_width/2 + width_tolerance:.2f}m")
print(f"  Y (length): ±{court_length/2 + length_tolerance:.2f}m")
print(f"  Z (height): 0 to {max_height:.2f}m")

print(f"\n" + "="*60)
print("Example rejections from frame 90:")
print("="*60)

# Example from the log
world_pos = (6.95, -4.04, 4.54)
x, y, z = world_pos

x_valid = abs(x) <= (court_width / 2 + width_tolerance)
y_valid = abs(y) <= (court_length / 2 + length_tolerance)
z_valid = 0 <= z <= max_height

print(f"\nWorld position: ({x:.2f}, {y:.2f}, {z:.2f})")
print(f"  X check: |{x:.2f}| <= {court_width/2 + width_tolerance:.2f} → {x_valid} {'✅' if x_valid else '❌'}")
print(f"  Y check: |{y:.2f}| <= {court_length/2 + length_tolerance:.2f} → {y_valid} {'✅' if y_valid else '❌'}")
print(f"  Z check: 0 <= {z:.2f} <= {max_height:.2f} → {z_valid} {'✅' if z_valid else '❌'}")
print(f"  Overall: {x_valid and y_valid and z_valid} {'✅' if (x_valid and y_valid and z_valid) else '❌'}")

if not x_valid:
    print(f"\n⚠️  X is out of bounds! Court width is too small.")
    print(f"   Current: {court_width}m, needs at least: {2 * (abs(x) - width_tolerance):.2f}m")
    
if not y_valid:
    print(f"\n⚠️  Y is out of bounds! Court length is too small.")
    print(f"   Current: {court_length}m, needs at least: {2 * (abs(y) - length_tolerance):.2f}m")

print(f"\n" + "="*60)
print("RECOMMENDATIONS:")
print("="*60)

print(f"\n1. Your court dimensions seem too small for synthetic footage")
print(f"   Current: {court_length}m x {court_width}m")
print(f"   Standard volleyball: 18m x 9m")
print(f"   Your Blender scene: Check actual court size")

print(f"\n2. If this is a scaled-down scene, you have two options:")
print(f"   A) Use realistic court dimensions (40m x 31m for outdoor)")
print(f"   B) Increase tolerance to 10m or more")
print(f"   C) Disable bounds checking for synthetic test data")

print(f"\n3. Quick fix - increase tolerance:")
print(f"   Edit src/hawkeye_pipeline.py, line ~62:")
print(f"   width_tolerance = 10.0  # was 3.0")
print(f"   length_tolerance = 10.0  # was 3.0")
