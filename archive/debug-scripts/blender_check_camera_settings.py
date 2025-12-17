"""
Blender script to extract camera properties for both left and right cameras.
Run this in Blender's scripting workspace to compare camera settings
between the newnew and final video renders.

Instructions:
1. Open your Blender project with the finalLeft/finalRight cameras
2. Go to Scripting workspace
3. Paste this script and click "Run Script" (Alt+P)
4. Check the console output (Window > Toggle System Console)
"""

import bpy
import math

def get_camera_properties(camera_name):
    """Extract all relevant camera properties."""
    if camera_name not in bpy.data.objects:
        return None
    
    cam_obj = bpy.data.objects[camera_name]
    cam_data = cam_obj.data
    
    # Get scene render settings
    scene = bpy.context.scene
    render_x = scene.render.resolution_x
    render_y = scene.render.resolution_y
    
    # Calculate sensor size and focal length
    sensor_width = cam_data.sensor_width
    sensor_height = cam_data.sensor_height
    focal_length_mm = cam_data.lens
    
    # Calculate focal length in pixels
    if cam_data.sensor_fit == 'HORIZONTAL' or (cam_data.sensor_fit == 'AUTO' and render_x >= render_y):
        focal_length_px = (focal_length_mm / sensor_width) * render_x
    else:
        focal_length_px = (focal_length_mm / sensor_height) * render_y
    
    # Get camera position and rotation
    location = cam_obj.location
    rotation = cam_obj.rotation_euler
    
    return {
        'name': camera_name,
        'focal_length_mm': focal_length_mm,
        'focal_length_px': focal_length_px,
        'sensor_width': sensor_width,
        'sensor_height': sensor_height,
        'sensor_fit': cam_data.sensor_fit,
        'location': (location.x, location.y, location.z),
        'rotation': (math.degrees(rotation.x), math.degrees(rotation.y), math.degrees(rotation.z)),
        'resolution': (render_x, render_y),
    }

def calculate_baseline(left_cam, right_cam):
    """Calculate distance between cameras (baseline)."""
    import math
    left_loc = left_cam['location']
    right_loc = right_cam['location']
    
    baseline = math.sqrt(
        (right_loc[0] - left_loc[0])**2 +
        (right_loc[1] - left_loc[1])**2 +
        (right_loc[2] - left_loc[2])**2
    )
    return baseline

# Try to find cameras (multiple possible names)
left_names = ['CameraLeft', 'Camera.Left', 'left', 'Left', 'camera_left', 'cam_left']
right_names = ['CameraRight', 'Camera.Right', 'right', 'Right', 'camera_right', 'cam_right']

left_cam = None
right_cam = None

for name in left_names:
    if name in bpy.data.objects:
        left_cam = get_camera_properties(name)
        break

for name in right_names:
    if name in bpy.data.objects:
        right_cam = get_camera_properties(name)
        break

print("\n" + "="*70)
print("BLENDER CAMERA PROPERTIES EXTRACTION")
print("="*70)

if left_cam:
    print(f"\n### LEFT CAMERA ({left_cam['name']}) ###")
    print(f"Focal length: {left_cam['focal_length_mm']:.2f} mm ({left_cam['focal_length_px']:.2f} px)")
    print(f"Sensor: {left_cam['sensor_width']:.2f} x {left_cam['sensor_height']:.2f} mm ({left_cam['sensor_fit']})")
    print(f"Resolution: {left_cam['resolution'][0]} x {left_cam['resolution'][1]}")
    print(f"Location: ({left_cam['location'][0]:.4f}, {left_cam['location'][1]:.4f}, {left_cam['location'][2]:.4f})")
    print(f"Rotation: ({left_cam['rotation'][0]:.2f}°, {left_cam['rotation'][1]:.2f}°, {left_cam['rotation'][2]:.2f}°)")
else:
    print("\n❌ LEFT CAMERA NOT FOUND!")
    print(f"Available cameras: {[obj.name for obj in bpy.data.objects if obj.type == 'CAMERA']}")

if right_cam:
    print(f"\n### RIGHT CAMERA ({right_cam['name']}) ###")
    print(f"Focal length: {right_cam['focal_length_mm']:.2f} mm ({right_cam['focal_length_px']:.2f} px)")
    print(f"Sensor: {right_cam['sensor_width']:.2f} x {right_cam['sensor_height']:.2f} mm ({right_cam['sensor_fit']})")
    print(f"Resolution: {right_cam['resolution'][0]} x {right_cam['resolution'][1]}")
    print(f"Location: ({right_cam['location'][0]:.4f}, {right_cam['location'][1]:.4f}, {right_cam['location'][2]:.4f})")
    print(f"Rotation: ({right_cam['rotation'][0]:.2f}°, {right_cam['rotation'][1]:.2f}°, {right_cam['rotation'][2]:.2f}°)")
else:
    print("\n❌ RIGHT CAMERA NOT FOUND!")
    print(f"Available cameras: {[obj.name for obj in bpy.data.objects if obj.type == 'CAMERA']}")

if left_cam and right_cam:
    baseline = calculate_baseline(left_cam, right_cam)
    print(f"\n### STEREO SETUP ###")
    print(f"Baseline (camera separation): {baseline:.4f} m")
    
    # Check if focal lengths match
    focal_diff = abs(left_cam['focal_length_px'] - right_cam['focal_length_px'])
    if focal_diff > 1.0:
        print(f"⚠️  WARNING: Focal lengths don't match! Difference: {focal_diff:.2f} px")
    else:
        print(f"✓ Focal lengths match: {left_cam['focal_length_px']:.2f} px")
    
    print(f"\n### COMPARISON TO camera_config.json ###")
    print(f"Config focal_length_px: 1600.0")
    print(f"Blender focal_length_px: {left_cam['focal_length_px']:.2f}")
    print(f"Difference: {abs(1600.0 - left_cam['focal_length_px']):.2f} px")
    print(f"\nConfig baseline_m: 3.0")
    print(f"Blender baseline_m: {baseline:.4f}")
    print(f"Difference: {abs(3.0 - baseline):.4f} m")

print("\n" + "="*70)
print("INSTRUCTIONS:")
print("1. If focal length or baseline differs, update camera_config.json")
print("2. Or recalibrate the pipeline using these new values")
print("="*70)

