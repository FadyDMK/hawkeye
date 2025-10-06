"""Test multiple frames to see why world coords are None"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from hawkeye_pipeline import HawkeyePipeline
from camera_config import load_camera_config

print("="*60)
print("TESTING MULTIPLE FRAMES - World Coords Analysis")
print("="*60)

config = load_camera_config()
print(f"\nConfig:")
print(f"  Court: {config.get('court_length_m')}m x {config.get('court_width_m')}m")
print(f"  Max height: {config.get('max_ball_height_m')}m")
print(f"  Tolerance in validation: ±3.0m")

pipeline = HawkeyePipeline(config)

print(f"\n" + "="*60)
print("Testing frames 90-99:")
print("="*60)

success_count = 0
camera_success = 0
world_none_count = 0

for frame_num in range(90, 100):
    print(f"\nFrame {frame_num}:")
    result = pipeline.process_single_frame(frame_num)
    
    if result and isinstance(result, dict):
        cam = result['camera_coords']
        world = result['world_coords']
        
        if cam[0] is not None:
            camera_success += 1
            print(f"  Camera: ({cam[0]:.2f}, {cam[1]:.2f}, {cam[2]:.2f})")
            
            if world[0] is not None:
                print(f"  World:  ({world[0]:.2f}, {world[1]:.2f}, {world[2]:.2f}) ✅")
                success_count += 1
            else:
                print(f"  World:  (None, None, None) ❌ - REJECTED BY VALIDATION")
                world_none_count += 1
                
                # Calculate what the world coords would have been
                from court_detection.transforms import ball_camera_to_world
                world_raw = ball_camera_to_world(cam, pipeline.t, pipeline.R)
                x, y, z = world_raw
                
                court_width = config.get("court_width_m", 9.0)
                court_length = config.get("court_length_m", 18.0)
                max_height = config.get("max_ball_height_m", 15.0)
                
                width_tol = 3.0
                length_tol = 3.0
                
                x_valid = abs(x) <= (court_width / 2 + width_tol)
                y_valid = abs(y) <= (court_length / 2 + length_tol)
                z_valid = 0 <= z <= max_height
                
                print(f"    Raw world: ({x:.2f}, {y:.2f}, {z:.2f})")
                print(f"    X: {x:.2f} vs ±{court_width/2 + width_tol:.2f} → {x_valid}")
                print(f"    Y: {y:.2f} vs ±{court_length/2 + length_tol:.2f} → {y_valid}")
                print(f"    Z: {z:.2f} vs [0, {max_height:.2f}] → {z_valid}")
        else:
            print(f"  ❌ No ball detected in camera coords")
    else:
        print(f"  ❌ Processing failed")

print(f"\n" + "="*60)
print("SUMMARY:")
print("="*60)
print(f"Camera detection: {camera_success}/10 frames")
print(f"World coords valid: {success_count}/10 frames")
print(f"World coords rejected: {world_none_count}/10 frames")

if world_none_count > 0:
    print(f"\n⚠️  {world_none_count} frames rejected by court bounds validation!")
    print(f"\nPossible solutions:")
    print(f"1. Check if court dimensions are correct in config")
    print(f"2. Check if camera transform matrices (t, R) are correct")
    print(f"3. Increase tolerance margins in validate_court_bounds()")
    print(f"4. Verify coordinate system matches between Blender and OpenCV")
