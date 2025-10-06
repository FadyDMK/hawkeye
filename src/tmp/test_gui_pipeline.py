"""
Diagnostic: Test if GUI is using the correct transformation with scaling.
This mimics exactly what the GUI does.
"""

import sys
import os
import cv2
sys.path.append('src')

from hawkeye_pipeline import HawkeyePipeline
from camera_config import load_camera_config

# Load config exactly like GUI does
config = load_camera_config()

# Create pipeline exactly like GUI does
pipeline = HawkeyePipeline(config)

print("="*80)
print("GUI PIPELINE DIAGNOSTIC")
print("="*80)
print(f"Pipeline scale factors: {pipeline.scale}")
print(f"Pipeline translation: {pipeline.t}")
print(f"Pipeline rotation: {pipeline.R}")
print(f"Baseline: {pipeline.config['baseline_m']}m")
print("="*80)

# Test frame 90 exactly like GUI processes it
print("\nProcessing frame 90 using GUI's process_single_frame()...")
result = pipeline.process_single_frame(90)

if result:
    print("\n✅ SUCCESS!")
    print(f"Camera coords: {result['camera_coords']}")
    print(f"World coords: {result['world_coords']}")
    
    # Expected values
    expected_camera = (7.537, 2.115, 18.000)
    expected_world = (-0.012, -9.007, 2.231)
    
    print(f"\n📊 Comparison with expected:")
    print(f"Expected camera: {expected_camera}")
    print(f"Expected world:  {expected_world}")
    
    import numpy as np
    cam_error = np.linalg.norm(np.array(result['camera_coords']) - np.array(expected_camera))
    world_error = np.linalg.norm(np.array(result['world_coords']) - np.array(expected_world))
    
    print(f"\nCamera coords difference: {cam_error:.3f}")
    print(f"World coords difference:  {world_error:.3f}")
    
    if world_error > 10:
        print("\n❌ MAJOR PROBLEM: World coords are WAY OFF!")
        print("   The GUI is NOT using the scaling transformation correctly!")
        print("   Expected Y ~ -9m, but got Y ~", result['world_coords'][1])
    elif world_error < 1:
        print("\n✅ EXCELLENT: World coords match expected values!")
    else:
        print("\n⚠️ MODERATE DIFFERENCE: Some discrepancy in world coords")
else:
    print("\n❌ FAILED to process frame 90")
    print("   No ball detected or processing error")

print("\n" + "="*80)
print("If the world Y-coordinate is positive and large (like +25m or +35m):")
print("  → The scaling is NOT being applied!")
print("  → Need to restart Python/GUI completely")
print("  → Make sure no old .pyc files are cached")
print("="*80)
