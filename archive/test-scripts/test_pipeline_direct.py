"""Test the hawkeye pipeline directly without GUI"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from hawkeye_pipeline import HawkeyePipeline
from camera_config import load_camera_config

print("="*60)
print("HAWKEYE PIPELINE TEST (like GUI does)")
print("="*60)

# Load config
config = load_camera_config()
print(f"\nConfig loaded:")
print(f"  Baseline: {config['baseline_m']}m")
print(f"  Z range: [{config['z_min_m']}, {config['z_max_m']}]m")
print(f"  Focal length: {config['focal_length_px']}px")

# Create pipeline
pipeline = HawkeyePipeline(config)

# Test frames 95-99
print(f"\n" + "="*60)
print("Testing frames 95-99:")
print("="*60)

success_count = 0
for frame_num in range(95, 100):
    print(f"\nProcessing frame {frame_num}...")
    result = pipeline.process_single_frame(frame_num)
    
    if result and isinstance(result, dict):
        print(f"  ✅ Success!")
        print(f"     Camera: {result['camera_coords']}")
        print(f"     World:  {result['world_coords']}")
        success_count += 1
    else:
        print(f"  ❌ Failed")

print(f"\n" + "="*60)
print(f"SUMMARY: {success_count}/5 frames successful")
print("="*60)

if success_count == 5:
    print("\n🎉 Pipeline working perfectly!")
    print("The Hawkeye Launcher GUI should now work correctly.")
else:
    print(f"\n⚠️  Only {success_count}/5 frames succeeded.")
    print("There may be additional issues with the pipeline.")
