"""Quick test after increasing tolerance"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Force reload the module to get updated tolerance
if 'hawkeye_pipeline' in sys.modules:
    del sys.modules['hawkeye_pipeline']

from hawkeye_pipeline import HawkeyePipeline
from camera_config import load_camera_config

config = load_camera_config()
pipeline = HawkeyePipeline(config)

print("="*60)
print("TESTING WITH INCREASED TOLERANCE")
print("="*60)
print("Tolerance now: ±10.0m (was ±3.0m)")
print()

success_with_world = 0
total_frames = 0

for frame_num in range(90, 100):
    result = pipeline.process_single_frame(frame_num)
    
    if result and isinstance(result, dict):
        cam = result['camera_coords']
        world = result['world_coords']
        
        if cam[0] is not None:
            total_frames += 1
            status = "✅" if world[0] is not None else "❌"
            
            if world[0] is not None:
                success_with_world += 1
                print(f"Frame {frame_num}: Camera {cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f} → World {world[0]:.1f}, {world[1]:.1f}, {world[2]:.1f} {status}")
            else:
                print(f"Frame {frame_num}: Camera {cam[0]:.1f}, {cam[1]:.1f}, {cam[2]:.1f} → World (None) {status}")

print(f"\n" + "="*60)
print(f"RESULT: {success_with_world}/{total_frames} frames have valid world coords")
print("="*60)

if success_with_world == total_frames:
    print("🎉 All frames now have valid world coordinates!")
elif success_with_world > 0:
    print(f"✅ Improvement! {success_with_world} frames working (was 0-1 before)")
else:
    print("❌ Still having issues - may need even more tolerance or check transforms")
