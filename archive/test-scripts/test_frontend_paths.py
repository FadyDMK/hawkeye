"""Quick test to verify front_end.py can find images correctly"""
import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Simulate what front_end.py does
src_dir = os.path.join(os.path.dirname(__file__), 'src')
left_frames_dir = os.path.join(src_dir, "..", "output_frames", "left")
right_frames_dir = os.path.join(src_dir, "..", "output_frames", "right")

print("="*60)
print("FRONT_END PATH TEST")
print("="*60)

# Test frame 99
frame_num = 99
frame_id = f"{frame_num:04d}"

left_img_path = os.path.join(left_frames_dir, f"left3_{frame_id}.jpg")
right_img_path = os.path.join(right_frames_dir, f"right3_{frame_id}.jpg")

print(f"\nTesting frame {frame_num}:")
print(f"\nLeft path:  {left_img_path}")
print(f"  Exists: {os.path.exists(left_img_path)}")

print(f"\nRight path: {right_img_path}")
print(f"  Exists: {os.path.exists(right_img_path)}")

if os.path.exists(left_img_path) and os.path.exists(right_img_path):
    print(f"\n✅ Both images found! GUI should work now.")
else:
    print(f"\n❌ Missing images!")
    
    # Check what files actually exist
    import glob
    print(f"\nLeft frames found:")
    left_files = sorted(glob.glob(os.path.join(left_frames_dir, "left3_*.jpg")))[:5]
    for f in left_files:
        print(f"  {os.path.basename(f)}")
    
    print(f"\nRight frames found:")
    right_files = sorted(glob.glob(os.path.join(right_frames_dir, "right3_*.jpg")))[:5]
    for f in right_files:
        print(f"  {os.path.basename(f)}")
