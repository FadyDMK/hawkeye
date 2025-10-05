import cv2
import sys
import os
from datetime import datetime

sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
from volleyball_detection import get_ball_xy

# Find the actual newest frame (not detection outputs)
left_dir = "output_frames/left"
frames = [f for f in os.listdir(left_dir) 
          if f.endswith('.jpg') 
          and 'DETECTED' not in f 
          and 'FAILED' not in f]

# Sort by modification time
frames_with_time = [(f, os.path.getmtime(os.path.join(left_dir, f))) for f in frames]
frames_with_time.sort(key=lambda x: x[1], reverse=True)

print("="*60)
print("NEWEST FRAME DETECTION TEST")
print("="*60)

if not frames_with_time:
    print("❌ No frame files found!")
    print("Check if you rendered new frames to output_frames/left")
    exit()

# Get the 3 newest frames
newest_frames = frames_with_time[:3]

print(f"\nFound {len(frames)} total frames")
print(f"\nTesting 3 newest frames:\n")

for i, (frame_name, mtime) in enumerate(newest_frames):
    mod_time = datetime.fromtimestamp(mtime)
    print(f"{i+1}. {frame_name}")
    print(f"   Modified: {mod_time}")
    print(f"   Age: {(datetime.now() - mod_time).total_seconds()/60:.1f} minutes ago")
    
print("\n" + "="*60)

for frame_name, mtime in newest_frames:
    frame_path = os.path.join(left_dir, frame_name)
    
    print(f"\nTesting: {frame_name}")
    print("-" * 40)
    
    # Load and check image
    img = cv2.imread(frame_path)
    if img is None:
        print("❌ Could not load image!")
        continue
    
    h, w = img.shape[:2]
    print(f"Resolution: {w}x{h}")
    
    # Check if image is mostly black/white
    mean_val = img.mean()
    print(f"Mean brightness: {mean_val:.1f} (0=black, 255=white)")
    
    if mean_val < 10:
        print("⚠️  Image is very dark - might be rendering issue")
    elif mean_val > 245:
        print("⚠️  Image is very bright - might be overexposed")
    
    # Try detection
    print("Running YOLO detection...")
    x, y = get_ball_xy(img)
    
    if x is not None and y is not None:
        print(f"✅ Ball detected at ({x}, {y})")
        
        # Check if ball is in reasonable position
        if x < 0 or x >= w or y < 0 or y >= h:
            print(f"⚠️  Ball position is outside frame bounds!")
        else:
            print(f"   Position looks valid (within {w}x{h} frame)")
            
    else:
        print(f"❌ NO BALL DETECTED")
        
        # Try with very low confidence
        print("   Trying with confidence=0.05...")
        x_low, y_low = get_ball_xy(img, conf=0.05)
        
        if x_low is not None:
            print(f"   Found with low conf at ({x_low}, {y_low})")
            print(f"   → Ball exists but model confidence is low")
            print(f"   → Check: ball color, lighting, background contrast")
        else:
            print(f"   Still not found even at 0.05 confidence")
            print(f"   → Ball might not look like a volleyball")
            print(f"   → Or ball is not in this frame")
            
            # Save image for manual inspection
            import shutil
            inspect_path = f"INSPECT_{frame_name}"
            shutil.copy(frame_path, inspect_path)
            print(f"   → Saved copy for inspection: {inspect_path}")
            print(f"   → Please open this image and check:")
            print(f"      1. Is there a ball visible?")
            print(f"      2. What color is it?")
            print(f"      3. Is it round and clear?")
            print(f"      4. Does it look like a volleyball?")

print("\n" + "="*60)
print("RECOMMENDATION:")
print("If ball is NOT being detected:")
print("1. Open INSPECT_*.jpg files to see what the ball looks like")
print("2. Check if ball has volleyball texture/pattern")
print("3. Verify ball is visible and not blended with background")
print("4. Consider: white/bright ball on white/bright background = invisible")
print("="*60)
