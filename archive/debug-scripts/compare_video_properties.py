import cv2

print("="*70)
print("VIDEO PROPERTIES COMPARISON")
print("="*70)

videos = [
    ("newnewLeft.mkv (100-frame validation)", "test-vids/newnewLeft.mkv"),
    ("newnewRight.mkv (100-frame validation)", "test-vids/newnewRight.mkv"),
    ("finalLeft.mkv (180-frame validation)", "test-vids/finalLeft.mkv"),
    ("finalRight.mkv (180-frame validation)", "test-vids/finalRight.mkv"),
]

for name, path in videos:
    try:
        cap = cv2.VideoCapture(path)
        
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\n{name}:")
        print(f"  Resolution: {width}x{height}")
        print(f"  FPS: {fps}")
        print(f"  Frame count: {frame_count}")
        
        cap.release()
    except Exception as e:
        print(f"\n{name}: ERROR - {e}")

print("\n" + "="*70)
print("CAMERA CONFIG (camera_config.json)")
print("="*70)

try:
    import json
    with open('src/camera_config.json', 'r') as f:
        config = json.load(f)
    
    print(f"\nFocal length: {config.get('focal_length_px')} px")
    print(f"Baseline: {config.get('baseline_m')} m")
    print(f"Resolution: {config.get('resolution_width')}x{config.get('resolution_height')}")
    print(f"Court center: ({config.get('court_center_x')}, {config.get('court_center_y')}, {config.get('court_center_z')})")
except Exception as e:
    print(f"\nERROR reading config: {e}")

print("\n" + "="*70)
print("Next: Run the Blender script to extract camera settings from the scene")
print("="*70)

