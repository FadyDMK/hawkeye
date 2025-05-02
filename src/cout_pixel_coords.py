import cv2
import numpy as np
import os

root = os.path.dirname(os.path.abspath(__file__))
img = os.path.join(root, "..\\output_frames\\left\\left3_0001.jpg")

# List to store clicks
pixels = []

def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pixels.append((x, y))
        print(f"Clicked pixel: {(x,y)}")

# Load one frame (replace with your frame grab)
frame = img.copy()
orig_h, orig_w = frame.shape[:2]

# downscale to fit screen
new_w, new_h = 960, 540
resized = cv2.resize(frame, (new_w, new_h))

scale_x = new_w  / orig_w
scale_y = new_h  / orig_h

pixels = []
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pixels) < 4:
        # convert back to original
        xo = x / scale_x
        yo = y / scale_y
        pixels.append((xo, yo))
        print(f"Resized click: ({x},{y}) → Original: ({xo:.1f},{yo:.1f})")

cv2.namedWindow('resized')
cv2.setMouseCallback('resized', on_mouse)

while True:
    cv2.imshow('resized', resized)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(pixels) == 4:
        break
cv2.destroyAllWindows()

print("Original-coord points:", pixels)
