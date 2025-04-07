from ultralytics import YOLO
import cv2
import os


def get_ball_xy(image_right, image_left):
    model = YOLO("../runs/detect/train13/weights/best.pt") 
    print("Model loaded")
    results_left = model(image_left, show=True, conf=0.5)
    results_right = model(image_right, show=True, conf=0.5)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls)
            if cls_id == 0:
                x_center, y_center = box.xywh[0][:2]
                return int(x_center), int(y_center)
    return None


## testing the function



print("Model loaded")



root = os.path.dirname(os.path.abspath(__file__))

left_path = os.path.join(root,"../output_frames/left/left3_0100.jpg")
right_path = os.path.join(root,"../output_frames/right/right3_0100.jpg")

if not os.path.exists(left_path) or not os.path.exists(right_path):
    raise FileNotFoundError("One or both image files do not exist. Please check the file paths.")

left = cv2.imread(left_path)
right = cv2.imread(right_path)
xy_left = get_ball_xy(left)
xy_right = get_ball_xy(right)
print(f"Ball coordinates in left image: {xy_left}")
print(f"Ball coordinates in right image: {xy_right}")