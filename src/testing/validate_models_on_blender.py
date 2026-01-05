from ultralytics import YOLO
import os

models = {
    "Train 13": r"f:\hawkeye\models\runs\detect\train13\weights\best.pt",
    "Train 18": r"f:\hawkeye\models\runs\detect\train18\weights\best.pt",
    "Retrain Hard Frames 2": r"f:\hawkeye\models\runs\detect\retrain_with_hard_frames2\weights\best.pt"
}

print("Model,Precision,Recall,mAP50,mAP50-95")

for name, path in models.items():
    try:
        model = YOLO(path)
        # Run validation on the blender dataset
        # We use split='test' because we put everything in 'images' and defined test=images in yaml
        metrics = model.val(data=r"f:\hawkeye\dataset\blender_val\data.yaml", split='val', verbose=False)
        
        print(f"{name},{metrics.box.mp:.4f},{metrics.box.mr:.4f},{metrics.box.map50:.4f},{metrics.box.map:.4f}")
    except Exception as e:
        print(f"Error evaluating {name}: {e}")
