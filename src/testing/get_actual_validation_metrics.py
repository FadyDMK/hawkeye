"""
Get the actual YOLO validation metrics by running model.val() on the standard validation set
This should reproduce the metrics referenced in the thesis
"""
from ultralytics import YOLO
import pandas as pd

MODELS = {
    "Train 13": r"f:\hawkeye\models\runs\detect\train13\weights\best.pt",
    "Train 18": r"f:\hawkeye\models\runs\detect\train18\weights\best.pt",
    "Retrain Hard Frames 2": r"f:\hawkeye\models\runs\detect\retrain_with_hard_frames2\weights\best.pt"
}

DATA_YAML = r"f:\hawkeye\dataset\data.yaml"

results = []

for name, path in MODELS.items():
    print(f"\nValidating {name}...")
    model = YOLO(path)
    metrics = model.val(data=DATA_YAML, split='val', verbose=True)
    
    result = {
        "Model": name,
        "Precision": f"{metrics.box.mp:.3f}",
        "Recall": f"{metrics.box.mr:.3f}",
        "mAP50": f"{metrics.box.map50:.3f}",
        "mAP50-95": f"{metrics.box.map:.3f}"
    }
    results.append(result)
    print(f"  Precision: {metrics.box.mp:.3f}, Recall: {metrics.box.mr:.3f}, mAP50: {metrics.box.map50:.3f}, mAP50-95: {metrics.box.map:.3f}")

df = pd.DataFrame(results)
output_path = r"f:\hawkeye\output\model_validation_metrics.csv"
df.to_csv(output_path, index=False)

print(f"\n{'='*60}")
print(df.to_string(index=False))
print(f"\nSaved to {output_path}")
