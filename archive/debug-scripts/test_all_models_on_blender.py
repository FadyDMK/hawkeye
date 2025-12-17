"""
Test all trained models on the Blender test dataset to find which performs best.
"""
from ultralytics import YOLO
import json
from pathlib import Path
import pandas as pd

# Define all models to test
models_to_test = {
    'train13': 'models/runs/detect/train13/weights/best.pt',
    'train18': 'models/runs/detect/train18/weights/best.pt',
    'train19_fast': 'models/runs/detect/train19_fast/weights/best.pt',
    'retrain_with_hard_frames2': 'models/runs/detect/retrain_with_hard_frames2/weights/best.pt',
    'train_finetune_fast416': 'models/runs/detect/train_finetune_fast416/weights/best.pt',
}

# Filter to only existing models
existing_models = {}
for name, path in models_to_test.items():
    full_path = Path(path)
    if full_path.exists():
        existing_models[name] = str(full_path)
        print(f"✓ Found: {name}")
    else:
        print(f"✗ Missing: {name}")

print(f"\nTesting {len(existing_models)} models on Blender test dataset...\n")

# Test each model
results = {}
data_yaml = 'dataset/data.yaml'

for name, path in existing_models.items():
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    try:
        model = YOLO(path)
        
        # Run validation on test set
        metrics = model.val(data=data_yaml, split='test', imgsz=640, batch=16, verbose=True)
        
        # Extract metrics
        results[name] = {
            'precision': float(metrics.box.p),
            'recall': float(metrics.box.r),
            'mAP50': float(metrics.box.map50),
            'mAP50-95': float(metrics.box.map),
            'model_path': path
        }
        
        print(f"\n✓ {name} Results:")
        print(f"  Precision: {results[name]['precision']*100:.2f}%")
        print(f"  Recall: {results[name]['recall']*100:.2f}%")
        print(f"  mAP50: {results[name]['mAP50']*100:.2f}%")
        print(f"  mAP50-95: {results[name]['mAP50-95']*100:.2f}%")
        
    except Exception as e:
        print(f"✗ Error testing {name}: {e}")
        results[name] = {'error': str(e)}

print(f"\n{'='*60}")
print("FINAL COMPARISON")
print(f"{'='*60}\n")

# Create comparison table
comparison_data = []
for name, metrics in results.items():
    if 'error' not in metrics:
        comparison_data.append({
            'Model': name,
            'Precision (%)': f"{metrics['precision']*100:.2f}",
            'Recall (%)': f"{metrics['recall']*100:.2f}",
            'mAP50 (%)': f"{metrics['mAP50']*100:.2f}",
            'mAP50-95 (%)': f"{metrics['mAP50-95']*100:.2f}"
        })

df = pd.DataFrame(comparison_data)
# Sort by mAP50 (descending)
df['mAP50_numeric'] = df['mAP50 (%)'].astype(float)
df = df.sort_values('mAP50_numeric', ascending=False)
df = df.drop('mAP50_numeric', axis=1)

print(df.to_string(index=False))

# Find best model
best_model = None
best_map50 = 0
for name, metrics in results.items():
    if 'error' not in metrics and metrics['mAP50'] > best_map50:
        best_map50 = metrics['mAP50']
        best_model = name

print(f"\n{'='*60}")
print(f"🏆 BEST MODEL: {best_model}")
print(f"   mAP50: {best_map50*100:.2f}%")
print(f"{'='*60}\n")

# Save results to JSON
output_file = 'blender_model_comparison.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"✓ Results saved to: {output_file}")

# Save comparison table to CSV
csv_file = 'blender_model_comparison.csv'
df.to_csv(csv_file, index=False)
print(f"✓ Comparison table saved to: {csv_file}")

print(f"\nRECOMMENDATION: Update camera_config.json to use {best_model}")
print(f"  detection_model_path: \"../models/runs/detect/{best_model}/weights/best.pt\"")
