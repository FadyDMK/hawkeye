import pandas as pd
import os

# Final metrics from validation
results = {
    'Model': ['Train 13', 'Train 18', 'Retrain Hard Frames 2'],
    'Precision': [0.3482, 0.1095, 0.1065],
    'Recall': [0.0879, 0.0879, 0.0879],
    'mAP50': [0.0584, 0.0397, 0.0295],
    'mAP50-95': [0.0141, 0.0188, 0.0117]
}

df = pd.DataFrame(results)

# Save to output directory
output_path = r"f:\hawkeye\output\blender_model_comparison.csv"
df.to_csv(output_path, index=False)

print(f"Saved metrics to {output_path}")
print("\n" + df.to_string(index=False))
