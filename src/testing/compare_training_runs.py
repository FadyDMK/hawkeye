import pandas as pd
import matplotlib.pyplot as plt
import os

# Define paths to the results.csv files
paths = {
    "Train 13": r"f:\hawkeye\models\runs\detect\train13\results.csv",
    "Train 18": r"f:\hawkeye\models\runs\detect\train18\results.csv",
    "Retrain Hard Frames 2": r"f:\hawkeye\models\runs\detect\retrain_with_hard_frames2\results.csv"
}

# Output directory for plots
output_dir = r"f:\hawkeye\output"
os.makedirs(output_dir, exist_ok=True)

# Metrics to compare (column names might have extra spaces, so we'll strip them)
metrics = {
    "mAP50": "metrics/mAP50(B)",
    "mAP50-95": "metrics/mAP50-95(B)",
    "Precision": "metrics/precision(B)",
    "Recall": "metrics/recall(B)"
}

dataframes = {}

# Read the CSV files
for name, path in paths.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        # Strip whitespace from column names
        df.columns = df.columns.str.strip()
        dataframes[name] = df
        print(f"Loaded {name} with {len(df)} epochs.")
    else:
        print(f"Error: File not found: {path}")

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(15, 10))
axes = axes.flatten()

for i, (metric_name, metric_col) in enumerate(metrics.items()):
    ax = axes[i]
    for name, df in dataframes.items():
        if metric_col in df.columns:
            ax.plot(df['epoch'], df[metric_col], label=name)
        else:
            print(f"Warning: Column '{metric_col}' not found in {name}")
    
    ax.set_title(f"{metric_name} over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(metric_name)
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plot_path = os.path.join(output_dir, "training_comparison_plots.png")
plt.savefig(plot_path)
print(f"Plots saved to {plot_path}")

# Print Summary Table
print("\n--- Best Performance Summary ---")
summary_data = []
for name, df in dataframes.items():
    row = {"Model": name}
    for metric_name, metric_col in metrics.items():
        if metric_col in df.columns:
            best_val = df[metric_col].max()
            row[metric_name] = best_val
    summary_data.append(row)

summary_df = pd.DataFrame(summary_data)
print(summary_df.to_string(index=False))

# Save summary to CSV
summary_csv_path = os.path.join(output_dir, "training_comparison_summary.csv")
summary_df.to_csv(summary_csv_path, index=False)
print(f"Summary saved to {summary_csv_path}")
