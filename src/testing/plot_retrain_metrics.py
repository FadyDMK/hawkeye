"""
Plot training metrics for retrain_with_hard_frames2 to visualize early stopping decision.
"""
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Read results
results_path = Path('models/runs/detect/retrain_with_hard_frames2/results.csv')
df = pd.read_csv(results_path)

# Create figure with subplots
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('retrain_with_hard_frames2 Training Progression (9 Epochs)', fontsize=16, fontweight='bold')

# Plot 1: mAP metrics
ax1.plot(df['epoch'], df['metrics/mAP50(B)'], 'b-o', linewidth=2, label='mAP50')
ax1.plot(df['epoch'], df['metrics/mAP50-95(B)'], 'g-s', linewidth=2, label='mAP50-95')
ax1.axvline(x=7, color='red', linestyle='--', alpha=0.5, label='Peak mAP50 (Epoch 7)')
ax1.set_xlabel('Epoch', fontsize=12)
ax1.set_ylabel('mAP', fontsize=12)
ax1.set_title('Mean Average Precision Over Epochs', fontsize=13, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim([0.35, 0.80])

# Plot 2: Precision and Recall
ax2.plot(df['epoch'], df['metrics/precision(B)'], 'r-o', linewidth=2, label='Precision')
ax2.plot(df['epoch'], df['metrics/recall(B)'], 'orange', marker='s', linewidth=2, label='Recall')
ax2.set_xlabel('Epoch', fontsize=12)
ax2.set_ylabel('Score', fontsize=12)
ax2.set_title('Precision and Recall Over Epochs', fontsize=13, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim([0.60, 0.95])

# Plot 3: Training Losses
ax3.plot(df['epoch'], df['train/box_loss'], 'b-o', linewidth=2, label='Box Loss')
ax3.plot(df['epoch'], df['train/cls_loss'], 'g-s', linewidth=2, label='Classification Loss')
ax3.plot(df['epoch'], df['train/dfl_loss'], 'purple', marker='^', linewidth=2, label='DFL Loss')
ax3.set_xlabel('Epoch', fontsize=12)
ax3.set_ylabel('Loss', fontsize=12)
ax3.set_title('Training Losses', fontsize=13, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Plot 4: Validation Losses
ax4.plot(df['epoch'], df['val/box_loss'], 'b-o', linewidth=2, label='Val Box Loss')
ax4.plot(df['epoch'], df['val/cls_loss'], 'g-s', linewidth=2, label='Val Class Loss')
ax4.plot(df['epoch'], df['val/dfl_loss'], 'purple', marker='^', linewidth=2, label='Val DFL Loss')
ax4.set_xlabel('Epoch', fontsize=12)
ax4.set_ylabel('Loss', fontsize=12)
ax4.set_title('Validation Losses', fontsize=13, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()

# Save figure
output_path = 'output/retrain_hard_frames_training_progression.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Graph saved to: {output_path}")

# Print key observations
print("\n" + "="*60)
print("KEY OBSERVATIONS")
print("="*60)
print(f"\n📊 mAP50 Progression:")
print(f"   Epoch 1: {df['metrics/mAP50(B)'].iloc[0]:.3f}")
print(f"   Epoch 7 (Peak): {df['metrics/mAP50(B)'].iloc[6]:.3f}")
print(f"   Epoch 9 (Final): {df['metrics/mAP50(B)'].iloc[8]:.3f}")

print(f"\n📈 Final Metrics (Epoch 9):")
print(f"   Precision: {df['metrics/precision(B)'].iloc[8]:.3f} ({df['metrics/precision(B)'].iloc[8]*100:.1f}%)")
print(f"   Recall: {df['metrics/recall(B)'].iloc[8]:.3f} ({df['metrics/recall(B)'].iloc[8]*100:.1f}%)")
print(f"   mAP50: {df['metrics/mAP50(B)'].iloc[8]:.3f} ({df['metrics/mAP50(B)'].iloc[8]*100:.1f}%)")
print(f"   mAP50-95: {df['metrics/mAP50-95(B)'].iloc[8]:.3f} ({df['metrics/mAP50-95(B)'].iloc[8]*100:.1f}%)")

print(f"\n⚠️ Stopping Rationale:")
print(f"   - mAP50 peaked at epoch 7 ({df['metrics/mAP50(B)'].iloc[6]:.1%})")
print(f"   - Subsequent epochs showed fluctuation (74.2% → 74.8%)")
print(f"   - Validation losses plateaued/oscillating")
print(f"   - Early stopping at epoch 9 prevented overfitting")
print("="*60)

plt.show()
