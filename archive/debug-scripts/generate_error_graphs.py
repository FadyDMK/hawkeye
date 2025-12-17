import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load validation results
df = pd.read_csv('output/validation_180frames_results.csv')

# Filter only successful reconstructions
valid = df[df['reconstruction_success'] == True].copy()

print(f"Valid reconstructions: {len(valid)} out of {len(df)} frames")
print(f"Error range: {valid['3d_error_cm'].min():.1f} - {valid['3d_error_cm'].max():.1f} cm")
print(f"Median error: {valid['3d_error_cm'].median():.1f} cm")
print(f"Mean error: {valid['3d_error_cm'].mean():.1f} cm")

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('180-Frame Validation Error Analysis (finalLeft/finalRight)', fontsize=14, fontweight='bold')

# 1. Error per frame (line plot)
ax1 = axes[0, 0]
ax1.plot(valid['frame'], valid['3d_error_cm'], 'b-', linewidth=1, alpha=0.7)
ax1.axhline(y=valid['3d_error_cm'].median(), color='r', linestyle='--', label=f'Median: {valid["3d_error_cm"].median():.1f} cm')
ax1.axhline(y=valid['3d_error_cm'].mean(), color='orange', linestyle='--', label=f'Mean: {valid["3d_error_cm"].mean():.1f} cm')
ax1.set_xlabel('Frame Number')
ax1.set_ylabel('3D Error (cm)')
ax1.set_title('Error per Frame')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 2. Error histogram
ax2 = axes[0, 1]
ax2.hist(valid['3d_error_cm'], bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax2.axvline(x=valid['3d_error_cm'].median(), color='r', linestyle='--', linewidth=2, label=f'Median: {valid["3d_error_cm"].median():.1f} cm')
ax2.axvline(x=valid['3d_error_cm'].mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean: {valid["3d_error_cm"].mean():.1f} cm')
ax2.set_xlabel('3D Error (cm)')
ax2.set_ylabel('Frequency')
ax2.set_title('Error Distribution (Histogram)')
ax2.legend()
ax2.grid(True, alpha=0.3)

# 3. Cumulative distribution
ax3 = axes[1, 0]
sorted_errors = np.sort(valid['3d_error_cm'])
cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
ax3.plot(sorted_errors, cumulative, 'g-', linewidth=2)
ax3.axvline(x=valid['3d_error_cm'].median(), color='r', linestyle='--', label=f'Median: {valid["3d_error_cm"].median():.1f} cm')
ax3.axhline(y=50, color='gray', linestyle=':', alpha=0.5)
ax3.axhline(y=95, color='gray', linestyle=':', alpha=0.5)
percentile_95 = np.percentile(valid['3d_error_cm'], 95)
ax3.axvline(x=percentile_95, color='orange', linestyle='--', label=f'95th: {percentile_95:.1f} cm')
ax3.set_xlabel('3D Error (cm)')
ax3.set_ylabel('Cumulative Percentage (%)')
ax3.set_title('Cumulative Error Distribution')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 4. Box plot with outliers
ax4 = axes[1, 1]
bp = ax4.boxplot(valid['3d_error_cm'], vert=True, patch_artist=True, widths=0.5)
bp['boxes'][0].set_facecolor('lightblue')
bp['medians'][0].set_color('red')
bp['medians'][0].set_linewidth(2)
ax4.set_ylabel('3D Error (cm)')
ax4.set_title('Error Distribution (Box Plot)')
ax4.grid(True, alpha=0.3, axis='y')

# Add statistics text
stats_text = f"""Statistics:
Frames: {len(valid)}/{len(df)}
Median: {valid['3d_error_cm'].median():.1f} cm
Mean: {valid['3d_error_cm'].mean():.1f} cm
Std Dev: {valid['3d_error_cm'].std():.1f} cm
Min: {valid['3d_error_cm'].min():.1f} cm
Max: {valid['3d_error_cm'].max():.1f} cm
95th %ile: {percentile_95:.1f} cm"""
ax4.text(1.5, valid['3d_error_cm'].median(), stats_text, fontsize=9, 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig('output/error_distribution_180frames.png', dpi=150, bbox_inches='tight')
print("\n✓ Saved graph to: output/error_distribution_180frames.png")
plt.close()

# Create a second figure showing frames with highest errors
fig2, ax = plt.subplots(figsize=(12, 6))
top_errors = valid.nlargest(20, '3d_error_cm')
colors = ['red' if e > 500 else 'orange' if e > 200 else 'yellow' for e in top_errors['3d_error_cm']]
ax.bar(range(len(top_errors)), top_errors['3d_error_cm'], color=colors, edgecolor='black')
ax.set_xticks(range(len(top_errors)))
ax.set_xticklabels(top_errors['frame'].astype(int), rotation=45)
ax.set_xlabel('Frame Number')
ax.set_ylabel('3D Error (cm)')
ax.set_title('Top 20 Frames with Highest Errors')
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(y=500, color='red', linestyle='--', alpha=0.5, label='500 cm threshold')
ax.axhline(y=200, color='orange', linestyle='--', alpha=0.5, label='200 cm threshold')
ax.legend()
plt.tight_layout()
plt.savefig('output/top_errors_180frames.png', dpi=150, bbox_inches='tight')
print("✓ Saved graph to: output/top_errors_180frames.png")
plt.close()

print(f"\nTop 5 worst frames:")
for idx, row in valid.nlargest(5, '3d_error_cm').iterrows():
    print(f"  Frame {int(row['frame'])}: {row['3d_error_cm']:.1f} cm")

