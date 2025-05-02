import pandas as pd
import math

blender_df = pd.read_csv("blender_distances.csv")              
pipeline_df = pd.read_csv("distances_from_pipeline.csv")      

# Rename columns for clarity
blender_df.rename(columns={"Distance": "Blender_Distance"}, inplace=True)
pipeline_df.rename(columns={"Distance": "Predicted_Distance"}, inplace=True)

# Merge on 'frame' (inner join to only keep frames present in both)
merged_df = pd.merge(blender_df, pipeline_df, on="Frame")

# Calculate errors
merged_df["Absolute_Error"] = abs(merged_df["Blender_Distance"] - merged_df["Predicted_Distance"])
merged_df["Relative_Error_%"] = (merged_df["Absolute_Error"] / merged_df["Blender_Distance"]) * 100

# Save the result
merged_df.to_csv("distance_comparison.csv", index=False)

# Print average relative error
mean_rel_error = merged_df["Relative_Error_%"].mean()
print(f"Comparison done. Average relative error: {mean_rel_error:.2f}%")

## Comparison done. Average relative error: 5.87%