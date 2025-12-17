import pandas as pd

# Load validation results
df = pd.read_csv('output/validation_180frames_results.csv')

# Get frame 143
f143 = df[df['frame'] == 143].iloc[0]

print("Frame 143 detection status:")
print(f"  Left camera detected:    {f143['detection_left_success']}")
print(f"  Right camera detected:   {f143['detection_right_success']}")
print(f"  Stereo match success:    {f143['stereo_match_success']}")
print(f"  Reconstruction success:  {f143['reconstruction_success']}")
print(f"\n  Processing time:         {f143['detection_time_ms']:.1f} ms")
print(f"  3D error:                {f143['3d_error_cm']:.1f} cm")

