"""
Check which validation file has the correct metrics
"""
import pandas as pd

print("=" * 60)
print("COMPREHENSIVE VALIDATION RESULTS:")
print("=" * 60)
df_comp = pd.read_csv('output/comprehensive_validation_results.csv')
errors_comp = df_comp[df_comp['reconstruction_success'] == True]['3d_error_cm'].dropna()
print(f"Total frames: {len(df_comp)}")
print(f"Success rate: {len(errors_comp)}/{len(df_comp)} = {len(errors_comp)/len(df_comp)*100:.1f}%")
print(f"Median error: {errors_comp.median():.2f} cm")
print(f"Mean error: {errors_comp.mean():.2f} cm")
print(f"95th percentile: {errors_comp.quantile(0.95):.2f} cm")

print("\n" + "=" * 60)
print("MKV VALIDATION RESULTS:")
print("=" * 60)
df_mkv = pd.read_csv('output/mkv_validation_results.csv')
errors_mkv = df_mkv[df_mkv['success'] == True]['error_cm'].dropna()
print(f"Total frames: {len(df_mkv)}")
print(f"Success rate: {len(errors_mkv)}/{len(df_mkv)} = {len(errors_mkv)/len(df_mkv)*100:.1f}%")
if len(errors_mkv) > 0:
    print(f"Median error: {errors_mkv.median():.2f} cm")
    print(f"Mean error: {errors_mkv.mean():.2f} cm")
    print(f"95th percentile: {errors_mkv.quantile(0.95):.2f} cm")
else:
    print("No successful frames!")

print("\n" + "=" * 60)
print("WHICH FILE MATCHES YOUR THESIS (3.8 cm median)?")
print("=" * 60)

