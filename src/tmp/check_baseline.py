"""Calculate what baseline gives correct depth"""
import pandas as pd
import math

df = pd.read_csv('ball_positions_blender.csv')
cam_pos = (18, -6, 3)
focal = 1600
disparities = {90: 188, 93: 186, 96: 186, 99: 183}

print('Calculating required baseline for correct depth:\n')

baselines = []
for frame, disp in disparities.items():
    gt = df[df['frame']==frame].iloc[0]
    dist = math.sqrt((gt.x-cam_pos[0])**2 + (gt.y-cam_pos[1])**2 + (gt.z-cam_pos[2])**2)
    baseline_needed = (dist * disp) / focal
    baselines.append(baseline_needed)
    print(f'Frame {frame}: dist={dist:.2f}m, disp={disp}px → baseline={baseline_needed:.3f}m')

avg_baseline = sum(baselines) / len(baselines)
print(f'\nAverage required baseline: {avg_baseline:.3f}m')
print(f'We are currently using: 1.442m')
print(f'Actual camera Y-separation: 3.0m')
print(f'\nRatio: {avg_baseline / 1.442:.3f}')
