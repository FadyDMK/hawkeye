"""
Blender script to extract ball positions for 180-frame validation footage.
Run this in Blender's scripting workspace.

Instructions:
1. Open your Blender project with the finalLeft/finalRight animation
2. Go to Scripting workspace (Scripting tab at the top)
3. Paste this script and click "Run Script" (Alt+P)
4. Check the console output (Window > Toggle System Console on Windows)
5. The script saves positions to: ball_positions_ground_truth_180.csv
"""

import bpy
import csv
import os

def extract_ball_positions():
    # Try to find the ball object
    ball_names = ['Volleyball', 'Ball', 'volleyball', 'ball', 'Sphere']
    ball = None
    
    for name in ball_names:
        if name in bpy.data.objects:
            ball = bpy.data.objects[name]
            print(f"✓ Found ball object: {name}")
            break
    
    if ball is None:
        print("ERROR: Could not find ball object!")
        print("Available objects:", list(bpy.data.objects.keys()))
        return
    
    # Get scene info
    scene = bpy.context.scene
    fps = scene.render.fps
    start_frame = scene.frame_start
    end_frame = scene.frame_end
    
    print(f"\n=== SCENE INFO ===")
    print(f"FPS: {fps}")
    print(f"Frame range: {start_frame} to {end_frame}")
    print(f"Total frames: {end_frame - start_frame + 1}")
    print(f"Ball object: {ball.name}")
    
    # Extract positions for each frame
    positions = []
    
    print(f"\n=== EXTRACTING BALL POSITIONS ===")
    for frame in range(start_frame, end_frame + 1):
        scene.frame_set(frame)
        
        # Get world space position
        world_pos = ball.matrix_world.translation
        x, y, z = world_pos.x, world_pos.y, world_pos.z
        
        positions.append({
            'frame': frame,
            'x': x,
            'y': y,
            'z': z
        })
        
        # Print progress every 20 frames
        if frame % 20 == 0 or frame == start_frame or frame == end_frame:
            print(f"  Frame {frame}: ({x:.4f}, {y:.4f}, {z:.4f})")
    
    # Save to CSV directly to the project's test-vids folder
    # CHANGE THIS PATH if your hawkeye project is in a different location
    hawkeye_project_path = r"F:\hawkeye"
    output_path = os.path.join(hawkeye_project_path, "test-vids", "ball_positions_ground_truth_180.csv")
    
    # Create test-vids directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for pos in positions:
            writer.writerow(pos)
    
    print(f"\n✓ Saved {len(positions)} positions to:")
    print(f"  {output_path}")
    print(f"\n✓ File is ready for validation - no need to copy manually!")
    
    # Print summary statistics
    xs = [p['x'] for p in positions]
    ys = [p['y'] for p in positions]
    zs = [p['z'] for p in positions]
    
    print(f"\n=== POSITION RANGES ===")
    print(f"X: {min(xs):.4f} to {max(xs):.4f} (range: {max(xs)-min(xs):.4f}m)")
    print(f"Y: {min(ys):.4f} to {max(ys):.4f} (range: {max(ys)-min(ys):.4f}m)")
    print(f"Z: {min(zs):.4f} to {max(zs):.4f} (range: {max(zs)-min(zs):.4f}m)")
    
    print(f"\n=== NEXT STEPS ===")
    print(f"1. Copy ball_positions_ground_truth_180.csv to F:/hawkeye/test-vids/")
    print(f"2. Run the validation script to compare against Hawkeye output")
    
    return output_path

# Run the extraction
extract_ball_positions()
