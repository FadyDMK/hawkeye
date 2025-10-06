"""
Blender script to extract actual ball positions from animation.
Run this in Blender's scripting workspace.

Instructions:
1. Open your Blender project with the volleyball animation
2. Go to Scripting workspace
3. Paste this script and click "Run Script"
4. Check the output in the console (Window > Toggle System Console)
5. The script will also save positions to a CSV file
"""

import bpy
import csv
import os

def extract_ball_positions():
    # Try to find the ball object (adjust the name if needed)
    ball_names = ['Volleyball', 'Ball', 'volleyball', 'ball', 'Sphere']
    ball = None
    
    for name in ball_names:
        if name in bpy.data.objects:
            ball = bpy.data.objects[name]
            print(f"Found ball object: {name}")
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
    
    print(f"\nScene info:")
    print(f"  FPS: {fps}")
    print(f"  Frame range: {start_frame} to {end_frame}")
    print(f"  Ball object: {ball.name}")
    
    # Extract positions for each frame
    positions = []
    
    print(f"\nExtracting ball positions...")
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
        
        if frame % 10 == 0 or frame in [90, 93, 96, 99]:
            print(f"  Frame {frame}: ({x:.4f}, {y:.4f}, {z:.4f})")
    
    # Save to CSV
    output_path = os.path.join(os.path.dirname(bpy.data.filepath), "ball_positions_blender.csv")
    
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['frame', 'x', 'y', 'z']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for pos in positions:
            writer.writerow(pos)
    
    print(f"\n✓ Saved {len(positions)} positions to: {output_path}")
    
    # Print summary statistics
    xs = [p['x'] for p in positions]
    ys = [p['y'] for p in positions]
    zs = [p['z'] for p in positions]
    
    print(f"\nPosition ranges:")
    print(f"  X: {min(xs):.4f} to {max(xs):.4f}")
    print(f"  Y: {min(ys):.4f} to {max(ys):.4f}")
    print(f"  Z: {min(zs):.4f} to {max(zs):.4f}")
    
    # Print specific frames we tested (90, 93, 96, 99)
    print(f"\nTest frames (for comparison with Hawkeye):")
    for frame_num in [90, 93, 96, 99]:
        if start_frame <= frame_num <= end_frame:
            pos = positions[frame_num - start_frame]
            print(f"  Frame {frame_num}: X={pos['x']:.4f}, Y={pos['y']:.4f}, Z={pos['z']:.4f}")
    
    # Also print camera positions for reference
    print(f"\nCamera positions (for reference):")
    for obj in bpy.data.objects:
        if obj.type == 'CAMERA':
            pos = obj.matrix_world.translation
            print(f"  {obj.name}: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
    
    # Print court center if it exists
    print(f"\nCourt/Ground object search:")
    court_objects = ['Court', 'court', 'Ground', 'ground', 'Plane', 'plane', 'Floor', 'floor']
    found_court = False
    for name in court_objects:
        if name in bpy.data.objects:
            court = bpy.data.objects[name]
            pos = court.matrix_world.translation
            print(f"  {court.name} (court): ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")
            found_court = True
            break
    
    if not found_court:
        print("  No court object found with standard names.")
        print("  All objects in scene:", list(bpy.data.objects.keys())[:20])
        print("\n⚠️  IMPORTANT: What is the court center position in your Blender scene?")
        print("     This is needed to correctly transform coordinates!")

# Run the extraction
extract_ball_positions()
