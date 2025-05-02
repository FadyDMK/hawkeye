import pandas as pd
import open3d as o3d
import time

df = pd.read_csv('interpolated_ball_positions.csv')
df = df.dropna(subset=["X", "Y", "Z"]).reset_index(drop=True)

#create visualizer window 
vis = o3d.visualization.Visualizer()
vis.create_window(window_name="Ball Path Animation", width=800, height=600)

#create ball
ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
ball.paint_uniform_color([1, 0, 0])  # Red color for the ball
vis.add_geometry(ball)

#add floor
floor = o3d.geometry.TriangleMesh.create_box(width=16, height=0.1, depth=9)
floor.translate([-8, -0.05, -4.5])
floor.paint_uniform_color([0.8, 0.8, 0.8])  # Grey color for the floor
#vis.add_geometry(floor)

for _,row in df.iterrows():
    x, y, z = row['X'], row['Y'], row['Z']
    
    #update ball position
    ball.translate((x, y, z) - ball.get_center(), relative=False)
    
    #update view
    vis.update_geometry(ball)
    vis.poll_events()
    vis.update_renderer()
    
    
    time.sleep(0.08)

vis.destroy_window()