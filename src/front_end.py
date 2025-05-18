import os
import tkinter as tk
from tkinter import ttk
import cv2
from PIL import Image, ImageTk
from numpy import isin
from pyvista import wrap
from hawkeye_pipeline import HawkeyePipeline

class FrameSelectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Frame Selector")
        self.root.geometry("1200x800")


        self.pipeline = HawkeyePipeline(None)

        src_dir = os.path.dirname(os.path.abspath(__file__))
        self.left_frames_dir = os.path.join(src_dir, "..", "output_frames", "left")

        import glob 
        self.frame_files = sorted(glob.glob(os.path.join(self.left_frames_dir, "left3_*.jpg")))
        self.total_frames = len(self.frame_files)

        if self.total_frames == 0:
            tk.Label(root, text="No frames found in the directory.").pack(pady=20)
            return
        
        self.current_frame = 0

        self.create_widgets()

        self.load_frame(0)
    
    def create_widgets(self):
        control_frame = ttk.Frame(self.root)
        control_frame.pack(fill="x",padx=10, pady=10)

        #Frame slider
        ttk.Label(control_frame, text="Frame:").pack(side="left", padx=(0, 10))
        self.frame_slider = ttk.Scale(
            control_frame,
            from_=0,
            to=self.total_frames - 1,
            orient="horizontal",
            length=400,
            command=self.on_slider_change
        )
        self.frame_slider.pack(side="left",fill="x",expand=True ,padx=(0, 10))

        # frame num display
        self.frame_label = ttk.Label(control_frame, text="0")
        self.frame_label.pack(side="left", padx=(0, 20))

        # Process button
        self.process_btn = ttk.Button(control_frame, text="Process Frame", command=self.process_current_frame)
        self.process_btn.pack(side="right", padx= 10)

        # visualize button
        self.visualize_btn = ttk.Button(control_frame, text="3D Visualize", command=self.visualize_3d)
        self.visualize_btn.pack(side="right", padx= 10)
        self.visualize_btn["state"] = "disabled"

        #frame display
        self.image_frame = ttk.Frame(self.root)
        self.image_frame.pack(fill="both", expand=True, padx=10, pady=10)

        #Left image
        self.left_img_label = ttk.Label(self.image_frame)
        self.left_img_label.pack(side="left", fill="both", expand=True)

        #Right image
        self.right_img_label = ttk.Label(self.image_frame)
        self.right_img_label.pack(side="right", fill="both", expand=True)

        #Results frame
        self.results_frame = ttk.LabelFrame(self.root, text="Processing Results")
        self.results_frame.pack(fill="x", padx=10, pady=10)

        #Results text
        self.results_text = tk.Text(self.results_frame, height=10, wrap="word")
        self.results_text.pack(fill="x", padx=10, pady=10)
    
    def on_slider_change(self, value):
        frame_num = int(float(value))
        if frame_num != self.current_frame:
            self.current_frame = frame_num
            self.frame_label.config(text=str(frame_num))
            self.load_frame(frame_num)
            self.visualize_btn["state"] = "disabled"
    
    def load_frame(self, frame_num):
        frame_id = f"{frame_num:04d}"

        #Load left image
        left_img_path = os.path.join(self.left_frames_dir, f"left3_{frame_id}.jpg")
        if os.path.exists(left_img_path):
            left_img = cv2.imread(left_img_path)
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB)
            left_img = cv2.resize(left_img, (600, 400))
            left_photo = ImageTk.PhotoImage(image=Image.fromarray(left_img))
            self.left_img_label.config(image=left_photo)
            self.left_img_label.image = left_photo

        #Load right image
        right_img_path = os.path.join(self.left_frames_dir, f"right3_{frame_id}.jpg")
        if os.path.exists(right_img_path):
            right_img = cv2.imread(right_img_path)
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB)
            right_img = cv2.resize(right_img, (600, 400))
            right_photo = ImageTk.PhotoImage(image=Image.fromarray(right_img))
            self.right_img_label.config(image=right_photo)
            self.right_img_label.image = right_photo
    
    def process_current_frame(self):
        #clear previous results 
        self.results_text.delete(1.0, tk.END)

        #show processing status
        self.results_text.insert(tk.END, f"Processing frame {self.current_frame}...\n")
        self.root.update()

        #process frame
        result = self.pipeline.process_single_frame(self.current_frame)

        if result and isinstance(result, dict):
            self.results_text.insert(tk.END, "Success!\n")
            self.results_text.insert(tk.END, f"Frame: {self.current_frame}\n")
            self.results_text.insert(tk.END, f"Camera coords: {result['camera_coords']}\n")
            self.results_text.insert(tk.END, f"World coords: {result['world_coords']}\n")
            self.visualize_btn["state"] = "normal"
        else:
            self.results_text.insert(tk.END, "Failed to process frame. No ball detected or stereo matching issue.\n")
            self.visualize_btn["state"] = "disabled"

    def visualize_3d(self):
        self.pipeline.visualize_results(type="3d")

if __name__ == "__main__":
    root = tk.Tk()
    app = FrameSelectorApp(root)
    root.mainloop()