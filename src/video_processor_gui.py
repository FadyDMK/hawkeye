import os
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import threading
from hawkeye_pipeline import HawkeyePipeline
from camera_config import load_camera_config

class VideoProcessorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Hawkeye - Video Processor")
        self.root.state('zoomed')  # Maximize window on Windows
        
        self.config = load_camera_config()
        self.pipeline = HawkeyePipeline(self.config)
        
        self.processing = False
        self.left_offset = 0
        self.right_offset = 0
        self.create_widgets()
        
    def create_widgets(self):
        # Create main canvas with scrollbar
        main_canvas = tk.Canvas(self.root)
        scrollbar = ttk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = ttk.Frame(main_canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        main_canvas.pack(side="left", fill="both", expand=True, padx=(20, 0), pady=20)
        scrollbar.pack(side="right", fill="y", pady=20)
        
        # Enable mousewheel scrolling
        def on_mousewheel(event):
            main_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        main_canvas.bind_all("<MouseWheel>", on_mousewheel)
        
        # Now use scrollable_frame as parent for all widgets
        # Title
        title_label = ttk.Label(scrollable_frame, text="Process Complete Video", font=("Arial", 16, "bold"))
        title_label.pack(pady=10)
        
        # Video file selection
        video_frame = ttk.LabelFrame(scrollable_frame, text="Video Files", padding=10)
        video_frame.pack(fill="x", pady=10)
        
        # Left video
        left_frame = ttk.Frame(video_frame)
        left_frame.pack(fill="x", pady=5)
        ttk.Label(left_frame, text="Left Video:").pack(side="left", padx=(0, 10))
        self.left_video_var = tk.StringVar(value="Use pre-extracted frames")
        ttk.Entry(left_frame, textvariable=self.left_video_var, width=40).pack(side="left", fill="x", expand=True)
        ttk.Button(left_frame, text="Browse", command=lambda: self.browse_video("left")).pack(side="left", padx=(5, 0))
        
        # Right video
        right_frame = ttk.Frame(video_frame)
        right_frame.pack(fill="x", pady=5)
        ttk.Label(right_frame, text="Right Video:").pack(side="left", padx=(0, 10))
        self.right_video_var = tk.StringVar(value="Use pre-extracted frames")
        ttk.Entry(right_frame, textvariable=self.right_video_var, width=40).pack(side="left", fill="x", expand=True)
        ttk.Button(right_frame, text="Browse", command=lambda: self.browse_video("right")).pack(side="left", padx=(5, 0))
        
        # Synchronization controls
        sync_frame = ttk.LabelFrame(scrollable_frame, text="Video Synchronization", padding=10)
        sync_frame.pack(fill="x", pady=10)
        
        # Preview frame slider
        preview_slider_frame = ttk.Frame(sync_frame)
        preview_slider_frame.pack(fill="x", pady=5)
        ttk.Label(preview_slider_frame, text="Preview Frame:").pack(side="left", padx=(0, 10))
        self.preview_frame_var = tk.IntVar(value=0)
        self.preview_slider = ttk.Scale(
            preview_slider_frame,
            from_=0,
            to=100,
            orient="horizontal",
            variable=self.preview_frame_var,
            command=self.on_preview_change
        )
        self.preview_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.preview_frame_label = ttk.Label(preview_slider_frame, text="0", width=5)
        self.preview_frame_label.pack(side="left")
        ttk.Button(preview_slider_frame, text="Load Preview", command=self.load_preview).pack(side="left", padx=(10, 0))
        
        # Left video offset
        left_sync_frame = ttk.Frame(sync_frame)
        left_sync_frame.pack(fill="x", pady=2)
        ttk.Label(left_sync_frame, text="Left Offset:").pack(side="left", padx=(0, 10))
        self.left_offset_var = tk.IntVar(value=0)
        self.left_offset_slider = ttk.Scale(
            left_sync_frame,
            from_=-100,
            to=100,
            orient="horizontal",
            variable=self.left_offset_var,
            command=self.on_sync_change
        )
        self.left_offset_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.left_offset_label = ttk.Label(left_sync_frame, text="0", width=5)
        self.left_offset_label.pack(side="left")
        ttk.Button(left_sync_frame, text="Reset", command=lambda: self.reset_offset('left')).pack(side="left", padx=(5, 0))
        
        # Right video offset
        right_sync_frame = ttk.Frame(sync_frame)
        right_sync_frame.pack(fill="x", pady=2)
        ttk.Label(right_sync_frame, text="Right Offset:").pack(side="left", padx=(0, 10))
        self.right_offset_var = tk.IntVar(value=0)
        self.right_offset_slider = ttk.Scale(
            right_sync_frame,
            from_=-100,
            to=100,
            orient="horizontal",
            variable=self.right_offset_var,
            command=self.on_sync_change
        )
        self.right_offset_slider.pack(side="left", fill="x", expand=True, padx=(0, 10))
        self.right_offset_label = ttk.Label(right_sync_frame, text="0", width=5)
        self.right_offset_label.pack(side="left")
        ttk.Button(right_sync_frame, text="Reset", command=lambda: self.reset_offset('right')).pack(side="left", padx=(5, 0))
        
        # Preview images frame
        preview_images_frame = ttk.Frame(sync_frame)
        preview_images_frame.pack(fill="both", expand=True, pady=10)
        
        self.left_preview_label = ttk.Label(preview_images_frame, text="Left Preview\n(Load videos first)")
        self.left_preview_label.pack(side="left", fill="both", expand=True, padx=5)
        
        self.right_preview_label = ttk.Label(preview_images_frame, text="Right Preview\n(Load videos first)")
        self.right_preview_label.pack(side="right", fill="both", expand=True, padx=5)
        
        # Frame range selection
        range_frame = ttk.LabelFrame(scrollable_frame, text="Frame Range", padding=10)
        range_frame.pack(fill="x", pady=10)
        
        # Start frame
        start_frame = ttk.Frame(range_frame)
        start_frame.pack(fill="x", pady=5)
        ttk.Label(start_frame, text="Start Frame:").pack(side="left", padx=(0, 10))
        self.start_var = tk.IntVar(value=0)
        ttk.Entry(start_frame, textvariable=self.start_var, width=10).pack(side="left")
        
        # End frame
        end_frame = ttk.Frame(range_frame)
        end_frame.pack(fill="x", pady=5)
        ttk.Label(end_frame, text="End Frame:").pack(side="left", padx=(0, 10))
        self.end_var = tk.StringVar(value="All")
        ttk.Entry(end_frame, textvariable=self.end_var, width=10).pack(side="left")
        ttk.Label(end_frame, text="(Leave 'All' for entire video)", font=("Arial", 9)).pack(side="left", padx=(10, 0))
        
        # Progress section
        progress_frame = ttk.LabelFrame(scrollable_frame, text="Progress", padding=10)
        progress_frame.pack(fill="both", expand=True, pady=10)
        
        # Progress label
        self.progress_label = ttk.Label(progress_frame, text="Ready to process", font=("Arial", 10))
        self.progress_label.pack(pady=5)
        
        # Progress bar with determinate mode
        self.progress_bar = ttk.Progressbar(progress_frame, mode="determinate", maximum=100)
        self.progress_bar.pack(fill="x", pady=10)
        
        self.status_text = tk.Text(progress_frame, height=10, wrap="word", state="disabled")
        self.status_text.pack(fill="both", expand=True)
        
        # Buttons
        button_frame = ttk.Frame(scrollable_frame)
        button_frame.pack(fill="x", pady=10)
        
        self.process_btn = ttk.Button(button_frame, text="Start Processing", command=self.start_processing)
        self.process_btn.pack(side="left", padx=5)
        
        self.export_btn = ttk.Button(button_frame, text="Export Results", command=self.export_results, state="disabled")
        self.export_btn.pack(side="left", padx=5)
        
        self.visualize_btn = ttk.Button(button_frame, text="Visualize 3D", command=self.visualize_3d, state="disabled")
        self.visualize_btn.pack(side="left", padx=5)
        
        ttk.Button(button_frame, text="Close", command=self.root.destroy).pack(side="right", padx=5)
    
    def browse_video(self, side):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title=f"Select {side} video",
            filetypes=[("Video files", "*.mkv *.mp4 *.avi"), ("All files", "*.*")]
        )
        if filename:
            if side == "left":
                self.left_video_var.set(filename)
            else:
                self.right_video_var.set(filename)
    
    def load_preview(self):
        """Load preview frames from videos"""
        left_video = self.left_video_var.get()
        right_video = self.right_video_var.get()
        
        # Check if using videos
        if left_video == "Use pre-extracted frames" or right_video == "Use pre-extracted frames":
            messagebox.showinfo("Info", "Please select both video files first")
            return
        
        if not os.path.exists(left_video) or not os.path.exists(right_video):
            messagebox.showerror("Error", "Video files not found")
            return
        
        try:
            import cv2
            
            # Open video captures
            self.left_cap = cv2.VideoCapture(left_video)
            self.right_cap = cv2.VideoCapture(right_video)
            
            if not self.left_cap.isOpened() or not self.right_cap.isOpened():
                messagebox.showerror("Error", "Failed to open video files")
                return
            
            # Update preview slider max
            total_frames = int(self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.preview_slider.config(to=total_frames - 1)
            
            # Show first frame
            self.update_preview()
            
            messagebox.showinfo("Success", f"Videos loaded! Total frames: {total_frames}\nUse preview slider to find sync point")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load videos: {str(e)}")
    
    def reset_offset(self, side):
        """Reset offset for left or right video"""
        if side == 'left':
            self.left_offset_var.set(0)
            self.left_offset = 0
            self.left_offset_label.config(text="0")
        else:
            self.right_offset_var.set(0)
            self.right_offset = 0
            self.right_offset_label.config(text="0")
        self.update_preview()
    
    def on_sync_change(self, value):
        """Handle synchronization offset changes"""
        self.left_offset = int(self.left_offset_var.get())
        self.right_offset = int(self.right_offset_var.get())
        
        self.left_offset_label.config(text=str(self.left_offset))
        self.right_offset_label.config(text=str(self.right_offset))
        
        self.update_preview()
    
    def on_preview_change(self, value):
        """Handle preview slider change"""
        frame_num = int(float(value))
        self.preview_frame_label.config(text=str(frame_num))
        self.update_preview()
    
    def update_preview(self):
        """Update preview images with current offsets"""
        if not hasattr(self, 'left_cap') or not hasattr(self, 'right_cap'):
            return
        
        if not self.left_cap.isOpened() or not self.right_cap.isOpened():
            return
        
        try:
            import cv2
            from PIL import Image, ImageTk, ImageDraw
            
            base_frame = self.preview_frame_var.get()
            left_frame_num = base_frame + self.left_offset
            right_frame_num = base_frame + self.right_offset
            
            # Get left frame
            self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, left_frame_num))
            ret_left, left_frame = self.left_cap.read()
            
            if ret_left:
                left_frame = cv2.cvtColor(left_frame, cv2.COLOR_BGR2RGB)
                left_frame = cv2.resize(left_frame, (400, 300))
                left_pil = Image.fromarray(left_frame)
                
                # Add frame number overlay
                draw = ImageDraw.Draw(left_pil)
                draw.text((10, 10), f"Left: {left_frame_num}", fill=(0, 255, 0))
                
                left_photo = ImageTk.PhotoImage(image=left_pil)
                self.left_preview_label.config(image=left_photo, text="")
                self.left_preview_label.image = left_photo
            
            # Get right frame
            self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, right_frame_num))
            ret_right, right_frame = self.right_cap.read()
            
            if ret_right:
                right_frame = cv2.cvtColor(right_frame, cv2.COLOR_BGR2RGB)
                right_frame = cv2.resize(right_frame, (400, 300))
                right_pil = Image.fromarray(right_frame)
                
                # Add frame number overlay
                draw = ImageDraw.Draw(right_pil)
                draw.text((10, 10), f"Right: {right_frame_num}", fill=(0, 255, 0))
                
                right_photo = ImageTk.PhotoImage(image=right_pil)
                self.right_preview_label.config(image=right_photo, text="")
                self.right_preview_label.image = right_photo
                
        except Exception as e:
            print(f"Preview update error: {e}")
        
    def log_message(self, message):
        """Add a message to the status text box"""
        self.status_text.config(state="normal")
        self.status_text.insert(tk.END, message + "\n")
        self.status_text.see(tk.END)
        self.status_text.config(state="disabled")
        self.root.update()
        
    def start_processing(self):
        if self.processing:
            return
        
        # Check if using videos or pre-extracted frames
        left_video = self.left_video_var.get()
        right_video = self.right_video_var.get()
        use_videos = (left_video != "Use pre-extracted frames" and 
                     right_video != "Use pre-extracted frames")
        
        if use_videos:
            # Validate video files exist
            if not os.path.exists(left_video):
                messagebox.showerror("Error", f"Left video not found: {left_video}")
                return
            if not os.path.exists(right_video):
                messagebox.showerror("Error", f"Right video not found: {right_video}")
                return
            
        # Get frame range
        start_frame = self.start_var.get()
        end_str = self.end_var.get().strip()
        end_frame = None if end_str.lower() == "all" else int(end_str)
        
        # Validate
        if start_frame < 0:
            messagebox.showerror("Error", "Start frame must be >= 0")
            return
            
        if end_frame is not None and end_frame <= start_frame:
            messagebox.showerror("Error", "End frame must be greater than start frame")
            return
        
        # Disable controls
        self.processing = True
        self.process_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.visualize_btn.config(state="disabled")
        self.progress_bar["value"] = 0
        self.progress_label.config(text="Starting...")
        
        # Clear previous results
        self.pipeline.clear_previous_results()
        self.status_text.config(state="normal")
        self.status_text.delete(1.0, tk.END)
        self.status_text.config(state="disabled")
        
        # Start processing in background thread
        thread = threading.Thread(target=self.process_video_thread, args=(start_frame, end_frame, use_videos, left_video, right_video))
        thread.daemon = True
        thread.start()
        
    def process_video_thread(self, start_frame, end_frame, use_videos, left_video_path, right_video_path):
        """Process video in background thread"""
        try:
            if use_videos:
                self.log_message(f"Processing from video files:")
                self.log_message(f"  Left: {os.path.basename(left_video_path)}")
                self.log_message(f"  Right: {os.path.basename(right_video_path)}")
            else:
                self.log_message(f"Processing from pre-extracted frames")
                
            self.log_message(f"Starting from frame {start_frame}...")
            if end_frame:
                self.log_message(f"Processing up to frame {end_frame}")
            else:
                self.log_message("Processing all available frames")
            self.log_message("-" * 50)
            
            # Open video captures if using videos
            if use_videos:
                import cv2
                
                # Reuse captures if already loaded for preview, otherwise open new
                if not hasattr(self, 'left_cap') or not self.left_cap.isOpened():
                    self.left_cap = cv2.VideoCapture(left_video_path)
                if not hasattr(self, 'right_cap') or not self.right_cap.isOpened():
                    self.right_cap = cv2.VideoCapture(right_video_path)
                
                if not self.left_cap.isOpened():
                    raise Exception(f"Failed to open left video: {left_video_path}")
                if not self.right_cap.isOpened():
                    raise Exception(f"Failed to open right video: {right_video_path}")
                
                total_frames = int(self.left_cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.log_message(f"Left video total frames: {total_frames}")
                
                if end_frame is None:
                    end_frame = total_frames
                else:
                    end_frame = min(end_frame, total_frames)
            else:
                # Count pre-extracted frames
                root = os.path.dirname(os.path.abspath(__file__))
                left_frames_dir = os.path.join(root, "..", "output_frames", "left")
                
                import glob
                left_files = sorted(glob.glob(os.path.join(left_frames_dir, "left3_*.jpg")))
                total_frames = len(left_files)
                
                if end_frame is None:
                    end_frame = total_frames
                else:
                    end_frame = min(end_frame, total_frames)
            
            frames_to_process = end_frame - start_frame
            self.log_message(f"Total frames to process: {frames_to_process}")
            self.log_message("-" * 50)
            
            # Process each frame
            import cv2
            success_count = 0
            fail_count = 0
            
            for frame_num in range(start_frame, end_frame):
                if use_videos:
                    # Read from video files WITH OFFSETS
                    left_frame_num = frame_num + self.left_offset
                    right_frame_num = frame_num + self.right_offset
                    
                    self.left_cap.set(cv2.CAP_PROP_POS_FRAMES, left_frame_num)
                    self.right_cap.set(cv2.CAP_PROP_POS_FRAMES, right_frame_num)
                    
                    ret_left, left_img = self.left_cap.read()
                    ret_right, right_img = self.right_cap.read()
                    
                    if not ret_left or not ret_right:
                        self.log_message(f"Frame {frame_num}: Failed to read from video")
                        fail_count += 1
                        continue
                else:
                    # Read from pre-extracted frames
                    frame_id = f"{frame_num:04d}"
                    root = os.path.dirname(os.path.abspath(__file__))
                    left_frames_dir = os.path.join(root, "..", "output_frames", "left")
                    right_frames_dir = os.path.join(root, "..", "output_frames", "right")
                    left_path = os.path.join(left_frames_dir, f"left3_{frame_id}.jpg")
                    right_path = os.path.join(right_frames_dir, f"right3_{frame_id}.jpg")
                    
                    if not os.path.exists(left_path) or not os.path.exists(right_path):
                        self.log_message(f"Frame {frame_num}: Files not found")
                        fail_count += 1
                        continue
                        
                    left_img = cv2.imread(left_path)
                    right_img = cv2.imread(right_path)
                    
                    if left_img is None or right_img is None:
                        self.log_message(f"Frame {frame_num}: Failed to load images")
                        fail_count += 1
                        continue
                
                # Process the frame pair
                result = self.pipeline.process_from_pair(left_img, right_img, frame_num)
                
                if result and isinstance(result, dict):
                    cam_coords = result['camera_coords']
                    world_coords = result['world_coords']
                    self.log_message(f"Frame {frame_num}: ✓ Camera: ({cam_coords[0]:.2f}, {cam_coords[1]:.2f}, {cam_coords[2]:.2f})")
                    success_count += 1
                else:
                    self.log_message(f"Frame {frame_num}: ✗ No detection")
                    fail_count += 1
                    
                # Update progress bar
                progress = ((frame_num - start_frame + 1) / frames_to_process) * 100
                self.progress_bar["value"] = progress
                self.progress_label.config(text=f"Processing: {frame_num - start_frame + 1}/{frames_to_process} frames ({progress:.1f}%)")
                self.root.update_idletasks()
            
            # Finish
            self.log_message("-" * 50)
            self.log_message(f"Processing complete!")
            self.log_message(f"Successful detections: {success_count}/{frames_to_process}")
            self.log_message(f"Failed detections: {fail_count}/{frames_to_process}")
            self.log_message(f"Success rate: {(success_count/frames_to_process)*100:.1f}%")
            
            self.progress_bar["value"] = 100
            self.progress_label.config(text=f"Complete! {success_count}/{frames_to_process} successful")
            
            # Close video captures if used
            if use_videos:
                self.left_cap.release()
                self.right_cap.release()
            
            # Enable export and visualize buttons
            self.export_btn.config(state="normal")
            self.visualize_btn.config(state="normal")
            
        except Exception as e:
            self.log_message(f"ERROR: {str(e)}")
            messagebox.showerror("Processing Error", str(e))
            
        finally:
            # Re-enable controls
            self.processing = False
            self.process_btn.config(state="normal")
            
    def export_results(self):
        """Export results to CSV"""
        try:
            output_dir = filedialog.askdirectory(title="Select output directory")
            if not output_dir:
                return
                
            self.pipeline.export_results(output_dir)
            self.log_message(f"Results exported to: {output_dir}")
            messagebox.showinfo("Success", f"Results exported to:\n{output_dir}")
            
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            
    def visualize_3d(self):
        """Visualize results in 3D"""
        try:
            self.pipeline.visualize_results(type="3d")
        except Exception as e:
            messagebox.showerror("Visualization Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = VideoProcessorGUI(root)
    root.mainloop()
