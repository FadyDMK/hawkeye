import os
import cv2
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import time
from pathlib import Path

class VideoFrameExtractor:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Hawkeye - Video Frame Extractor")
        self.root.geometry("800x600")
        self.root.configure(padx=20, pady=20)
        
        self.left_video_path = tk.StringVar()
        self.right_video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.frame_interval = tk.IntVar(value=1)  # Extract every nth frame
        self.start_frame = tk.IntVar(value=0)
        self.end_frame = tk.IntVar(value=-1)  # -1 means all frames
        self.prefix_left = tk.StringVar(value="left3_")
        self.prefix_right = tk.StringVar(value="right3_")
        
        # Default output directory
        src_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        default_output = os.path.join(src_dir, "output_frames")
        self.output_dir.set(default_output)
        
        # Track if videos are loaded
        self.has_left_video = False
        self.has_right_video = False
        self.extraction_running = False
        
        # Create UI elements
        self.create_widgets()
        
        # Set initial button states
        self.update_button_states()
        
        # Add trace to video path variables
        self.left_video_path.trace_add("write", self.on_video_path_change)
        self.right_video_path.trace_add("write", self.on_video_path_change)
    
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill="both", expand=True)
        
        # Video selection frame
        video_frame = ttk.LabelFrame(main_frame, text="Select Videos")
        video_frame.pack(fill="x", padx=10, pady=10)
        
        # Left video
        ttk.Label(video_frame, text="Left Camera Video:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(video_frame, textvariable=self.left_video_path, width=50).grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(video_frame, text="Browse...", command=self.browse_left_video).grid(row=0, column=2, padx=5, pady=5)
        
        # Right video
        ttk.Label(video_frame, text="Right Camera Video:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(video_frame, textvariable=self.right_video_path, width=50).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(video_frame, text="Browse...", command=self.browse_right_video).grid(row=1, column=2, padx=5, pady=5)
        
        # Output directory
        ttk.Label(video_frame, text="Output Directory:").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        ttk.Entry(video_frame, textvariable=self.output_dir, width=50).grid(row=2, column=1, padx=5, pady=5)
        ttk.Button(video_frame, text="Browse...", command=self.browse_output_dir).grid(row=2, column=2, padx=5, pady=5)
        
        # Parameters frame
        self.params_frame = ttk.LabelFrame(main_frame, text="Extraction Parameters")
        self.params_frame.pack(fill="x", padx=10, pady=10)
        
        # Frame interval
        ttk.Label(self.params_frame, text="Extract every N frames:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.interval_spinbox = ttk.Spinbox(self.params_frame, from_=1, to=30, textvariable=self.frame_interval, width=5)
        self.interval_spinbox.grid(row=0, column=1, sticky="w", padx=5, pady=5)
        
        # Start frame
        ttk.Label(self.params_frame, text="Start Frame:").grid(row=1, column=0, sticky="w", padx=5, pady=5)
        self.start_spinbox = ttk.Spinbox(self.params_frame, from_=0, to=100000, textvariable=self.start_frame, width=10)
        self.start_spinbox.grid(row=1, column=1, sticky="w", padx=5, pady=5)
        
        # End frame
        ttk.Label(self.params_frame, text="End Frame (-1 for all):").grid(row=2, column=0, sticky="w", padx=5, pady=5)
        self.end_spinbox = ttk.Spinbox(self.params_frame, from_=-1, to=100000, textvariable=self.end_frame, width=10)
        self.end_spinbox.grid(row=2, column=1, sticky="w", padx=5, pady=5)
        
        # Prefix settings
        ttk.Label(self.params_frame, text="Left Frame Prefix:").grid(row=3, column=0, sticky="w", padx=5, pady=5)
        self.left_prefix_entry = ttk.Entry(self.params_frame, textvariable=self.prefix_left, width=20)
        self.left_prefix_entry.grid(row=3, column=1, sticky="w", padx=5, pady=5)
        
        ttk.Label(self.params_frame, text="Right Frame Prefix:").grid(row=4, column=0, sticky="w", padx=5, pady=5)
        self.right_prefix_entry = ttk.Entry(self.params_frame, textvariable=self.prefix_right, width=20)
        self.right_prefix_entry.grid(row=4, column=1, sticky="w", padx=5, pady=5)
        
        # Progress frame
        progress_frame = ttk.LabelFrame(main_frame, text="Extraction Progress")
        progress_frame.pack(fill="x", padx=10, pady=10)
        
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill="x", padx=10, pady=10)
        
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(progress_frame, textvariable=self.status_var)
        status_label.pack(padx=10, pady=5)
        
        # Button frame
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", padx=10, pady=20)
        
        # Extract button
        self.extract_btn = ttk.Button(button_frame, text="Start Extraction", command=self.start_extraction)
        self.extract_btn.pack(side="right", padx=5)
        
        # Cancel button
        self.cancel_btn = ttk.Button(button_frame, text="Cancel", command=self.cancel_extraction)
        self.cancel_btn.pack(side="right", padx=5)
        self.cancel_btn["state"] = "disabled"

        # Log frame
        log_frame = ttk.LabelFrame(main_frame, text="Log")
        log_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        self.log_text = tk.Text(log_frame, height=10, wrap="word")
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Scrollbar for log
        scrollbar = ttk.Scrollbar(self.log_text, command=self.log_text.yview)
        scrollbar.pack(side="right", fill="y")
        self.log_text.config(yscrollcommand=scrollbar.set)
    
    def on_video_path_change(self, *args):
        # Check if video paths are valid and update states accordingly
        self.has_left_video = os.path.exists(self.left_video_path.get()) if self.left_video_path.get() else False
        self.has_right_video = os.path.exists(self.right_video_path.get()) if self.right_video_path.get() else False
        
        # Update the UI elements based on video availability
        self.update_button_states()
        
        if self.has_left_video:
            self.log_message(f"Left video file validated: {self.left_video_path.get()}")
        if self.has_right_video:
            self.log_message(f"Right video file validated: {self.right_video_path.get()}")
    
    def update_button_states(self):
        # Determine the button and control states
        has_any_video = self.has_left_video or self.has_right_video
        
        # Update extraction button state
        if has_any_video and not self.extraction_running:
            self.extract_btn["state"] = "normal"
        else:
            self.extract_btn["state"] = "disabled"
            
        # Update prefix entry states
        self.left_prefix_entry["state"] = "normal" if self.has_left_video else "disabled"
        self.right_prefix_entry["state"] = "normal" if self.has_right_video else "disabled"
            
        # Update parameter controls
        state = "normal" if has_any_video else "disabled"
        self.interval_spinbox["state"] = state
        self.start_spinbox["state"] = state
        self.end_spinbox["state"] = state
    
    def browse_left_video(self):
        filename = filedialog.askopenfilename(
            title="Select Left Camera Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )
        if filename:
            self.left_video_path.set(filename)
    
    def browse_right_video(self):
        filename = filedialog.askopenfilename(
            title="Select Right Camera Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
        )
        if filename:
            self.right_video_path.set(filename)
    
    def browse_output_dir(self):
        dirname = filedialog.askdirectory(title="Select Output Directory")
        if dirname:
            self.output_dir.set(dirname)
            self.log_message(f"Selected output directory: {dirname}")
    
    def log_message(self, message):
        self.log_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {message}\n")
        self.log_text.see(tk.END)
    
    def start_extraction(self):
        # Validate output directory
        if not os.path.exists(self.output_dir.get()):
            try:
                os.makedirs(self.output_dir.get())
                os.makedirs(os.path.join(self.output_dir.get(), "left"), exist_ok=True)
                os.makedirs(os.path.join(self.output_dir.get(), "right"), exist_ok=True)
                self.log_message("Created output directories")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to create output directory: {e}")
                return
        
        # Update UI state
        self.extraction_running = True
        self.extract_btn["state"] = "disabled"
        self.cancel_btn["state"] = "normal"
        self.status_var.set("Extracting frames...")
        self.update_button_states()
        
        # Start extraction in a separate thread
        self.extraction_thread = threading.Thread(target=self.extract_frames)
        self.extraction_thread.daemon = True
        self.extraction_thread.start()
    
    def cancel_extraction(self):
        if self.extraction_running:
            self.extraction_running = False
            self.status_var.set("Cancelling...")
            self.log_message("Extraction cancelled by user")
    
    def extract_frames(self):
        try:
            # Process left video if selected
            if self.has_left_video:
                self.process_video(
                    self.left_video_path.get(),
                    os.path.join(self.output_dir.get(), "left"),
                    self.prefix_left.get()
                )
            
            # Process right video if selected
            if self.has_right_video:
                self.process_video(
                    self.right_video_path.get(),
                    os.path.join(self.output_dir.get(), "right"),
                    self.prefix_right.get()
                )
            
            if self.extraction_running:
                self.status_var.set("Extraction complete")
                self.log_message("Frame extraction completed successfully")
                messagebox.showinfo("Success", "Frame extraction completed successfully!")
            else:
                self.status_var.set("Extraction cancelled")
            if self.extraction_running:
                self.status_var.set("Extraction complete")
                self.log_message("Frame extraction completed successfully")
                
                # Show dialog asking if they want to continue to Frame Analyzer
                result = messagebox.askquestion("Success", 
                                            "Frame extraction completed successfully!\n\nWould you like to open the Frame Analyzer to process these frames?",
                                            icon='info')
                if result == 'yes':
                    self.root.destroy()  # Close frame extractor
                    
                    # Launch the frame selector
                    import front_end
                    root = tk.Tk()
                    app = front_end.FrameSelectorApp(root)
                    root.mainloop()
                    return  # Exit this function since we're launching another app
                else:
                    # Just show success message if they choose not to continue
                    messagebox.showinfo("Success", "Frame extraction completed successfully!")
            else:
                self.status_var.set("Extraction cancelled")
        except Exception as e:
            self.log_message(f"Error during extraction: {str(e)}")
            self.status_var.set("Error occurred")
            messagebox.showerror("Error", f"An error occurred during extraction: {str(e)}")
        
        finally:
            # Reset UI
            self.extraction_running = False
            self.progress_var.set(0)
            self.update_button_states()
            self.cancel_btn["state"] = "disabled"
    
    def process_video(self, video_path, output_dir, prefix):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            self.log_message(f"Error: Could not open video {video_path}")
            return
        
        # Get video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Determine start and end frames
        start = self.start_frame.get()
        end = self.end_frame.get()
        if end == -1 or end >= total_frames:
            end = total_frames - 1
        
        frames_to_process = (end - start) // self.frame_interval.get() + 1
        
        self.log_message(f"Processing video: {Path(video_path).name}")
        self.log_message(f"Total frames: {total_frames}, FPS: {fps:.2f}")
        self.log_message(f"Extracting frames {start} to {end} (every {self.frame_interval.get()} frames)")
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Set the frame position to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        
        # Process frames
        frame_count = 0
        saved_count = 0
        
        while cap.isOpened() and frame_count <= (end - start) and self.extraction_running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Extract every Nth frame
            if frame_count % self.frame_interval.get() == 0:
                frame_number = start + frame_count
                frame_id = f"{frame_number:04d}"
                output_path = os.path.join(output_dir, f"{prefix}{frame_id}.jpg")
                
                cv2.imwrite(output_path, frame)
                saved_count += 1
                
                # Update progress
                progress = (frame_count / (end - start)) * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Saved {saved_count} frames... ({progress:.1f}%)")
                
                # Update UI occasionally
                if saved_count % 10 == 0:
                    self.root.update_idletasks()
            
            frame_count += 1
        
        cap.release()
        self.log_message(f"Saved {saved_count} frames to {output_dir}")
    
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoFrameExtractor()
    app.run()