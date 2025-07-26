import os
import tkinter as tk
from tkinter import ttk
import sys

class HawkeyeLauncher:
    def __init__(self, root):
        self.root = root
        self.root.title("Hawkeye Launcher")
        self.root.geometry("500x350")
        self.root.configure(padx=20, pady=20)
        
        self.create_widgets()
    
    def create_widgets(self):
        # Header with logo/title
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill="x", pady=10)
        
        title_label = ttk.Label(header_frame, text="Hawkeye Volleyball Tracking System", font=("Arial", 16))
        title_label.pack()
        
        # Main buttons frame
        btn_frame = ttk.Frame(self.root)
        btn_frame.pack(fill="both", expand=True, pady=20)
        
        # Configuration button
        config_btn = ttk.Button(btn_frame, text="Camera & Court Configuration", 
                               command=self.launch_configuration, width=30)
        config_btn.pack(pady=10)
        
        # Extract frames button
        extract_btn = ttk.Button(btn_frame, text="Video Frame Extractor", 
                                command=self.launch_frame_extractor, width=30)
        extract_btn.pack(pady=10)
        
        # Frame selector button
        frame_selector_btn = ttk.Button(btn_frame, text="Frame Analyzer", 
                                     command=self.launch_frame_selector, width=30)
        frame_selector_btn.pack(pady=10)
        
        # Process videos button
        process_btn = ttk.Button(btn_frame, text="Process Complete Videos", 
                              command=self.launch_video_processor, width=30)
        process_btn.pack(pady=10)
        
        # Footer
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill="x", pady=10, side="bottom")
        
        status_label = ttk.Label(footer_frame, text="Ready", font=("Arial", 10))
        status_label.pack(side="left")
        
        version_label = ttk.Label(footer_frame, text="v1.0.0", font=("Arial", 10))
        version_label.pack(side="right")
    
    def launch_frame_extractor(self):
        self.root.destroy()  # Close launcher
        # Import and run frame extractor
        from video_frame_extractor import VideoFrameExtractor
        app = VideoFrameExtractor()
        app.run()
    
    def launch_frame_selector(self):
        self.root.destroy()  # Close launcher
        # Import and run frame selector
        import front_end
        root = tk.Tk()
        app = front_end.FrameSelectorApp(root)
        root.mainloop()
    
    def launch_video_processor(self):
        self.root.destroy()  # Close launcher
        # You could create a simple GUI for this too, or just run the pipeline directly
        try:
            from process_videos_gui import ProcessVideosGUI
            app = ProcessVideosGUI()
            app.run()
        except ImportError:
            # Fallback if the GUI isn't implemented yet
            import tkinter as tk
            from tkinter import messagebox
            
            # Show message that this feature is coming soon
            root = tk.Tk()
            root.withdraw()  # Hide the root window
            messagebox.showinfo("Coming Soon", 
                            "The full video processing interface is under development.\n\n" +
                            "You can currently process frames using the Frame Analyzer.")
            root.destroy()
if __name__ == "__main__":
    root = tk.Tk()
    app = HawkeyeLauncher(root)
    root.mainloop()