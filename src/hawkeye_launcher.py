import os
import sys
import tkinter as tk
from tkinter import ttk, messagebox

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
        config_btn = ttk.Button(
            btn_frame,
            text="Camera & Court Configuration",
            command=self.launch_configuration,
            width=30,
        )
        config_btn.pack(pady=10)
        
        # Extract frames button
        extract_btn = ttk.Button(
            btn_frame,
            text="Video Frame Extractor",
            command=self.launch_frame_extractor,
            width=30,
        )
        extract_btn.pack(pady=10)
        
        # Frame selector button
        frame_selector_btn = ttk.Button(
            btn_frame,
            text="Frame Analyzer",
            command=self.launch_frame_selector,
            width=30,
        )
        frame_selector_btn.pack(pady=10)
        
        # Process videos button
        process_btn = ttk.Button(
            btn_frame,
            text="Process Complete Videos",
            command=self.launch_video_processor,
            width=30,
        )
        process_btn.pack(pady=10)
        
        # Footer
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill="x", pady=10, side="bottom")
        
        status_label = ttk.Label(footer_frame, text="Ready", font=("Arial", 10))
        status_label.pack(side="left")
        
        version_label = ttk.Label(footer_frame, text="v1.0.0", font=("Arial", 10))
        version_label.pack(side="right")
    
    def launch_configuration(self):
        # Open configuration in a child window without closing launcher
        try:
            import sys
            import os
            sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'config'))
            from camera_config import CameraConfigDialog
        except Exception as e:
            messagebox.showerror("Error", f"Cannot open configuration: {e}")
            return
        top = tk.Toplevel(self.root)
        top.title("Camera & Court Configuration")
        # CameraConfigDialog is expected to manage its own layout/events
        CameraConfigDialog(top)

    def _run_script(self, script_name: str):
        """Launch a Python script as a separate process, keeping launcher open."""
        script_path = os.path.join(os.path.dirname(__file__), script_name)
        if not os.path.exists(script_path):
            messagebox.showerror("Error", f"Not found: {script_name}")
            return
        try:
            # Use the current Python interpreter
            if os.name == 'nt':
                os.spawnl(os.P_NOWAIT, sys.executable, sys.executable, script_path)
            else:
                pid = os.fork()
                if pid == 0:
                    os.execl(sys.executable, sys.executable, script_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch {script_name}: {e}")
    
    def launch_frame_extractor(self):
        # Prefer the main extractor file
        self._run_script("video_frame_extractor.py")
    
    def launch_frame_selector(self):
        # Try main.py first; fallback to front_end.py
        if os.path.exists(os.path.join(os.path.dirname(__file__), "main.py")):
            self._run_script("main.py")
        else:
            self._run_script("front_end.py")
    
    def launch_video_processor(self):
        # Placeholder for future full-video processor GUI
        messagebox.showinfo(
            "Coming Soon",
            "The full video processing interface is under development.\n\n"
            "You can currently process frames using the Frame Analyzer."
        )
if __name__ == "__main__":
    root = tk.Tk()
    app = HawkeyeLauncher(root)
    root.mainloop()