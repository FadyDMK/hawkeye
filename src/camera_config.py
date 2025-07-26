import tkinter as tk
from tkinter import ttk, messagebox
import json
import os

class CameraConfigDialog:
    def __init__(self, parent=None):
        self.root = tk.Toplevel(parent) if parent else tk.Tk()
        self.root.title("Camera & Court Configuration")
        self.root.geometry("500x600")
        self.root.configure(padx=20, pady=20)
        
        # Configuration values
        self.config = {
            # Camera parameters
            'focal_length_mm': 26.0,
            'sensor_width_mm': 36.0,
            'resolution_width': 1920,
            'resolution_height': 1080,
            'baseline_m': 3.0,
            
            # Depth/Distance parameters
            'z_min_m': 15.0,
            'z_max_m': 40.0,
            
            # Court parameters
            'court_length_m': 18.0,  # Standard volleyball court length
            'court_width_m': 9.0,    # Standard volleyball court width
            'net_height_m': 2.43,    # Standard volleyball net height (men's)
            
            # Stereo matching parameters
            'sgbm_window_size': 5,
            'sgbm_block_size': 5,
            'sgbm_min_disp': -1,
            'sgbm_num_disp_factor': 16,
            'sgbm_uniqueness_ratio': 15,
            'sgbm_speckle_window_size': 50,
            'sgbm_speckle_range': 10,
            
            # WLS Filter parameters
            'wls_lambda': 5000.0,
            'wls_sigma': 1.0
        }
        
        self.result = None
        self.create_widgets()
        self.load_config()
        
        # Make dialog modal
        self.root.transient(parent)
        self.root.grab_set()
        
    def create_widgets(self):
        # Create notebook for tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, pady=(0, 10))
        
        # Camera Parameters Tab
        camera_frame = ttk.Frame(notebook)
        notebook.add(camera_frame, text="Camera Parameters")
        self.create_camera_tab(camera_frame)
        
        # Court Parameters Tab
        court_frame = ttk.Frame(notebook)
        notebook.add(court_frame, text="Court Parameters")
        self.create_court_tab(court_frame)
        
        # Stereo Matching Tab
        stereo_frame = ttk.Frame(notebook)
        notebook.add(stereo_frame, text="Stereo Matching")
        self.create_stereo_tab(stereo_frame)
        
        # Buttons
        button_frame = ttk.Frame(self.root)
        button_frame.pack(fill="x", pady=(10, 0))
        
        ttk.Button(button_frame, text="Save & Apply", command=self.save_config).pack(side="right", padx=(5, 0))
        ttk.Button(button_frame, text="Cancel", command=self.cancel).pack(side="right")
        ttk.Button(button_frame, text="Reset to Defaults", command=self.reset_defaults).pack(side="left")
        
    def create_camera_tab(self, parent):
        # Camera hardware parameters
        ttk.Label(parent, text="Camera Hardware", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(parent, text="Focal Length (mm):").grid(row=1, column=0, sticky="w", pady=2)
        self.focal_length_var = tk.DoubleVar(value=self.config['focal_length_mm'])
        ttk.Entry(parent, textvariable=self.focal_length_var, width=15).grid(row=1, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Sensor Width (mm):").grid(row=2, column=0, sticky="w", pady=2)
        self.sensor_width_var = tk.DoubleVar(value=self.config['sensor_width_mm'])
        ttk.Entry(parent, textvariable=self.sensor_width_var, width=15).grid(row=2, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Resolution Width (px):").grid(row=3, column=0, sticky="w", pady=2)
        self.res_width_var = tk.IntVar(value=self.config['resolution_width'])
        ttk.Entry(parent, textvariable=self.res_width_var, width=15).grid(row=3, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Resolution Height (px):").grid(row=4, column=0, sticky="w", pady=2)
        self.res_height_var = tk.IntVar(value=self.config['resolution_height'])
        ttk.Entry(parent, textvariable=self.res_height_var, width=15).grid(row=4, column=1, sticky="w", padx=(10, 0))
        
        # Camera setup parameters
        ttk.Label(parent, text="Camera Setup", font=("Arial", 12, "bold")).grid(row=5, column=0, columnspan=2, sticky="w", pady=(20, 5))
        
        ttk.Label(parent, text="Baseline (m):").grid(row=6, column=0, sticky="w", pady=2)
        self.baseline_var = tk.DoubleVar(value=self.config['baseline_m'])
        ttk.Entry(parent, textvariable=self.baseline_var, width=15).grid(row=6, column=1, sticky="w", padx=(10, 0))
        ttk.Label(parent, text="Distance between cameras", font=("Arial", 8)).grid(row=6, column=2, sticky="w", padx=(10, 0))
        
        # Depth range parameters
        ttk.Label(parent, text="Depth Range", font=("Arial", 12, "bold")).grid(row=7, column=0, columnspan=2, sticky="w", pady=(20, 5))
        
        ttk.Label(parent, text="Min Distance (m):").grid(row=8, column=0, sticky="w", pady=2)
        self.z_min_var = tk.DoubleVar(value=self.config['z_min_m'])
        ttk.Entry(parent, textvariable=self.z_min_var, width=15).grid(row=8, column=1, sticky="w", padx=(10, 0))
        ttk.Label(parent, text="Closest trackable distance", font=("Arial", 8)).grid(row=8, column=2, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Max Distance (m):").grid(row=9, column=0, sticky="w", pady=2)
        self.z_max_var = tk.DoubleVar(value=self.config['z_max_m'])
        ttk.Entry(parent, textvariable=self.z_max_var, width=15).grid(row=9, column=1, sticky="w", padx=(10, 0))
        ttk.Label(parent, text="Farthest trackable distance", font=("Arial", 8)).grid(row=9, column=2, sticky="w", padx=(10, 0))
        
        # Calculated values display
        ttk.Label(parent, text="Calculated Values", font=("Arial", 12, "bold")).grid(row=10, column=0, columnspan=2, sticky="w", pady=(20, 5))
        
        self.focal_px_label = ttk.Label(parent, text="Focal Length (px): ")
        self.focal_px_label.grid(row=11, column=0, columnspan=2, sticky="w", pady=2)
        
        # Update calculated values when parameters change
        for var in [self.focal_length_var, self.sensor_width_var, self.res_width_var]:
            var.trace_add("write", self.update_calculated_values)
        
        self.update_calculated_values()
        
    def create_court_tab(self, parent):
        ttk.Label(parent, text="Volleyball Court Dimensions", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(parent, text="Court Length (m):").grid(row=1, column=0, sticky="w", pady=2)
        self.court_length_var = tk.DoubleVar(value=self.config['court_length_m'])
        ttk.Entry(parent, textvariable=self.court_length_var, width=15).grid(row=1, column=1, sticky="w", padx=(10, 0))
        ttk.Label(parent, text="Standard: 18m", font=("Arial", 8)).grid(row=1, column=2, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Court Width (m):").grid(row=2, column=0, sticky="w", pady=2)
        self.court_width_var = tk.DoubleVar(value=self.config['court_width_m'])
        ttk.Entry(parent, textvariable=self.court_width_var, width=15).grid(row=2, column=1, sticky="w", padx=(10, 0))
        ttk.Label(parent, text="Standard: 9m", font=("Arial", 8)).grid(row=2, column=2, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Net Height (m):").grid(row=3, column=0, sticky="w", pady=2)
        self.net_height_var = tk.DoubleVar(value=self.config['net_height_m'])
        ttk.Entry(parent, textvariable=self.net_height_var, width=15).grid(row=3, column=1, sticky="w", padx=(10, 0))
        ttk.Label(parent, text="Men's: 2.43m, Women's: 2.24m", font=("Arial", 8)).grid(row=3, column=2, sticky="w", padx=(10, 0))
        
        # Preset buttons
        ttk.Label(parent, text="Presets", font=("Arial", 12, "bold")).grid(row=4, column=0, columnspan=2, sticky="w", pady=(20, 5))
        
        preset_frame = ttk.Frame(parent)
        preset_frame.grid(row=5, column=0, columnspan=3, sticky="w", pady=5)
        
        ttk.Button(preset_frame, text="Men's Standard", command=self.set_mens_court).pack(side="left", padx=(0, 5))
        ttk.Button(preset_frame, text="Women's Standard", command=self.set_womens_court).pack(side="left", padx=(0, 5))
        
    def create_stereo_tab(self, parent):
        # SGBM Parameters
        ttk.Label(parent, text="SGBM Algorithm Parameters", font=("Arial", 12, "bold")).grid(row=0, column=0, columnspan=2, sticky="w", pady=(10, 5))
        
        ttk.Label(parent, text="Window Size:").grid(row=1, column=0, sticky="w", pady=2)
        self.window_size_var = tk.IntVar(value=self.config['sgbm_window_size'])
        ttk.Spinbox(parent, from_=3, to=21, textvariable=self.window_size_var, width=10).grid(row=1, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Block Size:").grid(row=2, column=0, sticky="w", pady=2)
        self.block_size_var = tk.IntVar(value=self.config['sgbm_block_size'])
        ttk.Spinbox(parent, from_=3, to=21, textvariable=self.block_size_var, width=10).grid(row=2, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Min Disparity:").grid(row=3, column=0, sticky="w", pady=2)
        self.min_disp_var = tk.IntVar(value=self.config['sgbm_min_disp'])
        ttk.Entry(parent, textvariable=self.min_disp_var, width=15).grid(row=3, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Num Disparities Factor:").grid(row=4, column=0, sticky="w", pady=2)
        self.num_disp_factor_var = tk.IntVar(value=self.config['sgbm_num_disp_factor'])
        ttk.Spinbox(parent, from_=1, to=32, textvariable=self.num_disp_factor_var, width=10).grid(row=4, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Uniqueness Ratio:").grid(row=5, column=0, sticky="w", pady=2)
        self.uniqueness_var = tk.IntVar(value=self.config['sgbm_uniqueness_ratio'])
        ttk.Entry(parent, textvariable=self.uniqueness_var, width=15).grid(row=5, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Speckle Window Size:").grid(row=6, column=0, sticky="w", pady=2)
        self.speckle_window_var = tk.IntVar(value=self.config['sgbm_speckle_window_size'])
        ttk.Entry(parent, textvariable=self.speckle_window_var, width=15).grid(row=6, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Speckle Range:").grid(row=7, column=0, sticky="w", pady=2)
        self.speckle_range_var = tk.IntVar(value=self.config['sgbm_speckle_range'])
        ttk.Entry(parent, textvariable=self.speckle_range_var, width=15).grid(row=7, column=1, sticky="w", padx=(10, 0))
        
        # WLS Filter Parameters
        ttk.Label(parent, text="WLS Filter Parameters", font=("Arial", 12, "bold")).grid(row=8, column=0, columnspan=2, sticky="w", pady=(20, 5))
        
        ttk.Label(parent, text="Lambda:").grid(row=9, column=0, sticky="w", pady=2)
        self.wls_lambda_var = tk.DoubleVar(value=self.config['wls_lambda'])
        ttk.Entry(parent, textvariable=self.wls_lambda_var, width=15).grid(row=9, column=1, sticky="w", padx=(10, 0))
        
        ttk.Label(parent, text="Sigma:").grid(row=10, column=0, sticky="w", pady=2)
        self.wls_sigma_var = tk.DoubleVar(value=self.config['wls_sigma'])
        ttk.Entry(parent, textvariable=self.wls_sigma_var, width=15).grid(row=10, column=1, sticky="w", padx=(10, 0))
        
    def update_calculated_values(self, *args):
        try:
            focal_length_px = (self.focal_length_var.get() * self.res_width_var.get()) / self.sensor_width_var.get()
            self.focal_px_label.config(text=f"Focal Length (px): {focal_length_px:.2f}")
        except:
            self.focal_px_label.config(text="Focal Length (px): Invalid input")
    
    def set_mens_court(self):
        self.court_length_var.set(18.0)
        self.court_width_var.set(9.0)
        self.net_height_var.set(2.43)
    
    def set_womens_court(self):
        self.court_length_var.set(18.0)
        self.court_width_var.set(9.0)
        self.net_height_var.set(2.24)
    
    def reset_defaults(self):
        self.focal_length_var.set(26.0)
        self.sensor_width_var.set(36.0)
        self.res_width_var.set(1920)
        self.res_height_var.set(1080)
        self.baseline_var.set(3.0)
        self.z_min_var.set(15.0)
        self.z_max_var.set(40.0)
        self.court_length_var.set(18.0)
        self.court_width_var.set(9.0)
        self.net_height_var.set(2.43)
        self.window_size_var.set(5)
        self.block_size_var.set(5)
        self.min_disp_var.set(-1)
        self.num_disp_factor_var.set(16)
        self.uniqueness_var.set(15)
        self.speckle_window_var.set(50)
        self.speckle_range_var.set(10)
        self.wls_lambda_var.set(5000.0)
        self.wls_sigma_var.set(1.0)
    
    def save_config(self):
        try:
            # Update config with current values
            self.config.update({
                'focal_length_mm': self.focal_length_var.get(),
                'sensor_width_mm': self.sensor_width_var.get(),
                'resolution_width': self.res_width_var.get(),
                'resolution_height': self.res_height_var.get(),
                'baseline_m': self.baseline_var.get(),
                'z_min_m': self.z_min_var.get(),
                'z_max_m': self.z_max_var.get(),
                'court_length_m': self.court_length_var.get(),
                'court_width_m': self.court_width_var.get(),
                'net_height_m': self.net_height_var.get(),
                'sgbm_window_size': self.window_size_var.get(),
                'sgbm_block_size': self.block_size_var.get(),
                'sgbm_min_disp': self.min_disp_var.get(),
                'sgbm_num_disp_factor': self.num_disp_factor_var.get(),
                'sgbm_uniqueness_ratio': self.uniqueness_var.get(),
                'sgbm_speckle_window_size': self.speckle_window_var.get(),
                'sgbm_speckle_range': self.speckle_range_var.get(),
                'wls_lambda': self.wls_lambda_var.get(),
                'wls_sigma': self.wls_sigma_var.get()
            })
            
            # Calculate and add derived values
            self.config['focal_length_px'] = (self.config['focal_length_mm'] * self.config['resolution_width']) / self.config['sensor_width_mm']
            
            # Save to file
            config_path = os.path.join(os.path.dirname(__file__), "camera_config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
            
            self.result = self.config
            self.root.destroy()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")
    
    def cancel(self):
        self.result = None
        self.root.destroy()
    
    def load_config(self):
        try:
            config_path = os.path.join(os.path.dirname(__file__), "camera_config.json")
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    saved_config = json.load(f)
                    self.config.update(saved_config)
                    
                # Update GUI with loaded values
                self.focal_length_var.set(self.config['focal_length_mm'])
                self.sensor_width_var.set(self.config['sensor_width_mm'])
                self.res_width_var.set(self.config['resolution_width'])
                self.res_height_var.set(self.config['resolution_height'])
                self.baseline_var.set(self.config['baseline_m'])
                self.z_min_var.set(self.config['z_min_m'])
                self.z_max_var.set(self.config['z_max_m'])
                self.court_length_var.set(self.config['court_length_m'])
                self.court_width_var.set(self.config['court_width_m'])
                self.net_height_var.set(self.config['net_height_m'])
                
        except Exception as e:
            print(f"Could not load config: {e}")
    
    def get_config(self):
        """Run the dialog and return the configuration"""
        self.root.wait_window()
        return self.result

def load_camera_config():
    """Load camera configuration from file, or return defaults if file doesn't exist"""
    try:
        config_path = os.path.join(os.path.dirname(__file__), "camera_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
                # Calculate focal length in pixels if not present
                if 'focal_length_px' not in config:
                    config['focal_length_px'] = (config['focal_length_mm'] * config['resolution_width']) / config['sensor_width_mm']
                return config
    except Exception as e:
        print(f"Error loading config: {e}")
    
    # Return default configuration
    default_config = {
        'focal_length_mm': 26.0,
        'sensor_width_mm': 36.0,
        'resolution_width': 1920,
        'resolution_height': 1080,
        'baseline_m': 3.0,
        'z_min_m': 15.0,
        'z_max_m': 40.0,
        'court_length_m': 31.2,
        'court_width_m': 15.1,
        'net_height_m': 2.43,
        'sgbm_window_size': 5,
        'sgbm_block_size': 5,
        'sgbm_min_disp': -1,
        'sgbm_num_disp_factor': 16,
        'sgbm_uniqueness_ratio': 15,
        'sgbm_speckle_window_size': 50,
        'sgbm_speckle_range': 10,
        'wls_lambda': 5000.0,
        'wls_sigma': 1.0
    }
    default_config['focal_length_px'] = (default_config['focal_length_mm'] * default_config['resolution_width']) / default_config['sensor_width_mm']
    return default_config

if __name__ == "__main__":
    # Test the dialog
    dialog = CameraConfigDialog()
    config = dialog.get_config()
    if config:
        print("Configuration saved:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    else:
        print("Configuration cancelled")
