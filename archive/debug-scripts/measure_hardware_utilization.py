"""
Measure actual hardware resource utilization during pipeline processing.
Requires: pip install psutil GPUtil
"""

import psutil
import time
import cv2
import numpy as np
from pathlib import Path

try:
    import GPUtil
    GPU_AVAILABLE = True
except ImportError:
    print("GPUtil not installed. Install with: pip install gputil")
    GPU_AVAILABLE = False

# Import pipeline components
import sys
sys.path.insert(0, str(Path(__file__).parent / 'src'))
from volleyball_detection import get_ball_xy

def measure_resources_during_processing(num_frames=20):
    """Process frames while monitoring hardware utilization."""
    
    # Initialize pipeline
    print("Initializing detection model...")
    # Warm up the model by detecting on a dummy frame
    dummy_frames = list(Path("output_frames/left").glob("*.jpg"))
    if not dummy_frames:
        print("Error: No frames found in output_frames/left/")
        return
    dummy = cv2.imread(str(dummy_frames[0]))
    if dummy is not None:
        _ = get_ball_xy(dummy)
    print("Model loaded!")
    
    # Get test frames
    left_frames = sorted(Path("output_frames/left").glob("left*.jpg"))[:num_frames]
    right_frames = sorted(Path("output_frames/right").glob("right*.jpg"))[:num_frames]
    
    if not left_frames or not right_frames:
        print("Error: No frames found in output_frames/left or output_frames/right/")
        return
    
    print(f"Found {len(left_frames)} frame pairs")
    print("\nStarting measurement...")
    print("=" * 60)
    
    # Storage for measurements
    gpu_loads = []
    gpu_mems = []
    gpu_temps = []
    cpu_loads = []
    ram_usages = []
    
    # Process frames and measure
    for i, (left_path, right_path) in enumerate(zip(left_frames, right_frames)):
        # Load frames
        left_img = cv2.imread(str(left_path))
        right_img = cv2.imread(str(right_path))
        
        # Measure BEFORE processing
        cpu_before = psutil.cpu_percent(interval=0.1)
        ram_before = psutil.virtual_memory().used / (1024**3)  # GB
        
        if GPU_AVAILABLE:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load_before = gpu.load * 100
                gpu_mem_before = gpu.memoryUsed / 1024  # GB
                gpu_temp_before = gpu.temperature
        
        # Process frame - detect in both cameras
        left_detection = get_ball_xy(left_img)
        right_detection = get_ball_xy(right_img)
        
        # Simple success check
        success = left_detection is not None and right_detection is not None
        
        # Measure DURING/AFTER processing
        cpu_after = psutil.cpu_percent(interval=0.1)
        ram_after = psutil.virtual_memory().used / (1024**3)
        
        if GPU_AVAILABLE and gpus:
            gpus = GPUtil.getGPUs()
            if gpus:
                gpu = gpus[0]
                gpu_load_after = gpu.load * 100
                gpu_mem_after = gpu.memoryUsed / 1024
                gpu_temp_after = gpu.temperature
                
                # Store measurements
                gpu_loads.append(max(gpu_load_before, gpu_load_after))
                gpu_mems.append(max(gpu_mem_before, gpu_mem_after))
                gpu_temps.append(max(gpu_temp_before, gpu_temp_after))
        
        cpu_loads.append(max(cpu_before, cpu_after))
        ram_usages.append(max(ram_before, ram_after))
        
        # Print progress
        if (i + 1) % 5 == 0:
            print(f"Processed {i+1}/{num_frames} frames...")
    
    # Print results
    print("\n" + "=" * 60)
    print("HARDWARE UTILIZATION RESULTS")
    print("=" * 60)
    
    if GPU_AVAILABLE and gpu_loads:
        print(f"\n**GPU (GTX 1650):**")
        print(f"- Load: {min(gpu_loads):.1f}-{max(gpu_loads):.1f}% (avg: {np.mean(gpu_loads):.1f}%)")
        print(f"- VRAM: {min(gpu_mems):.1f}-{max(gpu_mems):.1f} GB (avg: {np.mean(gpu_mems):.1f} GB)")
        print(f"- Temperature: {min(gpu_temps):.0f}-{max(gpu_temps):.0f}°C (avg: {np.mean(gpu_temps):.0f}°C)")
    else:
        print("\n**GPU:** Not available or GPUtil not installed")
    
    print(f"\n**CPU (Ryzen 5 5600H):**")
    print(f"- Load: {min(cpu_loads):.1f}-{max(cpu_loads):.1f}% (avg: {np.mean(cpu_loads):.1f}%)")
    print(f"- RAM: {min(ram_usages):.1f}-{max(ram_usages):.1f} GB (avg: {np.mean(ram_usages):.1f} GB)")
    
    print("\n" + "=" * 60)
    print("\nCopy these values to Chapter 5, Section 5.5.3")
    print("=" * 60)

if __name__ == "__main__":
    print("Hardware Resource Utilization Measurement")
    print("=" * 60)
    print("This script measures actual GPU/CPU/RAM usage during processing.")
    print("It will process 20 frames and report average resource utilization.\n")
    
    try:
        measure_resources_during_processing(num_frames=20)
    except Exception as e:
        print(f"\nError: {e}")
        print("\nMake sure:")
        print("1. You have frames in '3D-models/Latest volley go brr/left' and 'right'")
        print("2. GPUtil is installed: pip install gputil")
        print("3. psutil is installed: pip install psutil")
