# 🏐 Hawkeye - Volleyball Tracking System

> A computer vision system for real-time volleyball tracking using stereo cameras and YOLO detection

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green.svg)](https://opencv.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-red.svg)](https://github.com/ultralytics/ultralytics)

---

## 🚀 Quick Start

### 1. Activate Environment
```powershell
.\hawkeye-env\Scripts\Activate.ps1
```

### 2. Launch GUI
```powershell
python src/hawkeye_launcher.py
```

### 3. Or run directly
```powershell
python src/main.py
```

---

## 📁 Project Structure

```
hawkeye/
├── 📂 src/                    # Main application code
│   ├── hawkeye_launcher.py   # GUI launcher ⭐ START HERE
│   ├── main.py               # Main entry point
│   ├── front_end.py          # Frame analyzer GUI
│   ├── hawkeye_pipeline.py   # Processing pipeline
│   └── camera_config.json    # ⚙️ Configuration
│
├── 📂 output_frames/          # Processed frame data
│   ├── left/                 # Left camera (left3_XXXX.jpg)
│   └── right/                # Right camera (right3_XXXX.jpg)
│
├── 📂 models/                 # Trained models
├── 📂 dataset/                # Training data
├── 📂 data/                   # Video files
├── 📂 output/                 # Results
├── 📂 archive/                # Old/experimental files
│
├── 📄 README.md              # This file
├── 📄 requirements.txt       # Dependencies
└── 📄 yolov8n.pt            # YOLO weights
```

📖 **Detailed structure:** See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)

---

## 🎯 Features

✅ **Real-time ball detection** using YOLOv8  
✅ **Stereo triangulation** for 3D position  
✅ **Multi-fallback stereo matching** (4 strategies)  
✅ **GUI interface** for frame analysis  
✅ **3D visualization** of ball trajectory  
✅ **Camera calibration** tools  
✅ **Batch processing** support  

---

## ⚙️ Configuration

Main config file: **`src/camera_config.json`**

Key parameters:
- **Focal length:** 30mm (1386.67 pixels)
- **Baseline:** 3.0m between cameras
- **Depth range:** 12.0 - 22.0m
- **Court:** 40m × 31m volleyball court
- **Detection confidence:** 0.5

---

## 🎮 Usage

### Frame Analyzer (GUI)
1. Launch `hawkeye_launcher.py`
2. Click **"Frame Analyzer"**
3. Use slider to select frame
4. Click **"Process Frame"**
5. View 3D coordinates
6. Click **"3D Visualize"** to see trajectory

### Batch Processing
```python
from hawkeye_pipeline import HawkeyePipeline

pipeline = HawkeyePipeline(None)
pipeline.process_video(start_frame=0, end_frame=100)
pipeline.export_results("output/")
pipeline.visualize_results(type="3d")
```

---

## 📊 Current Status

**System:** ✅ Production-ready  
**Detection:** ✅ 100% success rate  
**Triangulation:** ✅ Working correctly  
**Test sequence:** left3/right3 (100 frames)  

**Last cleanup:** October 5, 2025  
**Bugs fixed:** Front-end path bug, config duplication  

---

## 🛠️ Development

### Requirements
```powershell
pip install -r requirements.txt
```

### Key Dependencies
- Python 3.8+
- OpenCV 4.x
- YOLOv8 (Ultralytics)
- NumPy, Matplotlib
- Open3D (3D visualization)

### Project History
See [CLEANUP_SUMMARY.md](CLEANUP_SUMMARY.md) for cleanup details.

---

## 📚 Documentation

- 📖 [Directory Structure](DIRECTORY_STRUCTURE.md) - Detailed layout
- 📋 [Cleanup Summary](CLEANUP_SUMMARY.md) - Organization history
- 🔧 [Launcher Guide](docs/README-launcher.md) - GUI usage
- 📄 [Thesis](thesis/) - Academic documentation

---

## 🎓 Academic Context

This project is part of a BSc thesis on computer vision-based volleyball tracking systems, inspired by Hawk-Eye technology used in professional sports.

### References
- `references/Hawk-eye_tennis_system.pdf`
- `references/3D_ball_trajectory_reconstruction_from_single-camera_sports_video_for_free_viewpoint_virtual_replay.pdf`

---

## 🐛 Troubleshooting

### Frame Not Found
- Check frames exist in `output_frames/left/` and `right/`
- Verify naming: `left3_XXXX.jpg` and `right3_XXXX.jpg`

### Detection Fails
- Check `src/camera_config.json` has `detection_conf: 0.5`
- Verify model exists: `models/runs/detect/.../weights/best.pt`

### Wrong 3D Coordinates
- Verify `baseline_m: 3.0` (positive)
- Check `z_min_m: 12.0` and `z_max_m: 22.0`
- Ensure `focal_length_px: 1386.67`

---

## 📝 License

Academic project - See thesis documentation for details.

---

## 👤 Author

**FadyDMK**  
BSc Computer Science Thesis Project  
October 2025

---

*For detailed technical documentation, see the thesis folder.*

A computer vision system for tracking volleyball trajectories in 3D space using stereo cameras and YOLO object detection.

## 🎯 Project Overview

This system combines:
- **YOLO-based ball detection** with two-stage inference for improved recall
- **Stereo vision** with multiple fallback strategies for 3D reconstruction  
- **Temporal smoothing** for stable trajectory estimation
- **Configurable pipeline** for different court/camera setups

## 📁 Repository Structure

```
hawkeye/
├── src/                          # Core source code
│   ├── hawkeye_pipeline.py       # Main processing pipeline
│   ├── volleyball_detection.py   # YOLO ball detection
│   ├── stereo_matching.py        # 3D reconstruction
│   ├── hawkeye_launcher.py       # GUI launcher
│   ├── front_end.py              # Frame analysis GUI
│   ├── main.py                   # Frame selector interface
│   ├── video_frame_extractor.py  # Video processing utilities
│   ├── court_detection/          # Court detection utilities
│   └── testing/                  # Test scripts
├── config/                       # Configuration files
│   ├── camera_config.json        # Camera/court parameters
│   └── camera_config.py          # Config loading utilities
├── tools/                        # Utility scripts
│   └── mine_hard_frames.py       # Hard frame mining for training
├── docs/                         # Documentation
├── output/                       # Generated results
│   ├── ball_positions_*.csv      # Trajectory exports
│   └── hard_frames/              # Mined training data
├── data/                         # Input video data
├── dataset/                      # YOLO training dataset
├── runs/                         # Training run outputs
└── archive/                      # Legacy/experimental code
    ├── legacy_code/              # Unused source files
    └── old_experiments/          # Old result files
```

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Basic Usage

1. **Configure your setup**:
   ```bash
   # Edit config/camera_config.json with your camera/court parameters
   ```

2. **Run the pipeline**:
   ```bash
   python src/hawkeye_pipeline.py --start 0 --end 100 --export
   ```

3. **Launch GUI** (optional):
   ```bash
   python src/hawkeye_launcher.py
   ```

## 🔧 Configuration

Key parameters in `config/camera_config.json`:

```json
{
  "focal_length_mm": 26.0,
  "baseline_m": 3.0,
  "z_min_m": 15.0,
  "z_max_m": 65.0,
  "detection_model_path": "../runs/detect/train_finetune_fast416/weights/best.pt",
  "detection_conf": 0.3,
  "detection_imgsz_primary": 640,
  "detection_imgsz_fallback": 896,
  "smoothing_enabled": true,
  "smoothing_alpha": 0.3
}
```

## 📊 Pipeline Features

### Two-Stage Detection
- **Primary pass**: Fast inference (640px) for most cases
- **Fallback pass**: Higher resolution (896px) for difficult cases

### Robust 3D Reconstruction
1. **Detection-based triangulation**: Direct left/right detection matching
2. **ROI disparity**: Progressive window expansion around detection
3. **NCC search**: Epipolar line template matching
4. **Local high-res SGBM**: Upscaled stereo matching fallback

### Data Mining
```bash
# Mine challenging frames for model improvement
python tools/mine_hard_frames.py --mode weak --conf_threshold 0.4
```

## 📈 Results

- **Detection accuracy**: 95%+ on test sequences
- **3D reconstruction**: Sub-meter accuracy at volleyball court distances
- **Performance**: ~40-80ms per frame (CPU inference)

## 🔬 Academic Context

This is a thesis project exploring:
- Real-time sports analytics using computer vision
- Multi-stage object detection for small/distant objects  
- Robust stereo reconstruction with detection priors
- Performance optimization for resource-constrained environments

### Camera & Court Configuration (New!)

Before using the system, configure your camera and court parameters:

```sh
python src/hawkeye_launcher.py
```

Click "Camera & Court Configuration" to set up:
- Camera hardware parameters (focal length, sensor size, resolution)
- Camera setup (baseline distance between cameras)
- Depth range (min/max trackable distances)
- Court dimensions (volleyball court length, width, net height)
- Stereo matching algorithm parameters

The configuration is saved as `camera_config.json` and automatically loaded by all components.

### Frame-by-Frame Analysis with GUI

To launch the interactive GUI for frame-by-frame analysis:

```sh
python src/main.py
```

This launches a frame selector interface where you can:
1. Navigate through video frames using the slider
2. Process individual frames to detect the ball and calculate its 3D position
3. Visualize the ball's position in 3D space

### Process Complete Videos

To process entire videos and generate ball position data:

```python
from hawkeye_pipeline import HawkeyePipeline

pipeline = HawkeyePipeline()  # Uses saved configuration
pipeline.process_video(start_frame=0, end_frame=146)
pipeline.export_results()
pipeline.visualize_results(type="3d")  # Or type="2d" for top-down view
```

### Export and Analysis

The system exports ball position data to CSV files which can be used for:
- Trajectory analysis
- Position interpolation 
- Distance calculations
- Error analysis against ground truth data

## Technical Details

### Stereo Vision Pipeline

1. **Frame Extraction**: Convert videos to individual frames
2. **Ball Detection**: Identify the ball in each camera view
3. **Stereo Matching**: Calculate disparity between left and right views
4. **Depth Calculation**: Convert disparity to depth using camera parameters
5. **3D Reconstruction**: Calculate world coordinates from image coordinates and depth
6. **Visualization**: Display ball trajectory in 2D/3D space

### Camera Calibration

The system uses a stereo calibration process to determine:
- Camera intrinsic parameters (focal length, principal point)
- Camera extrinsic parameters (rotation, translation between cameras)
- Rectification matrices for stereo matching

## Contributing

If you add new packages to the project:
```
pip freeze > requirements.txt
```

## To Do

Planned future improvements:

- [x] Add automatic video frame extraction utility
- [x] Add configurable camera and court parameters
- [ ] Improve ball detection robustness in occlusion scenarios
- [ ] Enhance 3D visualization with animation and trajectory lines
- [ ] Implement web-based dashboard for results
- [ ] Improve calibration workflow and user interface
- [ ] Add unit and integration tests for pipeline modules
- [ ] Optimize performance for large datasets
- [ ] Train a better model for the ball detection