# Hawkeye - Volleyball Tracking System

A computer vision system for 3D volleyball tracking using stereo cameras and YOLO-based object detection.

## Quick Start

### 1. Activate Environment
```powershell
.\hawkeye-env\Scripts\Activate.ps1
```

### 2. Launch GUI
```powershell
python src/hawkeye_launcher.py
```

---

## Project Structure

```
hawkeye/
├── src/                       # Main application code
│   ├── hawkeye_launcher.py   # GUI launcher
│   ├── hawkeye_pipeline.py   # Processing pipeline
│   ├── front_end.py          # Frame analyzer GUI
│   ├── main.py               # Frame selector interface
│   ├── volleyball_detection.py # YOLO ball detection
│   ├── stereo_matching.py    # 3D reconstruction
│   ├── camera_config.json    # Configuration
│   ├── court_detection/      # Court detection utilities
│   └── testing/              # Test scripts
│
├── config/                    # Configuration files
│   ├── camera_config.json    # Camera/court parameters
│   └── camera_config.py      # Config loading utilities
│
├── output_frames/            # Processed frame data
│   ├── left/                 # Left camera frames
│   └── right/                # Right camera frames
│
├── models/                   # Trained YOLO models
├── dataset/                  # Training dataset
├── data/                     # Input video files
├── output/                   # Processing results
├── tools/                    # Utility scripts
├── archive/                  # Legacy/experimental code
│
├── README.md                 # This file
├── requirements.txt          # Python dependencies
└── thesis/                   # Academic documentation
```

**Detailed structure:** See [DIRECTORY_STRUCTURE.md](DIRECTORY_STRUCTURE.md)

---

## Features

- Real-time ball detection using YOLOv8
- Stereo triangulation for 3D position reconstruction
- Multi-strategy stereo matching with fallback mechanisms
- Interactive GUI for frame-by-frame analysis
- 3D trajectory visualization
- Camera calibration tools
- Batch video processing support
- Configurable detection pipeline with primary/fallback strategies

---

## Configuration

Main config file: **`src/camera_config.json`**

Key parameters:
```json
{
  "focal_length_mm": 30.0,
  "baseline_m": 3.0,
  "z_min_m": 12.0,
  "z_max_m": 22.0,
  "detection_model_path": "../models/runs/detect/retrain_with_hard_frames2/weights/best.pt",
  "detection_conf": 0.5,
  "detection_imgsz_primary": 640,
  "detection_imgsz_fallback": 896
}
```

---

## Usage

### Frame Analyzer (GUI)
1. Launch `hawkeye_launcher.py`
2. Click "Frame Analyzer"
3. Use slider to select frame
4. Click "Process Frame"
5. View 3D coordinates
6. Click "3D Visualize" to see trajectory

### Batch Processing
```python
from hawkeye_pipeline import HawkeyePipeline

pipeline = HawkeyePipeline(None)
pipeline.process_video(start_frame=0, end_frame=100)
pipeline.export_results("output/")
pipeline.visualize_results(type="3d")
```

### Camera & Court Configuration

Configure camera and court parameters:

```bash
python src/hawkeye_launcher.py
```

Click "Camera & Court Configuration" to set up:
- Camera hardware parameters (focal length, sensor size, resolution)
- Camera setup (baseline distance between cameras)
- Depth range (min/max trackable distances)
- Court dimensions (volleyball court length, width, net height)
- Stereo matching algorithm parameters

---

## Current Status

**System Status:**
- Production-ready implementation
- 98% detection success rate on validation set
- Functional stereo triangulation pipeline
- Tested on 100+ frame sequences

**Recent Updates:**
- Resolved front-end path configuration issues
- Improved detection fallback strategy

---

## Performance Results

- **Detection accuracy:** 91.1% precision, 71.1% recall on test set
- **3D reconstruction:** Median error of 3.8 cm (0.18x ball diameter)
- **Processing speed:** ~103ms per frame pair (9.7 FPS on GTX 1650)
- **End-to-end success rate:** 87% of frame pairs produce valid 3D positions

---

## Development

### Requirements
```bash
pip install -r requirements.txt
```

### Key Dependencies
- Python 3.8+
- OpenCV 4.x
- YOLOv8 (Ultralytics)
- NumPy, Matplotlib
- PyVista (3D visualization)

### Data Mining
```bash
# Mine challenging frames for model improvement
python tools/mine_hard_frames.py --mode weak --conf_threshold 0.4
```

---

## Pipeline Features

### Two-Stage Detection
- **Primary pass:** Fast inference (640px) for most cases
- **Fallback pass:** Higher resolution (896px) for difficult cases

### Robust 3D Reconstruction
1. **Detection-based triangulation:** Direct left/right detection matching
2. **ROI disparity:** Progressive window expansion around detection
3. **NCC search:** Epipolar line template matching
4. **Local high-res SGBM:** Upscaled stereo matching fallback

---

## Technical Details

### Stereo Vision Pipeline

1. **Frame Extraction:** Convert videos to individual frames
2. **Ball Detection:** Identify the ball in each camera view
3. **Stereo Matching:** Calculate disparity between left and right views
4. **Depth Calculation:** Convert disparity to depth using camera parameters
5. **3D Reconstruction:** Calculate world coordinates from image coordinates and depth
6. **Visualization:** Display ball trajectory in 2D/3D space

### Camera Calibration

The system uses a stereo calibration process to determine:
- Camera intrinsic parameters (focal length, principal point)
- Camera extrinsic parameters (rotation, translation between cameras)
- Rectification matrices for stereo matching

---

## Troubleshooting

### Frame Not Found
- Check frames exist in `output_frames/left/` and `right/`
- Verify naming: `left3_XXXX.jpg` and `right3_XXXX.jpg`

### Detection Fails
- Check `src/camera_config.json` has `detection_conf: 0.5`
- Verify model exists: `models/runs/detect/.../weights/best.pt`

### Wrong 3D Coordinates
- Verify `baseline_m: 3.0` (positive value)
- Check `z_min_m: 12.0` and `z_max_m: 22.0`
- Ensure `focal_length_px: 1600.0`

---

## Documentation

- [Thesis](thesis/) - Academic documentation
- [References](references/) - Used references

---

## Academic Context

This project is part of a BSc thesis on computer vision-based volleyball tracking systems, inspired by Hawk-Eye technology used in professional sports.

### Research Focus
- Real-time sports analytics using computer vision
- Multi-stage object detection for small/distant objects
- Robust stereo reconstruction with detection priors
- Performance optimization for resource-constrained environments

---

## Contributing

If you add new packages to the project:
```bash
pip freeze > requirements.txt
```

---

## To Do

Planned future improvements:

- [x] Add automatic video frame extraction utility
- [x] Add configurable camera and court parameters
- [x] Improve ball detection robustness in occlusion scenarios
- [x] Train a better model for ball detection
- [ ] Enhance 3D visualization with animation and trajectory lines
- [ ] Improve calibration workflow and user interface

---

## License

Academic project - See thesis documentation for details.

---

## Author

FadyDMK  
BSc Computer Science Thesis Project  
2025/2026

---

For detailed technical documentation, see the thesis folder.
