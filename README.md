# Hawkeye: Volleyball 3D Tracking System

A computer vision system for tracking volleyball positions in 3D space using stereo vision techniques. This project implements a complete pipeline for ball detection, 3D position reconstruction, and visualization.

## Overview

The Hawkeye system uses stereo cameras to track a volleyball's position in 3D space. It processes video frames from two synchronized cameras, detects the ball, performs stereo matching, and reconstructs the ball's position in 3D world coordinates.

## Features

- **Ball Detection**: Uses computer vision techniques (including YOLOv8) to detect the volleyball in each frame
- **Stereo Matching**: Implements SGBM algorithm for disparity calculation between camera views
- **3D Reconstruction**: Converts disparity maps to 3D point clouds and extracts ball coordinates
- **Court Detection**: Identifies volleyball court boundaries and features
- **Visualization**: Provides both 2D top-down and 3D visualization of ball trajectory
- **GUI Interface**: Interactive frame selector for analyzing specific video frames
- **Position Interpolation**: Fills gaps in ball detection using polynomial interpolation
- **Error Analysis**: Tools to compare results against ground truth data

## Setup

### Requirements

```sh
python -m venv hawkeye-env
hawkeye-env\Scripts\activate     # Windows
pip install -r requirements.txt
```

### Directory Structure

```
hawkeye/
├── data/               # Input video files
├── output_frames/      # Extracted video frames
│   ├── left/           # Left camera frames
│   └── right/          # Right camera frames
├── output/             # Output files (CSV, visualizations)
└── src/                # Source code
    ├── ball_tracking_pipeline.py
    ├── front_end.py    # GUI interface
    ├── hawkeye_pipeline.py
    ├── stereo_matching.py
    ├── volleyball_detection.py
    └── court_detection/
        └── court_detection.py
```

## Usage

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

pipeline = HawkeyePipeline(None)
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

- [ ] Add automatic video frame extraction utility
- [ ] Improve ball detection robustness in occlusion scenarios
- [ ] Enhance 3D visualization with animation and trajectory lines
- [ ] Implement web-based dashboard for results?
- [ ] Improve calibration workflow and user interface
- [ ] Add unit and integration tests for pipeline modules
- [ ] Optimize performance for large datasets
- [ ] Train a better model for the ball detection