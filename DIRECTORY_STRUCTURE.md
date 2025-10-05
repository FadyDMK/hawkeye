# 📊 Hawkeye Directory Structure

## Root Directory Layout

```
hawkeye/
├── src/                      # Main application source code
│   ├── hawkeye_launcher.py  # GUI launcher (START HERE)
│   ├── main.py              # Main entry point
│   ├── front_end.py         # Frame analyzer GUI
│   ├── hawkeye_pipeline.py  # Processing pipeline
│   ├── volleyball_detection.py
│   ├── stereo_matching.py
│   ├── camera_config.py
│   └── camera_config.json   # ⚠️ MAIN CONFIG FILE
│
├── data/                     # Video data
│   └── output_frames/       # Extracted frames
│       ├── left/            # Left camera frames (left3_XXXX.jpg)
│       └── right/           # Right camera frames (right3_XXXX.jpg)
│
├── models/                   # Trained models
│   ├── yolo/
│   └── runs/                # Training runs
│
├── dataset/                  # Training dataset
│   ├── train/
│   ├── valid/
│   └── test/
│
├── output/                   # Processing results
├── images/                   # Project images
├── docs/                     # Documentation
├── tools/                    # Utility scripts
├── thesis/                   # Thesis materials
├── presentation/             # Presentations
│
├── archive/                  # Old/experimental files
│   ├── debug-scripts/       # Debug scripts from development
│   ├── test-scripts/        # Test scripts from debugging
│   ├── old-footage/         # Old video sequences
│   ├── old-images/          # Old comparison images
│   ├── experiments/         # Experimental code
│   │   ├── depth_maps/
│   │   ├── autolabel/
│   │   └── sample_frames/
│   └── legacy_code/
│
├── hawkeye-env/             # Python virtual environment
├── README.md                # This file
├── requirements.txt         # Python dependencies
├── yolov8n.pt              # YOLO model weights
└── CLEANUP_PLAN.md         # Cleanup documentation
```

## 🚀 Quick Start

1. **Activate virtual environment:**
   ```powershell
   .\hawkeye-env\Scripts\Activate.ps1
   ```

2. **Launch the application:**
   ```powershell
   python src/hawkeye_launcher.py
   ```

3. **Or run directly:**
   ```powershell
   python src/main.py
   ```

## 📝 Important Files

### Configuration
- **`src/camera_config.json`** - Main camera configuration
  - This is the file that gets loaded by the system
  - Contains stereo camera parameters, court dimensions, detection settings

### Models
- **`yolov8n.pt`** - Base YOLO model
- **`models/runs/detect/.../weights/best.pt`** - Fine-tuned volleyball detection model

### Current Data
- **`data/output_frames/left/left3_XXXX.jpg`** - Left camera sequence (frames 0000-0099)
- **`data/output_frames/right/right3_XXXX.jpg`** - Right camera sequence (frames 0000-0099)

## 🧹 What Was Cleaned Up

### Moved to Archive:
- **Debug scripts** (15 files) → `archive/debug-scripts/`
  - analyze_*.py, debug_*.py, diagnose_*.py
  
- **Test scripts** (7 files) → `archive/test-scripts/`
  - test_*.py files used during debugging

- **Old footage** → `archive/old-footage/`
  - left.mp4, right.mp4, left1.mp4, right1.mp4, etc.
  - All .mkv test files

- **Experiments** → `archive/experiments/`
  - depth_maps/, autolabel/, sample_frames/

- **Old images** → `archive/old-images/`
  - camera_comparison.jpg

### Deleted:
- **node_modules/** - Unnecessary JavaScript dependencies
- **config/camera_config.json** - Duplicate config file (kept src/ version)

### Organized:
- **Model files** → `models/` directory
- **Training runs** → `models/runs/`

## ⚙️ System Configuration

Current setup (in `src/camera_config.json`):
- **Focal length:** 30mm (1386.67 pixels)
- **Baseline:** 3.0m
- **Depth range:** 12.0 - 22.0m
- **Court:** 40m x 31m volleyball court
- **Detection confidence:** 0.5

## 🎯 Current Status

✅ Detection working: 100% success rate  
✅ Stereo triangulation: Working correctly  
✅ Positive disparity: ~264 pixels  
✅ World coordinates: Correctly calculated  
✅ GUI functional: All bugs fixed  

## 📚 Development Notes

- The system loads config from `src/camera_config.json` (NOT from `config/` directory)
- Current test sequence: left3/right3 (100 frames, Blender rendered)
- Ball successfully detected and triangulated in 3D space
- Frames 95-99 tested and verified working

## 🔧 Troubleshooting

If you encounter issues:
1. Check `src/camera_config.json` has correct values
2. Verify frames exist in `data/output_frames/left/` and `right/`
3. Check frame naming: `left3_XXXX.jpg` and `right3_XXXX.jpg`
4. Ensure virtual environment is activated

## 📖 Documentation

- **User Guide:** `docs/README-launcher.md`
- **Cleanup Plan:** `CLEANUP_PLAN.md`
- **Thesis:** `thesis/` directory

---

*Last updated: October 5, 2025*
*Status: Production-ready after cleanup*
