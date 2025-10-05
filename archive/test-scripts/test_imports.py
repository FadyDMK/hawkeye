#!/usr/bin/env python3
"""
Test script to verify all imports work correctly
"""
import sys
import os

# Add config to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

try:
    from camera_config import load_camera_config, CameraConfigDialog
    print("✅ camera_config imports successful")
except ImportError as e:
    print(f"❌ camera_config import failed: {e}")

try:
    from src.hawkeye_pipeline import HawkeyePipeline
    print("✅ HawkeyePipeline import successful")
except ImportError as e:
    print(f"❌ HawkeyePipeline import failed: {e}")

try:
    from src.volleyball_detection import get_ball_xy
    print("✅ volleyball_detection imports successful")
except ImportError as e:
    print(f"❌ volleyball_detection import failed: {e}")

try:
    from src.stereo_matching import StereoMatching
    print("✅ StereoMatching import successful")
except ImportError as e:
    print(f"❌ StereoMatching import failed: {e}")

print("\n🎯 All import tests completed!")
