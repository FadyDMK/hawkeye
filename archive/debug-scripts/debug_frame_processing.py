#!/usr/bin/env python3
"""
Debug script to test single frame processing and see what's happening
"""
import sys
import os

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'config'))

from hawkeye_pipeline import HawkeyePipeline

def test_frame_processing():
    print("=== Debug: Single Frame Processing Test ===")
    
    # Create pipeline
    pipeline = HawkeyePipeline(None)
    print(f"Smoothing enabled in config: {pipeline.smoothing_enabled}")
    print(f"Smoothing alpha: {pipeline.smoothing_alpha}")
    print(f"Initial _last_world: {pipeline._last_world}")
    
    # Test frame (adjust frame number as needed)
    test_frame = 10
    
    print(f"\n--- First processing of frame {test_frame} ---")
    pipeline.clear_previous_results()
    print(f"After clear: _last_world = {pipeline._last_world}")
    
    result1 = pipeline.process_single_frame(test_frame)
    print(f"Result 1: {result1}")
    print(f"After processing: _last_world = {pipeline._last_world}")
    
    print(f"\n--- Second processing of frame {test_frame} ---")
    result2 = pipeline.process_single_frame(test_frame)
    print(f"Result 2: {result2}")
    print(f"After processing: _last_world = {pipeline._last_world}")
    
    # Compare results
    if result1 and result2:
        w1 = result1.get('world_coords')
        w2 = result2.get('world_coords')
        if w1 and w2:
            diff = [(a-b) if a and b else 'None' for a, b in zip(w1, w2)]
            print(f"\nDifference in world coords: {diff}")
    
    print("=== End Debug Test ===")

if __name__ == "__main__":
    test_frame_processing()
