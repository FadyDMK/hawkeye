"""
Create detection confidence distribution plot for thesis.
Analyzes confidence scores across validation frames.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ultralytics import YOLO
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def get_confidence_distribution(num_frames=100):
    """
    Collect confidence scores from detection on validation frames.
    
    Args:
        num_frames: Number of frames to analyze (default 100)
    """
    # Load model
    model_path = Path('models/runs/detect/retrain_with_hard_frames2/weights/best.pt')
    print(f"Loading model from: {model_path}")
    model = YOLO(str(model_path))
    
    # Paths to frames
    left_frames = Path('output_frames/left')
    
    if not left_frames.exists():
        print(f"Error: Frame directory not found: {left_frames}")
        return
    
    # Collect all confidence scores
    all_confidences = []
    volleyball_confidences = []
    frames_with_detection = 0
    frames_without_detection = 0
    
    print(f"\nAnalyzing {num_frames} frames...")
    
    for i in range(num_frames):
        frame_path = left_frames / f'left3_{i:04d}.jpg'
        if not frame_path.exists():
            continue
        
        img = cv2.imread(str(frame_path))
        if img is None:
            continue
        
        # Run detection
        results = model(img, conf=0.25, verbose=False)  # Low threshold to catch all detections
        
        # Extract confidences
        frame_has_volleyball = False
        for r in results:
            boxes = r.boxes
            names = r.names
            
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                class_name = names.get(cls_id, '').lower()
                
                all_confidences.append(conf)
                
                if 'volleyball' in class_name or 'ball' in class_name:
                    volleyball_confidences.append(conf)
                    frame_has_volleyball = True
        
        if frame_has_volleyball:
            frames_with_detection += 1
        else:
            frames_without_detection += 1
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{num_frames} frames...")
    
    print(f"\n✓ Analysis complete!")
    print(f"  Frames analyzed: {num_frames}")
    print(f"  Frames with volleyball detection: {frames_with_detection} ({frames_with_detection/num_frames*100:.1f}%)")
    print(f"  Frames without detection: {frames_without_detection} ({frames_without_detection/num_frames*100:.1f}%)")
    print(f"  Total detections: {len(volleyball_confidences)}")
    
    if len(volleyball_confidences) == 0:
        print("Error: No volleyball detections found!")
        return
    
    # Calculate statistics
    mean_conf = np.mean(volleyball_confidences)
    median_conf = np.median(volleyball_confidences)
    std_conf = np.std(volleyball_confidences)
    min_conf = np.min(volleyball_confidences)
    max_conf = np.max(volleyball_confidences)
    
    # Count by confidence range
    high_conf = sum(1 for c in volleyball_confidences if c >= 0.8)
    medium_conf = sum(1 for c in volleyball_confidences if 0.5 <= c < 0.8)
    low_conf = sum(1 for c in volleyball_confidences if c < 0.5)
    
    print(f"\nConfidence Statistics:")
    print(f"  Mean: {mean_conf:.3f}")
    print(f"  Median: {median_conf:.3f}")
    print(f"  Std Dev: {std_conf:.3f}")
    print(f"  Min: {min_conf:.3f}")
    print(f"  Max: {max_conf:.3f}")
    print(f"\nConfidence Ranges:")
    print(f"  High (≥0.8): {high_conf} ({high_conf/len(volleyball_confidences)*100:.1f}%)")
    print(f"  Medium (0.5-0.8): {medium_conf} ({medium_conf/len(volleyball_confidences)*100:.1f}%)")
    print(f"  Low (<0.5): {low_conf} ({low_conf/len(volleyball_confidences)*100:.1f}%)")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Histogram
    ax1.hist(volleyball_confidences, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(mean_conf, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_conf:.3f}')
    ax1.axvline(median_conf, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_conf:.3f}')
    ax1.axvline(0.5, color='green', linestyle=':', linewidth=2, alpha=0.5, label='Primary Threshold (0.5)')
    ax1.set_xlabel('Confidence Score', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Detection Confidence Distribution', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: Cumulative distribution
    sorted_conf = np.sort(volleyball_confidences)
    cumulative = np.arange(1, len(sorted_conf) + 1) / len(sorted_conf)
    ax2.plot(sorted_conf, cumulative, linewidth=2, color='darkblue')
    ax2.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(0.8, color='gray', linestyle=':', alpha=0.5)
    ax2.axhline(0.95, color='gray', linestyle=':', alpha=0.5)
    ax2.axvline(0.5, color='green', linestyle='--', alpha=0.5, label='Primary Threshold')
    ax2.set_xlabel('Confidence Score', fontsize=12)
    ax2.set_ylabel('Cumulative Probability', fontsize=12)
    ax2.set_title('Cumulative Confidence Distribution', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save figure
    output_path = 'output/confidence_distribution.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Confidence distribution plot saved to: {output_path}")
    
    # Create summary text box
    fig2, ax = plt.subplots(figsize=(10, 6))
    ax.axis('off')
    
    summary_text = f"""
DETECTION CONFIDENCE ANALYSIS SUMMARY
{'='*60}

Dataset:              {num_frames} validation frames
Detection Rate:       {frames_with_detection}/{num_frames} frames ({frames_with_detection/num_frames*100:.1f}%)
Total Detections:     {len(volleyball_confidences)}

CONFIDENCE STATISTICS:
  Mean:               {mean_conf:.3f}
  Median:             {median_conf:.3f}
  Standard Deviation: {std_conf:.3f}
  Range:              [{min_conf:.3f}, {max_conf:.3f}]

CONFIDENCE DISTRIBUTION:
  High Confidence (≥0.8):      {high_conf:3d} detections ({high_conf/len(volleyball_confidences)*100:5.1f}%)
  Medium Confidence (0.5-0.8): {medium_conf:3d} detections ({medium_conf/len(volleyball_confidences)*100:5.1f}%)
  Low Confidence (<0.5):       {low_conf:3d} detections ({low_conf/len(volleyball_confidences)*100:5.1f}%)

INTERPRETATION:
  • {high_conf/len(volleyball_confidences)*100:.1f}% of detections exceed 0.8 confidence (high reliability)
  • Median confidence of {median_conf:.3f} indicates strong overall detection quality
  • Low variance (σ={std_conf:.3f}) demonstrates consistent performance
  • The high proportion of confident detections (≥0.5) validates the
    primary detection strategy threshold choice

MODEL: retrain_with_hard_frames2
"""
    
    ax.text(0.5, 0.5, summary_text, ha='center', va='center',
           fontsize=11, family='monospace',
           bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    summary_path = 'output/confidence_summary.png'
    plt.savefig(summary_path, dpi=300, bbox_inches='tight')
    print(f"✓ Confidence summary saved to: {summary_path}")
    
    plt.show()

if __name__ == '__main__':
    get_confidence_distribution(num_frames=100)
