# Hard Frames for Manual Labeling

This directory contains challenging frames automatically mined from your dataset to improve model performance.

## Structure

- `misses/` - Frames where no ball was detected at all
- `weak/` - Frames with weak detections (low confidence, small bbox, or fallback pass used)

## Current Status

- **Misses**: 0 frames (model detects all in tested range 140-199)
- **Weak**: 44 frames (mostly small bbox < 1500px², some low confidence < 0.4)

## Weak Detection Breakdown

From `weak/index.csv`:
- **Small bbox only**: 42 frames (area < 1500px²)
- **Low confidence + small bbox**: 1 frame (conf < 0.4 AND area < 1500px²)
- **Fallback used + small bbox**: 1 frame (needed larger imgsz pass)

## Next Steps for Labeling

1. **Import to Label Studio**:
   ```bash
   # Start Label Studio
   label-studio start
   
   # Create new project, upload images from weak/left/ folder
   # Use bounding box labeling interface
   ```

2. **Label Priority**:
   - Focus on frames 0000-0056 (early sequence, small balls)
   - Frame 0156 (fallback case) - especially important
   - Frames 0154-0157 (smallest areas: 721-1334px²)

3. **Add Negatives**:
   - Sample 10-20 frames with no ball (background only)
   - Include some with similar objects (chairs, equipment)

## Training Recipe

After labeling ~50-100 frames:

```bash
# Convert to YOLO format and add to dataset/train/
# Fine-tune with stronger settings:
yolo detect train data=dataset/data.yaml \
  model=runs/detect/train_finetune_fast416/weights/best.pt \
  epochs=25 imgsz=512 batch=16 workers=0 device=cpu \
  cache=ram freeze=8 patience=7 augment=True \
  name=train_hard_frames
```

## Mining Configuration

Current thresholds:
- `conf_threshold`: 0.4 (detections below this are "weak")  
- `area_threshold`: 1500.0 (bboxes smaller than this are "weak")

Adjust in `src/tools/mine_hard_frames.py` if needed.
