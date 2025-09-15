import argparse
import glob
import os
import shutil
import sys
from typing import Optional, Tuple

import cv2

# Ensure we can import from the src directory
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(THIS_DIR, "..", "src")
CONFIG_DIR = os.path.join(THIS_DIR, "..", "config")
if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)
if CONFIG_DIR not in sys.path:
    sys.path.append(CONFIG_DIR)

# Reuse your detector
from volleyball_detection import get_ball_xy, _load_detection_settings, _get_model


def get_detection_details(image) -> Tuple[Optional[int], Optional[int], float, float, bool]:
    """Get detection position, confidence, bbox area, and whether fallback was used."""
    settings = _load_detection_settings()
    model = _get_model()
    aliases = {"volleyball", "sports ball", "sports_ball", "ball"}
    
    # Primary pass
    conf_primary = settings.get("detection_conf", 0.5)
    imgsz_primary = int(settings.get("detection_imgsz_primary", settings.get("detection_imgsz", 640)))
    
    results_primary = model(image, show=False, conf=conf_primary, 
                           iou=settings.get("detection_iou", 0.5), imgsz=imgsz_primary)
    
    best_primary = None  # (score_adjusted, x, y, conf, area)
    for r in results_primary:
        names = getattr(r, "names", None) or getattr(model, "names", {})
        for box in r.boxes:
            cls_id = int(box.cls)
            name = str(names.get(cls_id, cls_id)).lower() if isinstance(names, dict) else str(cls_id)
            x_center, y_center, bw, bh = box.xywh[0]
            score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            area = float(bw * bh)
            
            if name in aliases:
                return int(x_center), int(y_center), score, area, False
            
            key = (score - 0.00001 * area)
            if best_primary is None or key > best_primary[0]:
                best_primary = (key, int(x_center), int(y_center), score, area)
    
    if best_primary is not None:
        return best_primary[1], best_primary[2], best_primary[3], best_primary[4], False
    
    # Fallback pass
    conf_fb = float(settings.get("detection_conf_fallback", max(0.1, conf_primary - 0.1)))
    imgsz_fb = int(settings.get("detection_imgsz_fallback", max(imgsz_primary, 896)))
    
    results_fb = model(image, show=False, conf=conf_fb, 
                      iou=settings.get("detection_iou", 0.5), imgsz=imgsz_fb)
    
    best_fb = None
    for r in results_fb:
        names = getattr(r, "names", None) or getattr(model, "names", {})
        for box in r.boxes:
            cls_id = int(box.cls)
            name = str(names.get(cls_id, cls_id)).lower() if isinstance(names, dict) else str(cls_id)
            x_center, y_center, bw, bh = box.xywh[0]
            score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
            area = float(bw * bh)
            
            if name in aliases:
                return int(x_center), int(y_center), score, area, True
            
            key = (score - 0.00001 * area)
            if best_fb is None or key > best_fb[0]:
                best_fb = (key, int(x_center), int(y_center), score, area)
    
    if best_fb is not None:
        return best_fb[1], best_fb[2], best_fb[3], best_fb[4], True
    
    return None, None, 0.0, 0.0, True


def mine_weak_frames(left_dir: str,
                     right_dir: str,
                     out_dir: str,
                     pattern: str = "left3_*.jpg",
                     start: Optional[int] = None,
                     end: Optional[int] = None,
                     conf_threshold: float = 0.35,
                     area_threshold: float = 1000.0) -> int:
    """Mine frames with weak detections: low confidence, small bbox, or fallback used."""
    os.makedirs(os.path.join(out_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "right"), exist_ok=True)
    index_csv = os.path.join(out_dir, "index.csv")

    left_files = sorted(glob.glob(os.path.join(left_dir, pattern)))
    # Optional frame sub-range
    if start is not None or end is not None:
        start_i = start if start is not None else 0
        end_i = end if end is not None else len(left_files)
        left_files = left_files[start_i:end_i]

    weak_count = 0
    with open(index_csv, "w", encoding="utf-8") as f:
        f.write("frame,left_path,right_path,confidence,area,fallback_used,reason\n")
        for left_path in left_files:
            basename = os.path.basename(left_path)
            # Infer right filename (replace left3_ with right3_)
            right_name = basename.replace("left3_", "right3_")
            right_path = os.path.join(right_dir, right_name)
            if not os.path.exists(right_path):
                continue

            left_img = cv2.imread(left_path)
            if left_img is None:
                continue

            x, y, conf, area, fallback_used = get_detection_details(left_img)
            
            # Determine if this is a "weak" detection
            is_weak = False
            reasons = []
            
            if x is None or y is None:
                is_weak = True
                reasons.append("no_detection")
            else:
                if conf < conf_threshold:
                    is_weak = True
                    reasons.append("low_confidence")
                if area < area_threshold:
                    is_weak = True
                    reasons.append("small_bbox")
                if fallback_used:
                    is_weak = True
                    reasons.append("fallback_used")
            
            if is_weak:
                # Copy to output for manual labeling
                shutil.copy2(left_path, os.path.join(out_dir, "left", basename))
                shutil.copy2(right_path, os.path.join(out_dir, "right", right_name))
                # Try parse frame number from filename suffix
                frame_num = os.path.splitext(basename)[0].split("_")[-1]
                reason_str = "|".join(reasons)
                f.write(f"{frame_num},{left_path},{right_path},{conf:.3f},{area:.1f},{fallback_used},{reason_str}\n")
                weak_count += 1

    return weak_count


def mine_hard_frames(left_dir: str,
                     right_dir: str,
                     out_dir: str,
                     pattern: str = "left3_*.jpg",
                     start: Optional[int] = None,
                     end: Optional[int] = None) -> int:
    os.makedirs(os.path.join(out_dir, "left"), exist_ok=True)
    os.makedirs(os.path.join(out_dir, "right"), exist_ok=True)
    index_csv = os.path.join(out_dir, "index.csv")

    left_files = sorted(glob.glob(os.path.join(left_dir, pattern)))
    # Optional frame sub-range
    if start is not None or end is not None:
        start_i = start if start is not None else 0
        end_i = end if end is not None else len(left_files)
        left_files = left_files[start_i:end_i]

    misses = 0
    with open(index_csv, "w", encoding="utf-8") as f:
        f.write("frame,left_path,right_path\n")
        for left_path in left_files:
            basename = os.path.basename(left_path)
            # Infer right filename (replace left3_ with right3_)
            right_name = basename.replace("left3_", "right3_")
            right_path = os.path.join(right_dir, right_name)
            if not os.path.exists(right_path):
                continue

            left_img = cv2.imread(left_path)
            if left_img is None:
                continue

            x, y = get_ball_xy(left_img)
            if x is None or y is None:
                # Copy to output for manual labeling
                shutil.copy2(left_path, os.path.join(out_dir, "left", basename))
                shutil.copy2(right_path, os.path.join(out_dir, "right", right_name))
                # Try parse frame number from filename suffix
                frame_num = os.path.splitext(basename)[0].split("_")[-1]
                f.write(f"{frame_num},{left_path},{right_path}\n")
                misses += 1

    return misses


def main():
    parser = argparse.ArgumentParser(description="Mine frames with no ball detection or weak detections for manual labeling")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    default_left = os.path.join(root, "..", "output_frames", "left")
    default_right = os.path.join(root, "..", "output_frames", "right")
    default_out_misses = os.path.join(root, "..", "output", "hard_frames", "misses")
    default_out_weak = os.path.join(root, "..", "output", "hard_frames", "weak")

    parser.add_argument("--mode", choices=["misses", "weak", "both"], default="both", 
                       help="Mine complete misses, weak detections, or both")
    parser.add_argument("--left_dir", default=default_left)
    parser.add_argument("--right_dir", default=default_right)
    parser.add_argument("--out_misses", default=default_out_misses)
    parser.add_argument("--out_weak", default=default_out_weak)
    parser.add_argument("--pattern", default="left3_*.jpg")
    parser.add_argument("--start", type=int, default=None, help="start index within matched files list (optional)")
    parser.add_argument("--end", type=int, default=None, help="end index within matched files list (optional)")
    parser.add_argument("--conf_threshold", type=float, default=0.35, help="confidence threshold for weak detection")
    parser.add_argument("--area_threshold", type=float, default=1000.0, help="bbox area threshold for weak detection")
    args = parser.parse_args()

    if args.mode in ["misses", "both"]:
        misses = mine_hard_frames(args.left_dir, args.right_dir, args.out_misses, args.pattern, args.start, args.end)
        print(f"Collected {misses} frames with no ball detection -> {args.out_misses}")
    
    if args.mode in ["weak", "both"]:
        weak_count = mine_weak_frames(args.left_dir, args.right_dir, args.out_weak, args.pattern, 
                                     args.start, args.end, args.conf_threshold, args.area_threshold)
        print(f"Collected {weak_count} frames with weak detections -> {args.out_weak}")


if __name__ == "__main__":
    main()
