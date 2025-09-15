from ultralytics import YOLO
import cv2
import os
from pathlib import Path
import json


_MODEL = None


def _load_detection_settings():
    """Load model path and thresholds from camera_config.json if present."""
    cfg_path = Path(__file__).parent.parent / "config" / "camera_config.json"
    defaults = {
        "detection_model_path": str(Path(__file__).parent.parent / "runs" / "detect" / "train18" / "weights" / "best.pt"),
    "detection_conf": 0.5,
        "detection_iou": 0.5,
    "detection_imgsz": 640,
    "detection_imgsz_primary": 640,
    "detection_imgsz_fallback": 896,
    "detection_conf_fallback": 0.3,
    }
    try:
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)
            for k in defaults:
                if k in cfg:
                    defaults[k] = cfg[k]
    except Exception:
        pass
    # Normalize path
    p = Path(defaults["detection_model_path"])
    if not p.is_absolute():
        p = (Path(__file__).parent / p).resolve()
    defaults["detection_model_path"] = str(p)
    return defaults


def _get_model():
    global _MODEL
    if _MODEL is None:
        settings = _load_detection_settings()
        weights = Path(settings["detection_model_path"]).resolve()
        # Try configured weights; fallback to local yolov8n.pt or default if not found/failed
        try:
            if weights.exists():
                _MODEL = YOLO(str(weights))
            else:
                raise FileNotFoundError(str(weights))
        except Exception:
            fallback_local = Path(__file__).parent.parent / "yolov8n.pt"
            try:
                _MODEL = YOLO(str(fallback_local if fallback_local.exists() else "yolov8n.pt"))
            except Exception:
                # Last resort: try tiny model name
                _MODEL = YOLO("yolov8n.pt")
    return _MODEL


def get_ball_xy(image, conf: float | None = None):
    settings = _load_detection_settings()
    model = _get_model()
    aliases = {"volleyball", "sports ball", "sports_ball", "ball"}

    def infer_once(imgsz_val: int, conf_val: float):
        results_local = model(image, show=False, conf=conf_val, iou=settings.get("detection_iou", 0.5), imgsz=imgsz_val)
        best_local = None  # (score_adjusted, x, y)
        for r in results_local:
            names = getattr(r, "names", None) or getattr(model, "names", {})
            for box in r.boxes:
                cls_id = int(box.cls)
                name = str(names.get(cls_id, cls_id)).lower() if isinstance(names, dict) else str(cls_id)
                x_center, y_center, bw, bh = box.xywh[0]
                score = float(box.conf[0]) if hasattr(box, "conf") else 0.0
                if name in aliases:
                    return (int(x_center), int(y_center))
                # candidate fallback: prefer smaller boxes with higher score
                area = float(bw * bh)
                key = (score - 0.00001 * area)
                if best_local is None or key > best_local[0]:
                    best_local = (key, int(x_center), int(y_center))
        if best_local is not None:
            return (best_local[1], best_local[2])
        return (None, None)

    # Primary fast pass
    conf_primary = settings.get("detection_conf", 0.5) if conf is None else conf
    imgsz_primary = int(settings.get("detection_imgsz_primary", settings.get("detection_imgsz", 640)))
    x, y = infer_once(imgsz_primary, conf_primary)
    if x is not None:
        return (x, y)

    # Fallback larger pass
    conf_fb = float(settings.get("detection_conf_fallback", max(0.1, conf_primary - 0.1)))
    imgsz_fb = int(settings.get("detection_imgsz_fallback", max(imgsz_primary, 896)))
    return infer_once(imgsz_fb, conf_fb)





if __name__ == "__main__":
    # testing the function
    root = os.path.dirname(os.path.abspath(__file__))
    left_path = os.path.join(root, "../output_frames/left/left3_0104.jpg")
    right_path = os.path.join(root, "../output_frames/right/right3_0104.jpg")
    if not os.path.exists(left_path) or not os.path.exists(right_path):
        raise FileNotFoundError("One or both image files do not exist. Please check the file paths.")
    left = cv2.imread(left_path)
    right = cv2.imread(right_path)
    xy_left = get_ball_xy(left)
    xy_right = get_ball_xy(right)
    print(f"Ball coordinates in left image: {xy_left}")
    print(f"Ball coordinates in right image: {xy_right}")