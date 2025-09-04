import os
import glob
from dataclasses import dataclass
from typing import Dict, Any, List

from ultralytics import YOLO


@dataclass
class DetectionResult:
    output_path: str
    classes: List[str]
    confs: List[float]
    model: str


class YOLODetector:
    def __init__(self, model_name: str = "yolov8n.pt") -> None:
        # Lazy load the model to avoid slow import time on app start
        self._model_name = model_name
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is None:
            self._model = YOLO(self._model_name)

    def predict(self, source_path: str, outputs_root: str, conf: float = 0.35, iou: float = 0.45) -> Dict[str, Any]:
        """
        Run prediction on an image or video. The annotated output is saved under
        outputs_root/<run_name>/pred/ with the same base filename.
        Returns a dict with output_path, classes, confs, model.
        """
        self._ensure_loaded()

        run_name = os.path.splitext(os.path.basename(source_path))[0]
        save_project = os.path.join(outputs_root, run_name)
        os.makedirs(save_project, exist_ok=True)

        results = self._model.predict(
            source=source_path,
            conf=conf,
            iou=iou,
            imgsz=640,
            save=True,
            project=save_project,
            name="pred",
            exist_ok=True,
            verbose=False,
        )

        # Collect classes and confidences from first result
        classes: List[str] = []
        confs: List[float] = []
        if len(results) > 0 and hasattr(results[0], "boxes") and results[0].boxes is not None:
            names = results[0].names if hasattr(results[0], "names") else {}
            boxes = results[0].boxes
            try:
                cls_ids = boxes.cls.tolist() if hasattr(boxes, "cls") else []
                conf_vals = boxes.conf.tolist() if hasattr(boxes, "conf") else []
            except Exception:
                cls_ids = []
                conf_vals = []
            for idx, conf_v in zip(cls_ids, conf_vals):
                label = names.get(int(idx), str(int(idx))) if isinstance(names, dict) else str(int(idx))
                classes.append(label)
                try:
                    confs.append(float(conf_v))
                except Exception:
                    confs.append(0.0)

        # Determine output file path saved by ultralytics
        pred_dir = os.path.join(save_project, "pred")
        base = os.path.basename(source_path)
        stem, _ = os.path.splitext(base)

        # Search for a saved file matching the stem with any extension
        candidates = glob.glob(os.path.join(pred_dir, f"{stem}.*"))
        if not candidates:
            # Fallback: pick any file in pred_dir
            files = sorted(glob.glob(os.path.join(pred_dir, "*")))
            if not files:
                raise RuntimeError("Prediction completed but no output file was found.")
            output_path = files[0]
        else:
            # Prefer common image/video extensions ordering
            preferred_exts = [".jpg", ".png", ".mp4", ".avi", ".mov", ".mkv", ".webm", ".bmp", ".jpeg", ".webp"]
            ranked = sorted(
                candidates,
                key=lambda p: preferred_exts.index(os.path.splitext(p)[1].lower()) if os.path.splitext(p)[1].lower() in preferred_exts else 999,
            )
            output_path = ranked[0]

        return {
            "output_path": output_path,
            "classes": classes,
            "confs": confs,
            "model": self._model_name,
        } 