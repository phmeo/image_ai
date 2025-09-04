from typing import List, Dict

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image


# COCO labels for SSD MobileNet V2 from TF Hub (91 indexed to 90; model returns indices)
# We'll load from the model signature's class labels when possible; fallback hardcoded short list.

_DETECTOR = None

# COCO 2017 label map (subset with holes kept as dict)
_COCO_LABELS: Dict[int, str] = {
    1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus", 7: "train", 8: "truck", 9: "boat",
    10: "traffic light", 11: "fire hydrant", 13: "stop sign", 14: "parking meter", 15: "bench",
    16: "bird", 17: "cat", 18: "dog", 19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
    34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat", 40: "baseball glove",
    41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle", 46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl",
    52: "banana", 53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut", 61: "cake",
    62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 67: "dining table", 70: "toilet",
    72: "tv", 73: "laptop", 74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone",
    78: "microwave", 79: "oven", 80: "toaster", 81: "sink", 82: "refrigerator",
    84: "book", 85: "clock", 86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}


def _load_detector():
    global _DETECTOR
    if _DETECTOR is None:
        # SSD MobileNet V2 FPNLite 640x640
        _DETECTOR = hub.load("https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_640x640/1")
    return _DETECTOR


def detect_objects(image: Image.Image, score_threshold: float = 0.4, max_results: int = 50) -> Dict:
    model = _load_detector()

    rgb = image.convert("RGB")
    img_arr = np.array(rgb)
    input_tensor = tf.convert_to_tensor(img_arr, dtype=tf.uint8)
    input_tensor = input_tensor[tf.newaxis, ...]

    outputs = model(input_tensor)
    # outputs: dict with 'detection_boxes', 'detection_scores', 'detection_classes', 'num_detections'
    boxes = outputs["detection_boxes"][0].numpy()  # yMin, xMin, yMax, xMax (normalized)
    scores = outputs["detection_scores"][0].numpy()
    classes = outputs["detection_classes"][0].numpy().astype(np.int32)

    h, w = img_arr.shape[:2]
    results = []
    for i in range(min(len(scores), max_results)):
        if scores[i] < score_threshold:
            continue
        y_min, x_min, y_max, x_max = boxes[i]
        # normalized box (0-1)
        boxn = [float(x_min), float(y_min), float(x_max), float(y_max)]
        # pixel box
        box_px = [int(x_min * w), int(y_min * h), int(x_max * w), int(y_max * h)]  # x1,y1,x2,y2
        class_id = int(classes[i])
        label = _COCO_LABELS.get(class_id, f"id {class_id}")
        results.append({
            "box": box_px,
            "boxn": boxn,
            "score": float(scores[i]),
            "class_id": class_id,
            "label": label,
        })

    return {"detections": results, "width": w, "height": h} 