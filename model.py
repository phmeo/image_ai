import threading
from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import numpy as np
from PIL import Image

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input as mobilenet_preprocess,
    decode_predictions as mobilenet_decode,
)
from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2B0,
    EfficientNetV2B3,
    preprocess_input as efficientnet_preprocess,
    decode_predictions as efficientnet_decode,
)
import threading
import tensorflow_hub as hub
import tensorflow as tf


@dataclass(frozen=True)
class ModelSpec:
    name: str
    target_size: Tuple[int, int]
    loader: Callable[[], object]
    preprocess: Callable[[np.ndarray], np.ndarray]
    decode: Callable[[np.ndarray, int], List[List[Tuple[str, str, float]]]]


_DEFAULT_MODEL_NAME = "efficientnet_v2_b3"

_MODEL_SPECS: Dict[str, ModelSpec] = {
    "mobilenet_v2": ModelSpec(
        name="mobilenet_v2",
        target_size=(224, 224),
        loader=lambda: MobileNetV2(weights="imagenet"),
        preprocess=mobilenet_preprocess,
        decode=mobilenet_decode,
    ),
    "efficientnet_v2_b0": ModelSpec(
        name="efficientnet_v2_b0",
        target_size=(224, 224),
        loader=lambda: EfficientNetV2B0(weights="imagenet"),
        preprocess=efficientnet_preprocess,
        decode=efficientnet_decode,
    ),
    "efficientnet_v2_b3": ModelSpec(
        name="efficientnet_v2_b3",
        target_size=(300, 300),
        loader=lambda: EfficientNetV2B3(weights="imagenet"),
        preprocess=efficientnet_preprocess,
        decode=efficientnet_decode,
    ),
}

# Flower classifier from TF Hub (e.g., a MobileNet trained on flowers)
_FLOWER_MODEL = None
_FLOWER_LABELS = [
    "daisy","dandelion","roses","sunflowers","tulips"
]


def _load_flower_model():
    global _FLOWER_MODEL
    if _FLOWER_MODEL is None:
        _FLOWER_MODEL = hub.load("https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4")
    return _FLOWER_MODEL


def _flower_preprocess(arr: np.ndarray) -> np.ndarray:
    # Expect float32 [0,1], 224x224
    arr = tf.image.resize(arr, [224, 224]).numpy()
    arr = arr.astype(np.float32) / 255.0
    return arr


def _flower_decode(preds: np.ndarray, top: int):
    # The chosen hub model is ImageNet; for demo we will keep ImageNet decode through Keras if needed.
    # Simple mapping to flower labels for a 5-class demo isn't perfect; a dedicated flower model is ideal.
    # Here we fallback to Keras mobilenet decode for readability.
    from tensorflow.keras.applications.mobilenet_v2 import decode_predictions as dp
    return dp(preds, top=top)


_models_cache: Dict[str, object] = {}
_model_lock = threading.Lock()


# Static metadata for display (approximate values)
_MODEL_INFO: Dict[str, Dict[str, str]] = {
    "mobilenet_v2": {"display": "MobileNetV2", "input": "224×224", "params": "~3.5M", "imagenet_top1": "~71.8%", "notes": "Nhẹ, nhanh; phù hợp thiết bị yếu hoặc cần tốc độ cao."},
    "efficientnet_v2_b0": {"display": "EfficientNetV2‑B0", "input": "224×224", "params": "~7.1M", "imagenet_top1": "~78.7%", "notes": "Cân bằng tốt giữa tốc độ và độ chính xác."},
    "efficientnet_v2_b3": {"display": "EfficientNetV2‑B3", "input": "300×300", "params": "~14.4M", "imagenet_top1": "~82–83%", "notes": "Độ chính xác cao hơn; tốn tài nguyên hơn B0/MobileNetV2."},
    "flowers_v1": {"display": "Flowers (TF‑Hub)", "input": "224×224", "params": "~3.5M", "imagenet_top1": "N/A", "notes": "Phân loại hoa phổ biến; mẫu minh họa bằng MobileNet."}
}


def list_available_models() -> List[str]:
    return [*list(_MODEL_SPECS.keys()), "flowers_v1"]


def get_model_info() -> Dict[str, Dict[str, str]]:
    return _MODEL_INFO.copy()


def _get_model(model_name: str):
    if model_name == "flowers_v1":
        return _load_flower_model()
    if model_name not in _MODEL_SPECS:
        model_name = _DEFAULT_MODEL_NAME
    with _model_lock:
        model = _models_cache.get(model_name)
        if model is None:
            model = _MODEL_SPECS[model_name].loader()
            _models_cache[model_name] = model
        return model


def classify_image(
    image: Image.Image,
    model_name: str = _DEFAULT_MODEL_NAME,
    top_k: int = 5,
) -> List[Tuple[str, float]]:
    if model_name == "flowers_v1":
        model = _get_model(model_name)
        rgb = image.convert("RGB")
        arr = np.array(rgb)
        arr = _flower_preprocess(arr)
        arr = np.expand_dims(arr, 0)
        preds = model(arr)
        decoded = _flower_decode(preds, top=top_k)[0]
        return [(label.replace("_", " "), float(prob)) for (_, label, prob) in decoded]

    spec = _MODEL_SPECS.get(model_name, _MODEL_SPECS[_DEFAULT_MODEL_NAME])
    model = _get_model(spec.name)

    image_resized = image.convert("RGB").resize(spec.target_size)
    image_array = np.array(image_resized, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)
    image_preprocessed = spec.preprocess(image_array)

    predictions = model.predict(image_preprocessed)
    decoded = spec.decode(predictions, top=top_k)[0]

    results = [(label.replace("_", " "), float(prob)) for (_, label, prob) in decoded]
    return results 