# src/neural_network/infer.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2
import numpy as np
import tensorflow as tf

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "trained_model.h5"
FALLBACK_MODEL_PATH = MODELS_DIR / "trained_model_final.h5"
CLASS_MAP_PATH = MODELS_DIR / "class_map.json"

IMG_SIZE = (150, 150)

_model = None
_inv_class_map = None


def _load_model_and_classes():
    global _model, _inv_class_map

    if _model is None:
        if MODEL_PATH.exists():
            _model = tf.keras.models.load_model(MODEL_PATH)
        elif FALLBACK_MODEL_PATH.exists():
            _model = tf.keras.models.load_model(FALLBACK_MODEL_PATH)
        else:
            raise FileNotFoundError(
                "Nu găsesc models/trained_model.h5 sau models/trained_model_final.h5.\n"
                "Rulează: python -m src.neural_network.train"
            )

    if _inv_class_map is None:
        if not CLASS_MAP_PATH.exists():
            raise FileNotFoundError(
                "Nu găsesc models/class_map.json.\n"
                "Rulează: python -m src.neural_network.train"
            )

        with open(CLASS_MAP_PATH, "r", encoding="utf-8") as f:
            class_map = json.load(f)

        # class_map: { "clasa": 0, ... } -> invers: {0: "clasa"}
        _inv_class_map = {int(v): k for k, v in class_map.items()}


def _preprocess_image(image_path: str) -> np.ndarray:
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise ValueError(f"Nu pot citi imaginea: {image_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, IMG_SIZE, interpolation=cv2.INTER_AREA)

    # IMPORTANT: NU împărțim la 255 aici, pentru că modelul are Rescaling(1/255) în model.py
    img_rgb = img_rgb.astype("float32")  # modelul face Rescaling(1/255) în model.py

    # (1, 150, 150, 3)
    return np.expand_dims(img_rgb, axis=0)


def predict_image(image_path: str, top_k: int = 3) -> Dict[str, object]:
    """
    Returnează predicția modelului pentru o imagine.
    """
    _load_model_and_classes()

    x = _preprocess_image(image_path)
    probs = _model.predict(x, verbose=0)[0]  # (num_classes,)

    # siguranță: top_k să nu depășească numărul de clase
    top_k = max(1, min(int(top_k), len(probs)))

    idx_sorted = np.argsort(probs)[::-1]
    top = idx_sorted[:top_k]

    return {
        "label": _inv_class_map[int(top[0])],
        "confidence": float(probs[top[0]]),
        "top_k": [
            {"label": _inv_class_map[int(i)], "confidence": float(probs[i])}
            for i in top
        ],
    }