"""
ONNX inference service.

Loads the exported MobileNetV2 model ONCE (called from app lifespan) and
exposes a single predict function. Everything here is CPU-based: free-tier
hosting has no GPU and onnxruntime is fast enough for one 224x224 image.

Label-order safety: labels.json is in the model's TRUE output order
(verified against ml/reports/history.json class_names). Never sort or
regenerate it; if the model changes, re-run ml/src/export_onnx.py.
"""

import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort

from ..config import settings

logger = logging.getLogger(__name__)

_session: ort.InferenceSession | None = None
_labels: list[dict] = []  # [{class, en, bn, crop}] indexed by model output


def load_model() -> None:
    """Load ONNX session + labels once. Safe to call again (idempotent)."""
    global _session, _labels

    model_path = Path(settings.model_path)
    labels_path = Path(settings.labels_path)
    if not model_path.exists():
        raise RuntimeError(
            f"Model file not found: {model_path}. "
            "See README: place model.onnx in backend/app/data/."
        )
    if not labels_path.exists():
        raise RuntimeError(f"labels.json not found at {labels_path}.")

    _session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    with open(labels_path, encoding="utf-8") as fh:
        payload = json.load(fh)
    _labels = [payload["labels"][str(i)] for i in range(payload["n_classes"])]

    if n_outputs := _session.get_outputs()[0].shape[-1]:
        if n_outputs != len(_labels):
            raise RuntimeError(
                f"Model outputs {n_outputs} classes but labels.json has "
                f"{len(_labels)} — label/model mismatch, refusing to serve."
            )
    logger.info("[inference] ONNX model loaded: %s (%d classes)", model_path.name, len(_labels))


def predict_top3(batch: np.ndarray) -> list[dict]:
    """Run the model and return the top-3 ranked predictions.

    Args:
        batch: float32 array [1, 224, 224, 3], normalized to [-1, +1].

    Returns:
        List of 3 dicts: {rank, class_name, class_name_bn, crop, confidence}.
    """
    if _session is None or not _labels:
        raise RuntimeError("Model not loaded — call load_model() first.")

    output = _session.run(None, {_session.get_inputs()[0].name: batch})[0][0]

    # Numerical softmax (the Keras export already applies softmax, but onnxruntime
    # occasionally outputs logits depending on export flags; softmax of softmax
    # output changes values negligibly, so we compute it defensively).
    exps = np.exp(output - output.max())
    probs = exps / exps.sum()

    top_idx = probs.argsort()[::-1][:3]
    return [
        {
            "rank": rank + 1,
            "class_name": _labels[i]["class"],
            "class_name_bn": _labels[i]["bn"],
            "crop": _labels[i]["crop"],
            "confidence": float(probs[i]),
        }
        for rank, i in enumerate(top_idx)
    ]
