"""
Image validation + preprocessing for the serving path.

CRITICAL: normalization here must be byte-identical to training
(ml/src/augmentations.py -> resize 224x224, then scale to [-1, +1] the
MobileNetV2 way: (x / 127.5) - 1.0). Any drift between train-time and
serve-time preprocessing silently destroys accuracy.
"""

import io

import numpy as np
from PIL import Image, ImageOps

from ..config import settings


class ImageValidationError(ValueError):
    """Raised with a human-readable (EN) reason the upload was rejected."""


def validate_image(data: bytes) -> Image.Image:
    """Check size/type/dimensions and return an opened PIL image.

    Raises:
        ImageValidationError: >5 MB, wrong format, or lazy "0-byte" images.
    """
    if not data:
        raise ImageValidationError("Empty upload.")
    if len(data) > settings.max_upload_mb * 1024 * 1024:
        raise ImageValidationError(
            f"Image exceeds {settings.max_upload_mb} MB limit."
        )
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:  # noqa: BLE001 - PIL raises many types
        raise ImageValidationError("Not a valid image file (JPG/PNG only).") from exc

    if img.format not in {"JPEG", "PNG"}:
        raise ImageValidationError(
            f"Unsupported format '{img.format}' — upload a JPG or PNG leaf photo."
        )
    w, h = img.size
    if w < settings.min_image_px or h < settings.min_image_px:
        raise ImageValidationError(
            f"Image too small ({w}x{h}); minimum {settings.min_image_px}px."
        )
    return img


def preprocess(img: Image.Image) -> np.ndarray:
    """PIL image -> float32 batch [1, H, W, 3] in [-1, +1], model-ready.

    Why this exact sequence:
    1. convert to RGB (JPEG grayscale/RGBA uploads must not silently change
       channel semantics),
    2. EXIF rotation applied (phone photos often carry orientation tags),
    3. plain resize to 224x224 — the validations sets were built AFTER
       augmentation, which used a fixed crop, so serving uses plain resize,
    4. (x / 127.5) - 1.0 — identical to training normalization.
    """
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        rgb = img.convert("RGB")
    else:
        rgb = img
    size = (settings.input_size, settings.input_size)
    resized = rgb.resize(size, Image.BILINEAR)

    arr = np.asarray(resized, dtype=np.float32)        # 0..255 -> float
    arr = (arr / 127.5) - 1.0                          # -> [-1, +1]
    return np.expand_dims(arr, axis=0)                 # batch of one
