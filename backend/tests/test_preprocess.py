"""Unit tests for image validation and preprocessing (no app startup)."""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path
from tests.conftest import make_png  # noqa: E402

from app.config import settings  # noqa: E402
from app.services.preprocess import ImageValidationError, preprocess, validate_image


def test_valid_png_passes_and_normalizes():
    """A decent-size PNG must open, resize to 224 and land in [-1, +1]."""
    data = make_png(300, 300)
    img = validate_image(data)
    batch = preprocess(img)
    assert batch.shape == (1, settings.input_size, settings.input_size, 3)
    assert batch.dtype == np.float32
    assert batch.min() >= -1.0 and batch.max() <= 1.0


def test_too_large_image_rejected():
    """Uploads over the limit must raise, not crash the endpoint."""
    big = b"x" * (settings.max_upload_mb * 1024 * 1024 + 1)
    with pytest.raises(ImageValidationError, match="exceeds"):
        validate_image(big)


def test_garbage_bytes_rejected():
    with pytest.raises(ImageValidationError, match="valid image"):
        validate_image(b"not-an-image")


def test_tiny_image_rejected():
    data = make_png(10, 10)  # below min_image_px
    with pytest.raises(ImageValidationError, match="too small"):
        validate_image(data)


def test_grayscale_converted_to_rgb():
    """Grayscale uploads must become 3 channels like the model expects."""
    import io

    from PIL import Image
    img = Image.new("L", (256, 256), 128)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    batch = preprocess(validate_image(buf.getvalue()))
    assert batch.shape[-1] == 3
