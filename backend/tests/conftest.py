"""
Shared pytest fixtures.

Uses the REAL model.onnx from backend/app/data and a disposable SQLite
database (env DATABASE_URL override) so the API tests exercise the actual
inference path end to end — no mocks for anything production-critical.
"""

import os
import tempfile
from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

# A temp DB keeps scan-history tests isolated from any dev data.
_TMP_DIR = tempfile.mkdtemp(prefix="cropdet-tests-")
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DIR}/test.db".replace("\\", "/")

from app.main import app  # noqa: E402  (after env override)


@pytest.fixture(scope="session")
def client() -> Iterator[TestClient]:
    """One app instance for the whole test session; lifespan loads the model."""
    with TestClient(app) as c:
        yield c


def make_png(width: int = 320, height: int = 320, color: tuple = (60, 130, 60)) -> bytes:
    """Build a synthetic leaf-looking PNG in memory (no file IO needed).

    Texture is added on purpose: the plant_guard sanity gate (correctly)
    refuses one-solid-color "photo of nothing" uploads, and the positive
    tests should resemble a real photo, which always carries detail.
    """
    import io
    import random

    import numpy as np
    from PIL import Image

    # Mild horizontal gradient + noise: enough detail for the flat-photo
    # guard to pass, and closer to a real photo than a flat fill.
    xs = np.arange(width, dtype=np.int16)
    shading = ((87 - 0.25 * xs) // 12)[None, :]
    arr = np.full((height, width, 3), color, dtype=np.int16)
    arr[..., 1] = np.clip(arr[..., 1] + shading, 0, 255)
    noise = np.random.default_rng(7).integers(-18, 19, size=arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
