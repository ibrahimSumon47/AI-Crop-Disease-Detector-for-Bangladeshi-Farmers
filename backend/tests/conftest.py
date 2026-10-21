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
    """Build a synthetic leaf-looking PNG in memory (no file IO needed)."""
    import io

    from PIL import Image
    img = Image.new("RGB", (width, height), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
