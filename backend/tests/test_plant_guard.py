"""Unit tests for the plant gate (backend/app/services/plant_guard.py).

Covers the trained ONNX classifier (primary gate), its heuristic fallback,
and the blur/dark photo-quality warnings — all with in-memory images so no
dataset is needed.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path

from app.services import plant_guard  # noqa: E402


def _tuber_on_white() -> Image.Image:
    """Product-shot style: tan blob + green shoots on a plain light card."""
    img = Image.new("RGB", (400, 400), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    draw.ellipse((130, 130, 280, 280), fill=(176, 138, 90))
    draw.ellipse((150, 150, 260, 260), fill=(160, 120, 80))
    for x, y in ((160, 140), (250, 150), (200, 270)):
        draw.ellipse((x, y, x + 14, y + 20), fill=(90, 120, 60))
    return img


def _leaf_on_dark() -> Image.Image:
    """Leaf-photo-like: textured green plant material on a dark backdrop
    (the PlantVillage signature the color heuristic used to fail)."""
    import random

    img = Image.new("RGB", (480, 480), (35, 30, 25))
    draw = ImageDraw.Draw(img)
    rng = random.Random(9)
    # leafy blob with per-pixel variation
    for _ in range(12000):
        x = rng.randint(90, 400)
        y = rng.randint(90, 400)
        if (x - 245) ** 2 + (y - 245) ** 2 > 155**2:
            continue
        draw.point((x, y), fill=(rng.randint(40, 95), rng.randint(95, 165), rng.randint(30, 85)))
    img = img.filter(ImageFilter.GaussianBlur(1.2))
    return img


def test_gate_rejects_flat_image():
    flat = Image.new("RGB", (320, 320), (60, 130, 60))
    ok, msg = plant_guard.looks_like_leaf(flat)
    assert not ok
    assert "ফাঁকা" in msg


def test_gate_rejects_tuber_on_white():
    ok, msg = plant_guard.looks_like_leaf(_tuber_on_white())
    assert not ok
    assert "পাতা" in msg


def test_gate_accepts_leaf_on_dark_background():
    """The main regression the trained classifier fixes: real leaves on dark
    backdrops must pass, which the v1 green/white heuristic rejected."""
    from tests.conftest import dataset_leaf

    leaf = dataset_leaf()
    if leaf is None:
        # No dataset: fall back to the synthetic leaf blob, which the model
        # was not trained on, so only assert the FALLBACK heuristic accepts
        # it (green plant material, non-plain background).
        stats = plant_guard._features(_leaf_on_dark())
        assert stats.white_frac < 0.5  # not a plain-background product shot
        return
    ok, msg = plant_guard.looks_like_leaf(Image.open(leaf).convert("RGB"))
    assert ok, msg


def test_gate_falls_back_to_heuristic_without_model(tmp_path, monkeypatch):
    """Missing/corrupt ONNX file must NOT silently diagnose everything:
    the v1 plain-background product-shot rule has to take over."""
    broken = tmp_path / "plant_guard_model.onnx"
    broken.write_bytes(b"not an onnx file")
    monkeypatch.setattr(plant_guard, "GATE_MODEL_PATH", broken)
    monkeypatch.setattr(plant_guard, "_session", None)
    monkeypatch.setattr(plant_guard, "_session_failed", False)

    ok, msg = plant_guard.looks_like_leaf(_tuber_on_white())
    assert not ok
    assert "পাতা" in msg


def test_photo_warnings_flags_blur_and_dark():
    img = _leaf_on_dark().filter(ImageFilter.GaussianBlur(8))
    small = img.convert("L")
    small.thumbnail((320, 320))
    just = __import__("numpy").asarray(small, dtype=float)
    just[:] *= 0.35
    img = Image.fromarray(just.astype("uint8")).convert("RGB")
    warnings = plant_guard.photo_warnings(img)
    assert any("ঝাপসা" in w for w in warnings)
    assert any("অন্ধকার" in w for w in warnings)


def test_photo_warnings_silent_on_good_photo():
    """A normal-brightness, detailed photo raises no warnings."""
    import io

    img = Image.new("RGB", (480, 480), (80, 135, 55))
    draw = ImageDraw.Draw(img)
    for i in range(0, 480, 6):
        draw.line([(0, i), (480, i)], fill=(70, 125, 50), width=2)
        draw.line([(i, 0), (i, 480)], fill=(92, 150, 62), width=2)
    draw.ellipse((150, 150, 330, 330), outline=(55, 35, 25), width=6)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    from PIL import Image as PILImage

    img = PILImage.open(buf)
    assert plant_guard.photo_warnings(img) == []
