"""API integration tests: /health, /predict (real model), /history, validation."""

import sys
from pathlib import Path

import json

import pytest

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path
from tests.conftest import dataset_leaf, make_png  # noqa: E402


def _known_crops() -> set[str]:
    """Every crop the CURRENT model can emit, from backend/app/data/labels.json.

    Read at test time so this stays correct as the class list grows
    (it was written for a 3-crop model and failed the moment the
    61-class model legitimately predicted 'maize').
    """
    labels_path = Path(__file__).resolve().parents[1] / "app" / "data" / "labels.json"
    payload = json.loads(labels_path.read_text(encoding="utf-8"))
    return {entry["crop"] for entry in payload["labels"].values()}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True  # lifespan loaded the real model


def test_predict_returns_full_bangla_response(client):
    """End-to-end: real leaf (or synthetic fallback) -> real ONNX inference
    -> structured JSON. The trained gate refuses featureless synthetic
    junk, so a dataset leaf is used when available; skipped when neither
    is usable."""
    leaf = dataset_leaf()
    if leaf:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.open(leaf).convert("RGB").save(buf, format="JPEG", quality=92)
        files = {"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")}
    else:
        files = {"file": ("leaf.png", make_png(), "image/png")}
    resp = client.post("/predict", files=files)
    if resp.status_code == 422 and not leaf:
        pytest.skip("no clean leaf available for the diagnose path")
    assert resp.status_code == 200
    body = resp.json()

    # Contract the frontend codes against:
    for key in ("crop", "disease", "disease_bn", "confidence", "top3", "is_uncertain", "advice"):
        assert key in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top3"]) == 3
    assert body["top3"][0]["rank"] == 1
    assert body["crop"] in _known_crops()
    assert isinstance(body["is_uncertain"], bool)
    assert body["scan_id"] is not None


def test_predict_rejects_bad_file(client):
    resp = client.post(
        "/predict",
        files={"file": ("evil.txt", b"definitely not an image", "text/plain")},
    )
    assert resp.status_code == 422


def test_predict_rejects_empty_file(client):
    resp = client.post("/predict", files={"file": ("empty.png", b"", "image/png")})
    assert resp.status_code == 422


def test_predict_rejects_flat_color_image(client):
    """One-solid-color 'photo of nothing' must not get any diagnosis."""
    import io

    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", (320, 320), (60, 130, 60)).save(buf, format="PNG")
    resp = client.post(
        "/predict",
        files={"file": ("flat.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 422
    assert "ফাঁকা" in resp.json()["detail"]


def test_predict_rejects_non_leaf_object_on_plain_background(client):
    """A tuber/product-style object on a plain white background must be
    rejected with leaf guidance — not a misleading 2% diagnosis.
    """
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (400, 400), (250, 250, 250))
    draw = ImageDraw.Draw(img)
    # A tan/brown blob (like the sprouted potato) + a few green shoots.
    draw.ellipse((130, 130, 280, 280), fill=(176, 138, 90))
    draw.ellipse((150, 150, 260, 260), fill=(160, 120, 80))
    for x, y in ((160, 140), (250, 150), (200, 270)):
        draw.ellipse((x, y, x + 14, y + 20), fill=(90, 120, 60))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    resp = client.post(
        "/predict",
        files={"file": ("tuber.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 422
    assert "পাতা" in resp.json()["detail"]


def _degrade(path: Path, brightness: float, blur: float) -> bytes:
    """Simulate a bad phone shot of that same leaf: blurred + darkened."""
    import io

    from PIL import Image, ImageEnhance, ImageFilter

    img = Image.open(path).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(blur))
    img = ImageEnhance.Brightness(img).enhance(brightness)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    return buf.getvalue()


def test_predict_warns_on_dark_blurry_photo(client):
    """A real leaf photo degraded to dark+blur must STILL be diagnosed
    (warnings, not refusal) and must carry Bangla photo-quality notes."""
    leaf = dataset_leaf(require_clean=False)
    if leaf is None:
        pytest.skip("data/processed/test not present")
    resp = client.post(
        "/predict",
        files={"file": ("bad.jpg", _degrade(leaf, 0.35, 6.0), "image/jpeg")},
    )
    assert resp.status_code == 200, resp.json()
    body = resp.json()
    assert len(body["photo_warnings"]) >= 1
    assert any("ঝাপসা" in w for w in body["photo_warnings"]), body["photo_warnings"]


def test_predict_no_warnings_on_crisp_leaf(client):
    """An undegraded real leaf must produce NO quality warnings."""
    leaf = dataset_leaf(require_clean=True)
    if leaf is None:
        pytest.skip("data/processed/test not present (or no clean leaf found)")
    import io

    buf = io.BytesIO()
    from PIL import Image

    Image.open(leaf).convert("RGB").save(buf, format="JPEG", quality=92)
    resp = client.post("/predict", files={"file": ("crisp.jpg", buf.getvalue(), "image/jpeg")})
    assert resp.status_code == 200, resp.json()
    assert resp.json()["photo_warnings"] == []



def test_history_lists_saved_scans(client):
    """A successful predict must be visible in history afterwards."""
    leaf = dataset_leaf()
    if leaf:
        import io

        from PIL import Image

        buf = io.BytesIO()
        Image.open(leaf).convert("RGB").save(buf, format="JPEG", quality=92)
        files = {"file": ("leaf.jpg", buf.getvalue(), "image/jpeg")}
    else:
        files = {"file": ("leaf.png", make_png(), "image/png")}
    resp_post = client.post("/predict", files=files)
    if resp_post.status_code == 422 and not leaf:
        pytest.skip("no clean leaf available for the diagnose path")
    resp = client.get("/history?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert len(body["items"]) >= 1
    item = body["items"][0]
    assert item["crop"] in _known_crops()
    assert isinstance(item["is_uncertain"], bool)
