"""API integration tests: /health, /predict (real model), /history, validation."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path
from tests.conftest import make_png  # noqa: E402


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True  # lifespan loaded the real model


def test_predict_returns_full_bangla_response(client):
    """End-to-end: synthetic PNG -> real ONNX inference -> structured JSON."""
    resp = client.post(
        "/predict",
        files={"file": ("leaf.png", make_png(), "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()

    # Contract the frontend codes against:
    for key in ("crop", "disease", "disease_bn", "confidence", "top3", "is_uncertain", "advice"):
        assert key in body
    assert 0.0 <= body["confidence"] <= 1.0
    assert len(body["top3"]) == 3
    assert body["top3"][0]["rank"] == 1
    assert body["crop"] in {"rice", "potato", "tomato"}
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


def test_history_lists_saved_scans(client):
    """A successful predict must be visible in history afterwards."""
    client.post("/predict", files={"file": ("leaf.png", make_png(), "image/png")})
    resp = client.get("/history?limit=5")
    assert resp.status_code == 200
    body = resp.json()
    assert body["total"] >= 1
    assert len(body["items"]) >= 1
    item = body["items"][0]
    assert item["crop"] in {"rice", "potato", "tomato"}
    assert isinstance(item["is_uncertain"], bool)
