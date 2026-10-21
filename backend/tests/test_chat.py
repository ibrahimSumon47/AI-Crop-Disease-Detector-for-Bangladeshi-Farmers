"""Tests for the /chat endpoint (key-free fallback path, no LLM key set)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path
from tests.conftest import client  # noqa: E402, F401


def test_chat_app_usage_question(client):
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "অ্যাপটা কীভাবে ব্যবহার করবো?"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["source"] == "fallback"
    assert "ছবি" in body["reply"]


def test_chat_disease_keyword_gives_guideline(client):
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "ধানের ব্লাস্ট কীভাবে সারাবো?"}]})
    assert resp.status_code == 200
    body = resp.json()
    assert "ব্লাস্ট" in body["reply"]
    assert "চিকিৎসা" in body["reply"] or "প্রতিরোধ" in body["reply"]


def test_chat_unknown_question_is_honest(client):
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "আজ কি বৃষ্টি হবে?"}]})
    body = resp.json()
    assert "মেলেনি" in body["reply"]


def test_chat_empty_messages_rejected(client):
    resp = client.post("/chat", json={"messages": []})
    assert resp.status_code == 422  # schema requires at least one message
