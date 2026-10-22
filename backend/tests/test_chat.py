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


def test_chat_greeting_is_bangla_salam_not_namaskar(client):
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "আসসালামু আলাইকুম"}]})
    reply = resp.json()["reply"]
    assert "আলাইকুমুস সালাম" in reply
    assert "নমস্কার" not in reply


def test_chat_lists_all_conditions_for_a_crop(client):
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "আলুর রোগ কি কি?"}]})
    reply = resp.json()["reply"]
    assert "আলু" in reply
    # all three potato classes are offered, not just one
    assert reply.count("•") >= 3


def test_chat_answers_a_disease_beyond_the_old_keyword_list(client):
    """Cucumber downy mildew never matched the old 7-class keyword router."""
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "শসার ডাউনি মিলডিউ কী?"}]})
    reply = resp.json()["reply"]
    assert "ডাউনি মিলডিউ" in reply
    assert "লক্ষণ" in reply


def test_chat_symptom_only_question_stays_on_topic(client):
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "ভুট্টার মরিচায় লক্ষণ কী?"}]})
    reply = resp.json()["reply"]
    assert "লক্ষণ" in reply
    assert "চিকিৎসা" not in reply  # asked for symptoms only


def test_chat_same_disease_on_two_hosts_asks_which_crop(client):
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "লেট ব্লাইট"}]})
    reply = resp.json()["reply"]
    assert "আলু" in reply and "টমেটো" in reply


def test_chat_does_not_treat_ami_as_mango(client):
    """"আমি" contains "আম" (mango) — a self-reference must not become a crop."""
    resp = client.post("/chat", json={"messages": [{"role": "user", "content": "আমি কী করব বুঝতে পারছি না"}]})
    reply = resp.json()["reply"]
    assert "আমের" not in reply
