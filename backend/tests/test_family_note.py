"""Tests for processing.attach_family_note (potato<->tomato blight stabilizer)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path

from app.services.processing import attach_family_note  # noqa: E402


def _p(cls, c):
    return {"class_name": cls, "confidence": c}


def test_note_fires_when_torn_between_potato_and_tomato_blight():
    top3 = [
        _p("potato_early_blight", 0.495),
        _p("potato_late_blight", 0.273),
        _p("tomato_early_blight", 0.10),
    ]
    note = attach_family_note(top3)
    assert note is not None
    # Note must be Bangla, mention both crops and not promise a specific crop.
    assert "আলু" in note and "টমেটো" in note


def test_note_not_fired_when_family_prob_too_low():
    top3 = [
        _p("potato_early_blight", 0.10),
        _p("potato_late_blight", 0.05),
        _p("tomato_early_blight", 0.02),
    ]
    assert attach_family_note(top3) is None


def test_note_not_fired_when_relative_absent():
    # tomato_early_blight NOT in top3 -> no torn decision, no note
    top3 = [
        _p("potato_early_blight", 0.80),
        _p("potato_late_blight", 0.10),
        _p("potato_healthy", 0.05),
    ]
    assert attach_family_note(top3) is None


def test_note_not_fired_for_healthy_or_rice():
    top3 = [
        _p("tomato_healthy", 0.60),
        _p("potato_healthy", 0.30),  # healthy has no cross-crop family rule
        _p("rice_blast", 0.05),
    ]
    assert attach_family_note(top3) is None


def test_note_fires_from_tomato_top1():
    top3 = [
        _p("tomato_late_blight", 0.45),
        _p("potato_late_blight", 0.40),
        _p("tomato_healthy", 0.10),
    ]
    note = attach_family_note(top3)
    assert note is not None
    assert "অনাগাম" in note
