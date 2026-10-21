"""Tests for processing.attach_family_note (potato<->tomato blight stabilizer)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # backend/ on path

from app.services.processing import RICE_NOTE, attach_family_note, is_crop_unclear  # noqa: E402


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


def test_rice_plant_photo_gets_closeup_note():
    # Whole-plant rice photo: rice_blast 52% — right family, unsure disease.
    top3 = [
        _p("rice_blast", 0.522),
        _p("rice_brown_spot", 0.371),
        _p("rice_healthy", 0.107),
    ]
    note = attach_family_note(top3)
    assert note is RICE_NOTE
    assert "পাতার কাছ থেকে" in note or "কাছ থেকে" in note


def test_crop_unclear_when_top1_below_25_percent():
    # User-observed case: rice plant photo landed on potato_healthy 19%.
    top3 = [
        _p("potato_healthy", 0.19),
        _p("tomato_healthy", 0.09),
        _p("potato_late_blight", 0.08),
    ]
    assert is_crop_unclear(top3) is True


def test_crop_clear_when_top1_25_percent_or_more():
    top3 = [_p("potato_healthy", 0.31), _p("rice_healthy", 0.20), _p("tomato_healthy", 0.10)]
    assert is_crop_unclear(top3) is False


def test_rice_high_confidence_no_note():
    # A crisply identified rice disease at lab-level confidence: no note.
    top3 = [
        _p("rice_blast", 0.80),
        _p("rice_brown_spot", 0.10),
        _p("rice_healthy", 0.05),
    ]
    assert attach_family_note(top3) is None
