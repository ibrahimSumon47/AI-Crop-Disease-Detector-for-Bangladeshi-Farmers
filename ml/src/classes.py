"""
classes.py — the single source of truth for the model's class list.

Why one canonical file?
    Every other component (data prep script, training script, backend
    labels.json, backend inference, frontend display) must agree on the
    class ORDER. A mismatch silently shuffles predictions — e.g. the model
    says "tomato late blight" but the backend reads it as "potato healthy".
    So there is exactly one place that defines the order, and
    `export_onnx.py` writes `labels.json` straight from this file.

Class naming rules
    - Lowercase snake_case, prefixed with the crop: `<crop>_<disease>`.
    - Every crop has its own `healthy` class, so the model learns crop +
      disease together in a single multi-class head (13 classes total).
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# CLASSES — the final, ordered list. Index in this list == model output index.
#
# Selected to match what the three public datasets actually contain:
#   * PlantVillage provides potato (early/late blight + healthy) and tomato
#     (many diseases; we take a practical subset).
#   * A public rice leaf disease dataset (e.g. Rice Leaf Diseases from
#     UCI/Mendeley) provides brown_spot, blast, bacterial_leaf_blight, tungro.
# If a chosen dataset turns out to lack a class, this list must be revised
# TOGETHER WITH the user before training (see T2.2 notes).
# ---------------------------------------------------------------------------
CLASSES: list[str] = [
    # --- rice (ধান) ---
    "rice_bacterial_leaf_blight",
    "rice_brown_spot",
    "rice_blast",
    "rice_tungro",
    "rice_healthy",
    # --- potato (আলু) ---
    "potato_early_blight",
    "potato_late_blight",
    "potato_healthy",
    # --- tomato (টম্যাটো) ---
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_healthy",
]

# ---------------------------------------------------------------------------
# Derived mappings (always derived from CLASSES — never edited by hand).
# ---------------------------------------------------------------------------
N_CLASSES: int = len(CLASSES)

# Model output index -> class label
INDEX_TO_LABEL: dict[int, str] = {i: name for i, name in enumerate(CLASSES)}

# Class label -> model output index
LABEL_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(CLASSES)}


def crop_of(label: str) -> str:
    """Split '<crop>_<disease>' into the crop part (before the first '_')."""
    return label.split("_", 1)[0]


def disease_of(label: str) -> str:
    """Split '<crop>_<disease>' into the disease part (else 'healthy')."""
    return label.split("_", 1)[1]


# ---------------------------------------------------------------------------
# Human-readable names — used by the backend labels.json and the guideline
# lookup so the UI can show "পাতার ব্লাস্ট" instead of "rice_blast".
# ---------------------------------------------------------------------------
EN_NAMES: dict[str, str] = {
    "rice_bacterial_leaf_blight": "Rice Bacterial Leaf Blight",
    "rice_brown_spot": "Rice Brown Spot",
    "rice_blast": "Rice Blast",
    "rice_tungro": "Rice Tungro",
    "rice_healthy": "Rice - Healthy",
    "potato_early_blight": "Potato Early Blight",
    "potato_late_blight": "Potato Late Blight",
    "potato_healthy": "Potato - Healthy",
    "tomato_bacterial_spot": "Tomato Bacterial Spot",
    "tomato_early_blight": "Tomato Early Blight",
    "tomato_late_blight": "Tomato Late Blight",
    "tomato_leaf_mold": "Tomato Leaf Mold",
    "tomato_healthy": "Tomato - Healthy",
}

# Bangla names — served in the prediction response for the Bangla-first UI.
BN_NAMES: dict[str, str] = {
    "rice_bacterial_leaf_blight": "ধান — ব্যাকটেরিয়াল লিফ ব্লাইট",
    "rice_brown_spot": "ধান — বাদামি দাগ",
    "rice_blast": "ধান — ব্লাস্ট রোগ",
    "rice_tungro": "ধান — টুংরো রোগ",
    "rice_healthy": "ধান — সুস্থ পাতা",
    "potato_early_blight": "আলু — আগাম পাতা পোড়া (Early Blight)",
    "potato_late_blight": "আলু — অনাগাম পাতা পোড়া (Late Blight)",
    "potato_healthy": "আলু — সুস্থ পাতা",
    "tomato_bacterial_spot": "টম্যাটো — ব্যাকটেরিয়াল স্পট",
    "tomato_early_blight": "টম্যাটো — আগাম পাতা পোড়া (Early Blight)",
    "tomato_late_blight": "টম্যাটো — অনাগাম পাতা পোড়া (Late Blight)",
    "tomato_leaf_mold": "টম্যাটো — লিফ মোল্ড",
    "tomato_healthy": "টম্যাটো — সুস্থ পাতা",
}


def missing_from_dataset(folders: list[str]) -> list[str]:
    """Validate a dataset's class folders against the canonical class list.

    IMPORTANT — what the model's label order actually is:
        Keras (`image_dataset_from_directory`) builds its label index from the
        ALPHABETICAL order of the folder names, so the trained model's output
        order is the DATASET order, not this file's order (this file groups by
        crop: rice, potato, tomato). train.py therefore records the real order
        in ml/reports/class_order.json and export_onnx.py replays that exact
        order when it writes labels.json. This function only guards the SET of
        names, which is what actually protects us from silent label swaps.

    Args:
        folders: class folder names found in data/processed/train.

    Returns:
        Canonical classes the dataset does NOT contain. We train on what the
        data actually has (subset support) and report the rest, instead of
        crashing — e.g. the public rice dataset we downloaded has blast,
        brown spot and healthy, but no bacterial leaf blight / tungro images.

    Raises:
        SystemExit: when a folder is not a known class. That means the data
        preparation step produced a typo'd folder name, which would otherwise
        turn into an unknown 12th output index.
    """
    unknown = sorted({f for f in folders if f not in LABEL_TO_INDEX})
    if unknown:
        raise SystemExit(
            "ERROR: unknown dataset class folder(s): "
            f"{unknown}\nKnown classes: {sorted(CLASSES)}\n"
            "Fix prepare_data.py aliases or rename the folders."
        )
    present = set(folders)
    return [c for c in CLASSES if c not in present]


def validate() -> None:
    """Self-check called by other scripts: all mappings must be consistent."""
    assert len(set(CLASSES)) == N_CLASSES, "duplicate class names!"
    assert INDEX_TO_LABEL[N_CLASSES - 1] == CLASSES[-1]
    assert sorted(INDEX_TO_LABEL) == list(range(N_CLASSES))
    assert set(EN_NAMES) == set(CLASSES), "EN_NAMES missing/extra entries"
    assert set(BN_NAMES) == set(CLASSES), "BN_NAMES missing/extra entries"
    assert set(crop_of(c) for c in CLASSES) == {"rice", "potato", "tomato"}
    print(f"classes.py OK — {N_CLASSES} classes, crops: rice / potato / tomato")


if __name__ == "__main__":
    validate()
