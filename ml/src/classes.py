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
      disease together in a single multi-class head.

Adding a new crop (v2 expansion)
    The original 13 classes were rice / potato / tomato. Canon has since grown
    with 8 more crops sourced from free public datasets — the canonical class
    list below is the FULL target. A trained model only covers the subset its
    training data actually contained (train.py supports subsets and
    export_onnx.py writes labels.json from what was really trained), so this
    file may be ahead of the deployed model at any time. Eggplant remains a
    TODO: OLID I's class names must be confirmed before adding those classes.
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
    # --- chilli (মরিচ) — Bangladesh chilli dataset (Mendeley w9mr3vf56s) ---
    "chilli_bacterial_spot",
    "chilli_curl_virus",
    "chilli_cercospora_leaf_spot",
    "chilli_nutrient_deficiency",
    "chilli_white_spot",
    "chilli_healthy",
    # --- maize / corn (ভুট্টা) — PlantVillage + Tanzania maize (Zenodo) ---
    "maize_common_rust",
    "maize_gray_leaf_spot",
    "maize_northern_leaf_blight",
    "maize_lethal_necrosis",
    "maize_streak_virus",
    "maize_healthy",
    # --- grape (আঙুর) — PlantVillage + NGLD (Mendeley 8nnd2ypcv3) ---
    "grape_black_rot",
    "grape_esca",
    "grape_leaf_blight",
    "grape_downy_mildew",
    "grape_powdery_mildew",
    "grape_bacterial_leaf_spot",
    "grape_healthy",
    # --- cauliflower (ফুলকপি) — BD cauliflower dataset (Mendeley x995snz7p3) ---
    "cauliflower_black_rot",
    "cauliflower_insect_holes",
    "cauliflower_healthy",
    # --- cucumber (শসা) — Cucumber Disease Recognition (Mendeley y6d3z6f8z9) ---
    "cucumber_anthracnose",
    "cucumber_bacterial_wilt",
    "cucumber_belly_rot",
    "cucumber_downy_mildew",
    "cucumber_pythium_fruit_rot",
    "cucumber_gummy_stem_blight",
    "cucumber_healthy",
    # --- papaya (পেঁপে) — BDPapayaLeaf (Mendeley p997fvf526) ---
    "papaya_anthracnose",
    "papaya_bacterial_spot",
    "papaya_curl",
    "papaya_ring_spot",
    "papaya_healthy",
    # --- mango (আম) — MangoLeafBD (Mendeley hxsnvwty3r) ---
    "mango_anthracnose",
    "mango_bacterial_canker",
    "mango_cutting_weevil",
    "mango_die_back",
    "mango_gall_midge",
    "mango_powdery_mildew",
    "mango_sooty_mould",
    "mango_healthy",
    # --- okra (ঢেঁড়স) — Okra DiseaseNet (Mendeley nh7zk4hv8z) ---
    "okra_alternaria_leaf_spot",
    "okra_cercospora_leaf_spot",
    "okra_downy_mildew",
    "okra_leaf_curl_virus",
    "okra_phyllosticta_leaf_spot",
    "okra_healthy",
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
    # --- chilli ---
    "chilli_bacterial_spot": "Chilli Bacterial Spot",
    "chilli_curl_virus": "Chilli Curl Virus",
    "chilli_cercospora_leaf_spot": "Chilli Cercospora Leaf Spot",
    "chilli_nutrient_deficiency": "Chilli Nutrient Deficiency",
    "chilli_white_spot": "Chilli White Spot",
    "chilli_healthy": "Chilli - Healthy",
    # --- maize ---
    "maize_common_rust": "Maize Common Rust",
    "maize_gray_leaf_spot": "Maize Gray Leaf Spot",
    "maize_northern_leaf_blight": "Maize Northern Leaf Blight",
    "maize_lethal_necrosis": "Maize Lethal Necrosis",
    "maize_streak_virus": "Maize Streak Virus",
    "maize_healthy": "Maize - Healthy",
    # --- grape ---
    "grape_black_rot": "Grape Black Rot",
    "grape_esca": "Grape Esca",
    "grape_leaf_blight": "Grape Leaf Blight",
    "grape_downy_mildew": "Grape Downy Mildew",
    "grape_powdery_mildew": "Grape Powdery Mildew",
    "grape_bacterial_leaf_spot": "Grape Bacterial Leaf Spot",
    "grape_healthy": "Grape - Healthy",
    # --- cauliflower ---
    "cauliflower_black_rot": "Cauliflower Black Rot",
    "cauliflower_insect_holes": "Cauliflower Insect Holes",
    "cauliflower_healthy": "Cauliflower - Healthy",
    # --- cucumber ---
    "cucumber_anthracnose": "Cucumber Anthracnose",
    "cucumber_bacterial_wilt": "Cucumber Bacterial Wilt",
    "cucumber_belly_rot": "Cucumber Belly Rot",
    "cucumber_downy_mildew": "Cucumber Downy Mildew",
    "cucumber_pythium_fruit_rot": "Cucumber Pythium Fruit Rot",
    "cucumber_gummy_stem_blight": "Cucumber Gummy Stem Blight",
    "cucumber_healthy": "Cucumber - Healthy",
    # --- papaya ---
    "papaya_anthracnose": "Papaya Anthracnose",
    "papaya_bacterial_spot": "Papaya Bacterial Spot",
    "papaya_curl": "Papaya Curl",
    "papaya_ring_spot": "Papaya Ring Spot",
    "papaya_healthy": "Papaya - Healthy",
    # --- mango ---
    "mango_anthracnose": "Mango Anthracnose",
    "mango_bacterial_canker": "Mango Bacterial Canker",
    "mango_cutting_weevil": "Mango Cutting Weevil",
    "mango_die_back": "Mango Die Back",
    "mango_gall_midge": "Mango Gall Midge",
    "mango_powdery_mildew": "Mango Powdery Mildew",
    "mango_sooty_mould": "Mango Sooty Mould",
    "mango_healthy": "Mango - Healthy",
    # --- okra ---
    "okra_alternaria_leaf_spot": "Okra Alternaria Leaf Spot",
    "okra_cercospora_leaf_spot": "Okra Cercospora Leaf Spot",
    "okra_downy_mildew": "Okra Downy Mildew",
    "okra_leaf_curl_virus": "Okra Leaf Curl Virus",
    "okra_phyllosticta_leaf_spot": "Okra Phyllosticta Leaf Spot",
    "okra_healthy": "Okra - Healthy",
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
    # --- chilli (মরিচ) ---
    "chilli_bacterial_spot": "মরিচ — ব্যাকটেরিয়াল স্পট",
    "chilli_curl_virus": "মরিচ — পাতা কোঁকড়ানো ভাইরাস",
    "chilli_cercospora_leaf_spot": "মরিচ — সারকোস্পোরা পাতা দাগ",
    "chilli_nutrient_deficiency": "মরিচ — পুষ্টি উপাদানের ঘাটতি",
    "chilli_white_spot": "মরিচ — সাদা দাগ",
    "chilli_healthy": "মরিচ — সুস্থ পাতা",
    # --- maize (ভুট্টা) ---
    "maize_common_rust": "ভুট্টা — সাধারণ মরিচা (Common Rust)",
    "maize_gray_leaf_spot": "ভুট্টা — ধূসর পাতা দাগ (Gray Leaf Spot)",
    "maize_northern_leaf_blight": "ভুট্টা — উত্তর পাতা পোড়া (Northern Leaf Blight)",
    "maize_lethal_necrosis": "ভুট্টা — লিথাল নেক্রোসিস (মারাত্মক পাতা মৃত্যু)",
    "maize_streak_virus": "ভুট্টা — স্ট্রিক ভাইরাস",
    "maize_healthy": "ভুট্টা — সুস্থ পাতা",
    # --- grape (আঙুর) ---
    "grape_black_rot": "আঙুর — কালো পচা (Black Rot)",
    "grape_esca": "আঙুর — এসকা (Esca)",
    "grape_leaf_blight": "আঙুর — পাতা পোড়া (Leaf Blight)",
    "grape_downy_mildew": "আঙুর — ডাউনি মিলডিউ",
    "grape_powdery_mildew": "আঙুর — পাউডারি মিলডিউ",
    "grape_bacterial_leaf_spot": "আঙুর — ব্যাকটেরিয়াল পাতা দাগ",
    "grape_healthy": "আঙুর — সুস্থ পাতা",
    # --- cauliflower (ফুলকপি) ---
    "cauliflower_black_rot": "ফুলকপি — কালো পচা (Black Rot)",
    "cauliflower_insect_holes": "ফুলকপি — পোকার ছিদ্র",
    "cauliflower_healthy": "ফুলকপি — সুস্থ পাতা",
    # --- cucumber (শসা) ---
    "cucumber_anthracnose": "শসা — অ্যানথ্রাকনোজ",
    "cucumber_bacterial_wilt": "শসা — ব্যাকটেরিয়াল উইল্ট",
    "cucumber_belly_rot": "শসা — বেলি রট (ফল পচা)",
    "cucumber_downy_mildew": "শসা — ডাউনি মিলডিউ",
    "cucumber_pythium_fruit_rot": "শসা — পাইথিয়াম ফল পচা",
    "cucumber_gummy_stem_blight": "শসা — গামি স্টেম ব্লাইট",
    "cucumber_healthy": "শসা — সুস্থ পাতা",
    # --- papaya (পেঁপে) ---
    "papaya_anthracnose": "পেঁপে — অ্যানথ্রাকনোজ",
    "papaya_bacterial_spot": "পেঁপে — ব্যাকটেরিয়াল স্পট",
    "papaya_curl": "পেঁপে — পাতা কোঁকড়ানো",
    "papaya_ring_spot": "পেঁপে — রিং স্পট ভাইরাস",
    "papaya_healthy": "পেঁপে — সুস্থ পাতা",
    # --- mango (আম) ---
    "mango_anthracnose": "আম — অ্যানথ্রাকনোজ",
    "mango_bacterial_canker": "আম — ব্যাকটেরিয়াল ক্যানসার",
    "mango_cutting_weevil": "আম — কাটিং উইভিল পোকা",
    "mango_die_back": "আম — ডাই-ব্যাক (ডগা শুকিয়ে যাওয়া)",
    "mango_gall_midge": "আম — গল মিজ (পাতা ফোলা)",
    "mango_powdery_mildew": "আম — পাউডারি মিলডিউ",
    "mango_sooty_mould": "আম — সুটি ছাঁচ (কালো আস্তরণ)",
    "mango_healthy": "আম — সুস্থ পাতা",
    # --- okra (ঢেঁড়স) ---
    "okra_alternaria_leaf_spot": "ঢেঁড়স — অলটারনেরিয়া পাতা দাগ",
    "okra_cercospora_leaf_spot": "ঢেঁড়স — সারকোস্পোরা পাতা দাগ",
    "okra_downy_mildew": "ঢেঁড়স — ডাউনি মিলডিউ",
    "okra_leaf_curl_virus": "ঢেঁড়স — পাতা কোঁকড়ানো ভাইরাস",
    "okra_phyllosticta_leaf_spot": "ঢেঁড়স — ফাইলোস্টিক্টা পাতা দাগ",
    "okra_healthy": "ঢেঁড়স — সুস্থ পাতা",
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
    assert set(crop_of(c) for c in CLASSES) == {
        "rice", "potato", "tomato", "chilli", "maize", "grape",
        "cauliflower", "cucumber", "papaya", "mango", "okra",
    }
    print(
        f"classes.py OK — {N_CLASSES} classes across "
        f"{len(set(crop_of(c) for c in CLASSES))} crops: "
        + ", ".join(sorted(set(crop_of(c) for c in CLASSES)))
    )


if __name__ == "__main__":
    validate()
