"""
prepare_data.py — merge the three raw datasets into a stratified split.

WHAT IT DOES
    Takes images from the raw dataset folders described in
    `ml/src/download_data.md` and writes a merged, stratified split:

        data/
          processed/
            train/<class_name>/...
            val/<class_name>/...
            test/<class_name>/...

    - 70 % train / 15 % val / 15 % test, split stratified per class.
    - Fixed RANDOM_SEED so the split is reproducible across runs/machines.
    - Files are COPIED (not moved), so the raw sources stay reusable.
    - Dataset layouts can differ (class-nested subfolders); the script
      scans recursively and maps folder names to canonical classes via
      ALIAS_MAP below, with a separator-insensitive fallback pass
      ("Tomato___Early_blight", "tomato-early-blight" and
      "Tomato Early blight" all resolve to the same class).

USAGE
    python ml/src/prepare_data.py [--raw-dir ml/data/raw] [--out-dir data/processed]
    Requires: Pillow, numpy, scikit-learn. No GPU needed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from classes import CLASSES

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Repo root = parent of the `ml/` folder (this file lives in ml/src/).
REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_RAW_DIR = REPO_ROOT / "ml" / "data" / "raw"
DEFAULT_OUT_DIR = REPO_ROOT / "data" / "processed"

# Fixed seed: changing this invalidates previous splits/results, so it is
# intentionally a constant, not a CLI option.
RANDOM_SEED = 42

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15

VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}

# Classes smaller than MIN_CLASS_COUNT cannot produce a meaningful
# 70/15/15 split (val/test would be empty); they are skipped with a loud
# warning instead of crashing mid-run.
MIN_CLASS_COUNT = 4

# ---------------------------------------------------------------------------
# Folder-name aliases — every raw dataset names classes differently.
# Matching is done in TWO passes:
#   pass 1: normalized exact match  (lowercase, separators -> "_")
#   pass 2: separator-free match    ("LeafBlast" == "leaf_blast")
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, list[str]] = {
    # --- rice ---
    "rice_bacterial_leaf_blight": [
        "rice_bacterial_leaf_blight", "bacterial_leaf_blight",
        "bacterial leaf blight", "bacterialblight",
        "bacterial_leaf_blight_disease", "blb",
    ],
    "rice_brown_spot": ["rice_brown_spot", "brown_spot", "brown spot", "brownspot"],
    "rice_blast": ["blast", "leaf_blast", "leaf blast", "leafblast", "rice_blast"],
    "rice_tungro": ["tungro", "rice_tungro"],
    "rice_healthy": ["healthy", "healthy_leaf", "rice_healthy", "healty"],
    # --- potato (PlantVillage) ---
    "potato_early_blight": ["potato___early_blight", "potato___earlyblight", "potato_early_blight"],
    "potato_late_blight": ["potato___late_blight", "potato___lateblight", "potato_late_blight"],
    "potato_healthy": ["potato___healthy", "potato_healthy"],
    # --- tomato (PlantVillage) ---
    "tomato_bacterial_spot": ["tomato___bacterial_spot", "tomato_bacterial_spot"],
    "tomato_early_blight": ["tomato___early_blight", "tomato_early_blight"],
    "tomato_late_blight": ["tomato___late_blight", "tomato_late_blight"],
    "tomato_leaf_mold": ["tomato___leaf_mold", "tomato_leaf_mold"],
    "tomato_healthy": ["tomato___healthy", "tomato_healthy"],
    # --- chilli (Bangladesh chilli dataset, Mendeley w9mr3vf526s/ field photos) ---
    "chilli_bacterial_spot": ["chilli_bacterial_spot", "bacterial spot", "chilli_bacterialspot"],
    "chilli_curl_virus": ["chilli_curl_virus", "curl virus", "curl_virus", "leaf curl"],
    "chilli_cercospora_leaf_spot": ["chilli_cercospora", "chilli_cercospora_leaf_spot", "cercospora", "cercospora leaf spot"],
    "chilli_nutrient_deficiency": ["chilli_nutrient_deficiency", "nutrition deficiency", "nutrient deficiency"],
    "chilli_white_spot": ["chilli_white_spot", "white spot", "whitespot"],
    "chilli_healthy": ["chilli_healthy", "healthy_chilli", "chilli healthy"],
    # --- maize / corn (PlantVillage + Tanzania maize set, Zenodo 15685236) ---
    "maize_common_rust": ["maize___common_rust", "corn___common_rust", "corn_common_rust", "common rust"],
    "maize_gray_leaf_spot": ["maize___gray_leaf_spot", "corn___gray_leaf_spot", "gray leaf spot", "cercospora_zeae_maydis"],
    "maize_northern_leaf_blight": ["maize___northern_leaf_blight", "corn___northern_leaf_blight", "northern leaf blight"],
    "maize_lethal_necrosis": ["maize_lethal_necrosis", "maize lethal necrosis", "mln"],
    "maize_streak_virus": ["maize_streak_virus", "maize streak", "msv"],
    "maize_healthy": ["maize___healthy", "corn___healthy", "maize_healthy", "corn_healthy"],
    # --- grape (PlantVillage + NGLD set, Mendeley 8nnd2ypcv3) ---
    "grape_black_rot": ["grape___black_rot", "grape_black_rot", "black rot"],
    "grape_esca": ["grape___esca_black_measles", "grape_esca", "esca", "black measles"],
    "grape_leaf_blight": ["grape___leaf_blight_isariopsis_leaf_spot", "grape_leaf_blight", "isariopsis leaf spot"],
    "grape_downy_mildew": ["grape_downy_mildew", "downy mildew grape", "plasmopara viticola"],
    "grape_powdery_mildew": ["grape_powdery_mildew", "powdery mildew grape", "uncinula necator"],
    "grape_bacterial_leaf_spot": ["grape_bacterial_leaf_spot", "bacterial leaf spot grape"],
    "grape_healthy": ["grape___healthy", "grape_healthy"],
    # --- cauliflower (BD Cauliflower Leaf Disease, Mendeley x995snz7p3) ---
    "cauliflower_black_rot": ["cauliflower_black_rot", "black rot cauliflower"],
    "cauliflower_insect_holes": ["cauliflower_insect_holes", "insect holes", "insect_holes"],
    "cauliflower_healthy": ["cauliflower_healthy", "healthy cauliflower", "cauliflower healthy"],
    # --- cucumber (Cucumber Disease Recognition, Mendeley y6d3z6f8z9) ---
    "cucumber_anthracnose": ["cucumber_anthracnose", "anthracnose"],
    "cucumber_bacterial_wilt": ["cucumber_bacterial_wilt", "bacterial wilt"],
    "cucumber_belly_rot": ["cucumber_belly_rot", "belly rot"],
    "cucumber_downy_mildew": ["cucumber_downy_mildew", "downy mildew cucumber"],
    "cucumber_pythium_fruit_rot": ["cucumber_pythium_fruit_rot", "pythium fruit rot"],
    "cucumber_gummy_stem_blight": ["cucumber_gummy_stem_blight", "gummy stem blight"],
    "cucumber_healthy": ["cucumber_healthy", "fresh leaves cucumber", "cucumber healthy"],
    # --- papaya (BDPapayaLeaf, Mendeley p997fvf526; Ashulia Dhaka field photos) ---
    "papaya_anthracnose": ["papaya_anthracnose", "anthracnose papaya"],
    "papaya_bacterial_spot": ["papaya_bacterial_spot", "bacterial spot papaya"],
    "papaya_curl": ["papaya_curl", "curl papaya", "papaya leaf curl"],
    "papaya_ring_spot": ["papaya_ring_spot", "ring spot papaya", "papaya ringspot"],
    "papaya_healthy": ["papaya_healthy", "healthy papaya", "papaya healthy"],
    # --- mango (MangoLeafBD, Mendeley hxsnvwty3r; BD orchard photos) ---
    "mango_anthracnose": ["mango_anthracnose", "anthracnose mango"],
    "mango_bacterial_canker": ["mango_bacterial_canker", "bacterial canker"],
    "mango_cutting_weevil": ["mango_cutting_weevil", "cutting weevil"],
    "mango_die_back": ["mango_die_back", "die back", "dieback"],
    "mango_gall_midge": ["mango_gall_midge", "gall midge"],
    "mango_powdery_mildew": ["mango_powdery_mildew", "powdery mildew mango"],
    "mango_sooty_mould": ["mango_sooty_mould", "sooty mould", "sooty mold"],
    "mango_healthy": ["mango_healthy", "healthy mango", "mango healthy"],
    # --- okra (Okra DiseaseNet, Mendeley nh7zk4hv8z; Tamil Nadu field photos) ---
    "okra_alternaria_leaf_spot": ["okra_alternaria_leaf_spot", "alternaria leaf spot okra"],
    "okra_cercospora_leaf_spot": ["okra_cercospora_leaf_spot", "cercospora leaf spot okra"],
    "okra_downy_mildew": ["okra_downy_mildew", "downy mildew okra"],
    "okra_leaf_curl_virus": ["okra_leaf_curl_virus", "leaf curl okra", "yellow vein mosaic"],
    "okra_phyllosticta_leaf_spot": ["okra_phyllosticta_leaf_spot", "phyllosticta leaf spot okra"],
    "okra_healthy": ["okra_healthy", "healthy okra", "okra healthy"],
}

# When the same canonical class appears in more than one raw dataset,
# datasets earlier in this list win (more lab-clean data first). Names are
# matched against the FIRST path component under raw_dir, e.g.
# "plantvillage dataset" counts as "plantvillage".
PRIORITY_SOURCES = [
    "plantvillage", "rice_leaf", "riceleafs", "riceleafs_diseases", "plantdoc",
    # v2 crops: lab-style mirrors come first, BD/India field collections after —
    # a field collection wins only when no lab source has that class.
    "corn", "maize", "grape", "chilli", "cauliflower", "cucumber",
    "papaya", "mango", "okra", "olid", "brinjal", "eggplant",
]


def _normalize(name: str) -> str:
    """Lowercase and turn spaces/hyphens into single underscores."""
    out = name.strip().lower()
    for ch in (" ", "-", "___", "__"):
        out = out.replace(ch, "_")
    return out.replace("__", "_").strip("_")


def _squash(name: str) -> str:
    """Remove ALL separators — used for the fuzzy second pass."""
    return "".join(ch for ch in _normalize(name) if ch.isalnum())


def canonical_class_for(folder_name: str) -> str | None:
    """Map a raw dataset folder name to a canonical class, or None to skip.

    Pass 1: normalized exact match. Pass 2: separator-free fuzzy match
    (only used if pass 1 found nothing) — this catches 'LeafBlast',
    'BacterialLeafBlight' style single-word CamelCase folders.
    """
    key = _normalize(folder_name)
    for canonical, aliases in ALIAS_MAP.items():
        for alias in aliases:
            if _normalize(alias) == key:
                return canonical
    squashed = _squash(folder_name)
    for canonical, aliases in ALIAS_MAP.items():
        for alias in aliases:
            if _squash(alias) == squashed and squashed:
                return canonical
    return None


def source_of(image_path: Path, raw_dir: Path) -> str:
    """Return the dataset source name (first component under raw_dir)."""
    try:
        return image_path.relative_to(raw_dir).parts[0].lower()
    except (ValueError, IndexError):
        return ""


def source_rank(source: str) -> int:
    """Lower rank = higher priority. Unlisted sources rank last."""
    # Accept both 'plantvillage' and 'plantvillage dataset' style names.
    for listed, rank in {name: i for i, name in enumerate(PRIORITY_SOURCES)}.items():
        if source == listed or source.startswith(listed):
            return rank
    return len(PRIORITY_SOURCES)


def collect_images(raw_dir: Path, merge_all: bool = False) -> dict[str, list[Path]]:
    """Recursively find images under `raw_dir` and group them by canonical class.

    Images are bucketed per (class, source). Afterwards, for every class the
    source selection depends on `merge_all`:

    - merge_all=False (default, historical behavior): the HIGHEST-priority
      source is chosen as THE single source for that class (avoids
      lab-lab duplicates and mixed domains).
    - merge_all=True: images from ALL sources for that class are combined.
      Use this when progressively adding data on top of the existing set,
      especially real field photos (see ml/src/download_data.md).

    Deduplication: identical paths cannot occur, but the same image copied
    under two sources would count twice — name your additions uniquely.

    Returns {canonical_class: [image paths sorted by name]}.
    """
    buckets: dict[str, dict[str, list[Path]]] = {c: {} for c in CLASSES}
    skipped: list[str] = []

    for image_path in sorted(raw_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        if image_path.name.startswith("."):  # junk files like .DS_Store
            continue

        try:
            relative = image_path.relative_to(raw_dir)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            continue
        source = relative.parts[0].lower()
        class_folder = relative.parts[-2]

        canonical = canonical_class_for(class_folder)
        if canonical is None:
            skipped.append(class_folder)
            continue
        buckets[canonical].setdefault(source, []).append(image_path)

    # ---- pick the source strategy per class --------------------------------
    grouped: dict[str, list[Path]] = {c: [] for c in CLASSES}
    for canonical, by_source in buckets.items():
        if not by_source:
            continue
        if merge_all:
            all_paths: list[Path] = []
            for paths in by_source.values():
                all_paths.extend(paths)
            grouped[canonical] = sorted(all_paths)
            parts = {k: len(v) for k, v in sorted(by_source.items())}
            print(f"[prepare_data] class '{canonical}': MERGED {len(all_paths)} images "
                  f"from sources {parts}")
        else:
            best = sorted(by_source.items(), key=lambda kv: (source_rank(kv[0]), -len(kv[1])))[0]
            grouped[canonical] = sorted(best[1])
            others = {k: len(v) for k, v in by_source.items() if k != best[0]}
            if others:
                print(f"[prepare_data] class '{canonical}': using source '{best[0]}' "
                      f"({len(best[1])} images), ignored sources: {others}")

    if skipped:
        unique_skipped = sorted(set(skipped))
        print(f"[prepare_data] skipped {len(unique_skipped)} unknown class folders (not part of the canonical class list):")
        for name in unique_skipped:
            print(f"    - {name}")
    return grouped


def stratified_split(
    grouped: dict[str, list[Path]],
) -> tuple[dict[str, dict[str, list[Path]]], list[str]]:
    """Split every class independently into train/val/test.

    Very small classes (< MIN_CLASS_COUNT images) cannot fill every split;
    they are skipped entirely and reported. Returns (splits, skipped_classes).
    """
    result: dict[str, dict[str, list[Path]]] = {
        "train": {c: [] for c in CLASSES},
        "val": {c: [] for c in CLASSES},
        "test": {c: [] for c in CLASSES},
    }
    skipped_classes: list[str] = []

    for class_name, paths in grouped.items():
        n = len(paths)
        if n == 0:
            continue  # reported separately as "no images"
        if n < MIN_CLASS_COUNT:
            skipped_classes.append(f"{class_name} ({n} images — too few to split)")
            continue

        paths = sorted(paths)
        # Step 1: carve out the test share.
        train_val, test = train_test_split(
            paths,
            test_size=TEST_FRAC / (TRAIN_FRAC + VAL_FRAC + TEST_FRAC),
            random_state=RANDOM_SEED,
        )
        # Step 2: carve out val from the remaining train+val pool.
        train, val = train_test_split(
            train_val,
            test_size=VAL_FRAC / (TRAIN_FRAC + VAL_FRAC),
            random_state=RANDOM_SEED,
        )
        result["train"][class_name] = sorted(train)
        result["val"][class_name] = sorted(val)
        result["test"][class_name] = sorted(test)

    return result, skipped_classes


def write_split(split_files: dict[str, dict[str, list[Path]]], out_dir: Path) -> int:
    """Copy files to out_dir/<split>/<class>/ with deterministic renamed files.

    Renaming per split/class with a stable index guarantees two datasets
    never collide (both shipping '39.jpg').
    Returns the number of files written.
    """
    if out_dir.exists():
        # Idempotent re-runs start from a clean output dir. `out_dir` is
        # exclusively this script's output; the raw sources are untouched.
        shutil.rmtree(out_dir)
    total = 0
    for split, per_class in split_files.items():
        for class_name, paths in per_class.items():
            if not paths:
                continue
            target = out_dir / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for index, src in enumerate(paths):
                dst = target / f"{class_name}_{index:05d}{src.suffix.lower()}"
                shutil.copy2(src, dst)
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--merge-sources",
        action="store_true",
        help=(
            "Combine images from ALL raw sources per class instead of picking "
            "the single highest-priority source. Use when you have added new "
            "data (e.g. ml/data/raw/field/) on top of the existing datasets."
        ),
    )
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(
            f"Raw data folder not found: {args.raw_dir}\n"
            "See ml/src/download_data.md for the download steps."
        )

    print(f"[prepare_data] scanning {args.raw_dir} ... (merge_all={args.merge_sources})")
    grouped = collect_images(args.raw_dir, merge_all=args.merge_sources)
    print("[prepare_data] per-class image counts:")
    for class_name, paths in grouped.items():
        mark = "" if paths else "   <<< NO IMAGES"
        print(f"    {class_name:<35} {len(paths):>7} images{mark}")

    missing = [c for c in CLASSES if not grouped[c]]
    if missing:
        print("\n[prepare_data] CLASSES WITH NO IMAGES:", missing)
        print("  -> the notebook/NB should stop and ask the maintainer;")
        print("     do NOT train on a partial class list.")

    split, skipped_classes = stratified_split(grouped)
    if skipped_classes:
        print("\n[prepare_data] skipped classes (too few images to split):")
        for name in skipped_classes:
            print(f"    - {name}")

    n = write_split(split, args.out_dir)
    print(f"\n[prepare_data] wrote {n} images to {args.out_dir}")

    for split_name in ("train", "val", "test"):
        counts = {c: len(v) for c, v in split[split_name].items() if v}
        print(f"[prepare_data] {split_name}: {sum(counts.values())} images across {len(counts)} classes")


if __name__ == "__main__":
    main()
