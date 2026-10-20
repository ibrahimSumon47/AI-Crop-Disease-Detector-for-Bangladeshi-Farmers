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
          field_test/<class_name>/...   (optional, human-collected, untouched)

    - 70 % train / 15 % val / 15 % test, split *stratified per class*
      so that rare classes keep roughly proportional representation.
    - Fixed RANDOM_SEED so the split is reproducible across runs/machines.
    - Files are COPIED (not moved), so the raw sources stay reusable.
    - Dataset layouts can differ (class-nested subfolders); the script
      scans recursively and maps folder names to canonical classes via
      ALIAS_MAP below.

USAGE
    python ml/src/prepare_data.py [--raw-dir ml/data/raw]
                                  [--out-dir data/processed]
                                  [--train 70 --val 15]  # val+test→test
    Requires: Pillow, numpy, scikit-learn. No GPU needed.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from sklearn.model_selection import train_test_split

from classes import CLASSES, crop_of

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

# ---------------------------------------------------------------------------
# Folder-name aliases — every raw dataset names classes differently.
# Key: canonical class from classes.py. Values: case-insensitive folder
# names that should map to it. Only ONE dataset is used per class; if a
# duplicate class arrives from two datasets, the FIRST source wins
# (see PRIORITY list below).
# ---------------------------------------------------------------------------

ALIAS_MAP: dict[str, list[str]] = {
    # --- rice ---
    "rice_bacterial_leaf_blight": ["bacterial_leaf_blight", "bacterial leaf blight", "bacterialblight", "blb"],
    "rice_brown_spot": ["brown_spot", "brown spot", "brownspot"],
    "rice_blast": ["blast", "leaf_blast", "leaf blast", "rice_blast"],
    "rice_tungro": ["tungro", "rice_tungro"],
    "rice_healthy": ["healthy", "healthy_leaf", "rice_healthy", "sick_healty_more", "healty"],
    # --- potato (PlantVillage) ---
    "potato_early_blight": ["potato___early_blight", "potato___earlyblight", "potato_early_blight"],
    "potato_late_blight": ["potato___late_blight", "potato___lateblight", "potato_late_blight"],
    "potato_healthy": ["potato___healthy", "potato_healthy"],
    # --- tomato (PlantVillage) ---
    "tomato_bacterial_spot": ["tomato_bacterial_spot", "tomato___bacterial_spot"],
    "tomato_early_blight": ["tomato_early_blight", "tomato___early_blight"],
    "tomato_late_blight": ["tomato_late_blight", "tomato___late_blight"],
    "tomato_leaf_mold": ["tomato_leaf_mold", "tomato___leaf_mold", "tomato___leaf_mold_placeholder"],
    "tomato_healthy": ["tomato_healthy", "tomato___healthy"],
}

# When the same canonical class appears in more than one raw dataset,
# datasets earlier in this list win (more lab-clean data first).
PRIORITY_SOURCES = ["plantvillage", "rice_leaf", "riceleafs", "plantdoc"]


def canonical_class_for(folder_name: str) -> str | None:
    """Map a raw dataset folder name to a canonical class, or None to skip.

    Case- and whitespace-insensitive matching through ALIAS_MAP.
    """
    key = folder_name.strip().lower().replace("-", "_").replace(" ", "_")
    for canonical, aliases in ALIAS_MAP.items():
        for alias in aliases:
            if alias.strip().lower().replace(" ", "_") == key:
                return canonical
    return None


def collect_images(raw_dir: Path) -> dict[str, list[Path]]:
    """Recursively find images under `raw_dir` and group them by canonical class.

    The source name (plantvillage / rice_leaf / plantdoc) is inferred from
    the FIRST path component under raw_dir, which drives the priority rule.
    Returns {canonical_class: [image paths sorted by name]}.
    """
    grouped: dict[str, list[Path]] = {c: [] for c in CLASSES}
    skipped: list[str] = []

    for image_path in sorted(raw_dir.rglob("*")):
        if not image_path.is_file():
            continue
        if image_path.suffix.lower() not in VALID_IMAGE_EXTENSIONS:
            continue
        if image_path.name.startswith("."):  # junk files like .DS_Store
            continue

        # Infer which dataset this file belongs to: path like
        # raw/<source>/<...class folders...>/img.jpg
        try:
            relative = image_path.relative_to(raw_dir)
        except ValueError:
            continue
        if len(relative.parts) < 2:
            continue
        source = relative.parts[0].lower()
        # Class folder = last folder component in the relative path.
        class_folder = relative.parts[-2]

        canonical = canonical_class_for(class_folder)
        if canonical is None:
            skipped.append(class_folder)
            continue

        # Priority: only record the source once per class; if this class
        # already has images from a higher-priority source, skip extras.
        existing = grouped[canonical]
        if existing:
            existing_source = source_of(existing[0], raw_dir)
            if source_rank(existing_source) <= source_rank(source):
                continue
        grouped[canonical].append(image_path)

    if skipped:
        unique_skipped = sorted(set(skipped))
        print(f"[prepare_data] skipped {len(unique_skipped)} unknown class folders:")
        for name in unique_skipped:
            print(f"    - {name}")
    return grouped


def source_of(image_path: Path, raw_dir: Path) -> str:
    """Return the dataset source name (first component under raw_dir)."""
    try:
        return image_path.relative_to(raw_dir).parts[0].lower()
    except (ValueError, IndexError):
        return ""


def source_rank(source: str) -> int:
    """Lower rank = higher priority."""
    order = {name: i for i, name in enumerate(PRIORITY_SOURCES)}
    return order.get(source, len(order))


def stratified_split(
    grouped: dict[str, list[Path]],
) -> dict[str, dict[str, list[Path]]]:
    """Split every class independently into train/val/test.

    Stratified per class because we split each class's file list with its
    own ratios — exactly what "stratified" means for a folder-of-images
    dataset. All splits and the final combined shuffling use RANDOM_SEED.

    Returns {split: {class: [paths]}}.
    """
    result: dict[str, dict[str, list[Path]]] = {
        "train": {c: [] for c in CLASSES},
        "val": {c: [] for c in CLASSES},
        "test": {c: [] for c in CLASSES},
    }

    for class_name, paths in grouped.items():
        if not paths:
            print(f"[prepare_data] WARNING: no images found for class '{class_name}'")
            continue

        # Shuffle deterministically BEFORE splitting (useful when a raw
        # folder is ordered, e.g. all "good" shots first).
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

    return result


def write_split(split_files: dict[str, dict[str, list[Path]]], out_dir: Path) -> int:
    """Copy files to out_dir/<split>/<class>/IMG_train_0001.jpg style names.

    Filenames are renamed per split/class with a stable index so that two
    datasets never collide (e.g. both having '39.jpg').
    Returns the number of files written.
    """
    if out_dir.exists():
        # Idempotent re-runs start from a clean output dir. `out_dir` is
        # exclusively this script's output; the raw sources are untouched.
        shutil.rmtree(out_dir)
    total = 0
    for split, per_class in split_files.items():
        for class_name, paths in per_class.items():
            target = out_dir / split / class_name
            target.mkdir(parents=True, exist_ok=True)
            for index, src in enumerate(paths):
                stem = crop_of(class_name) + "_" + class_name.split("_", 1)[1]
                dst = target / f"{stem}_{index:05d}{src.suffix.lower()}"
                shutil.copy2(src, dst)
                total += 1
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=DEFAULT_RAW_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()

    if not args.raw_dir.exists():
        raise SystemExit(
            f"Raw data folder not found: {args.raw_dir}\n"
            "See ml/src/download_data.md for the download steps."
        )

    print(f"[prepare_data] scanning {args.raw_dir} ...")
    grouped = collect_images(args.raw_dir)
    for class_name, paths in grouped.items():
        print(f"    {class_name:<35} {len(paths):>6} images")

    missing = [c for c in CLASSES if not grouped[c]]
    if missing:
        print(f"[prepare_data] NOTE: classes with no images: {missing}")
        print("  -> revise ml/src/classes.py / ALIAS_MAP or find another dataset.")
        print("  -> STOP and report to the user; do NOT train on a partial list.")

    split = stratified_split(grouped)
    n = write_split(split, args.out_dir)
    print(f"[prepare_data] wrote {n} images to {args.out_dir}")

    # Print counts per split so the output can be pasted into reports.
    for split_name in ("train", "val", "test"):
        counts = {c: len(v) for c, v in split[split_name].items() if v}
        print(f"[prepare_data] {split_name}: {sum(counts.values())} images across {len(counts)} classes")


if __name__ == "__main__":
    main()
