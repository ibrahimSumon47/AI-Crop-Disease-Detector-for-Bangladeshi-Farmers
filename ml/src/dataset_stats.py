"""
dataset_stats.py — print class counts per split and save a bar chart.

WHY
    Before training you want to SEE the imbalance: PlantVillage's tomato
    classes have thousands of images while a rice-class dataset may have
    only a few hundred. This picture is what justifies (a) class weights
    in train.py and (b) augmentation strength in augmentations.py.

INPUT
    data/processed/{train,val,test}/<class>/*.jpg  — as produced by
    prepare_data.py.

OUTPUT
    1. A printed table of counts per class x split.
    2. ml/reports/class_distribution.png — one grouped bar chart per
       class (train/val/test side by side).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Use a non-interactive backend so the script works on headless machines
# (Kaggle notebooks, CI) without X display.
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402  (must come after use("Agg"))

from classes import CLASSES  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = REPO_ROOT / "data" / "processed"
DEFAULT_REPORT_DIR = REPO_ROOT / "ml" / "reports"
SPLITS = ["train", "val", "test"]
SPLIT_COLORS = {"train": "#2e7d32", "val": "#f9a825", "test": "#1565c0"}


def count_images(split_dir: Path) -> dict[str, int]:
    """Count images per class folder in one split directory."""
    counts = {}
    if not split_dir.exists():
        return counts
    for class_dir in sorted(split_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        n = sum(1 for f in class_dir.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        counts[class_dir.name] = n
    return counts


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Collect counts ------------------------------------------------
    all_counts: dict[str, dict[str, int]] = {
        split: count_images(args.data_dir / split) for split in SPLITS
    }

    if not any(all_counts.values()):
        print(f"No processed data found under {args.data_dir}.")
        print("Run `python ml/src/prepare_data.py` first (see ml/src/download_data.md).")
        return

    # ---- 2. Print the table ----------------------------------------------
    print(f"{'class':<35}{'train':>8}{'val':>8}{'test':>8}{'total':>9}")
    print("-" * 68)
    totals = {s: 0 for s in SPLITS}
    for class_name in CLASSES:
        counts = [all_counts[s].get(class_name, 0) for s in SPLITS]
        for split, n in zip(SPLITS, counts):
            totals[split] += n
        total = sum(counts)
        flag = " !" if total == 0 else "  "
        print(f"{class_name:<35}{counts[0]:>8}{counts[1]:>8}{counts[2]:>8}{total:>9}{flag}")
    print("-" * 68)
    print(f"{'TOTAL':<35}{totals['train']:>8}{totals['val']:>8}{totals['test']:>8}{sum(totals.values()):>9}")
    print("( ' !' marks classes with zero images in all splits — they need attention)")

    # ---- 3. Draw the grouped bar chart ------------------------------------
    fig, ax = plt.subplots(figsize=(12, 6))
    x_positions = range(len(CLASSES))
    bar_width = 0.26

    for i, split in enumerate(SPLITS):
        values = [all_counts[split].get(c, 0) for c in CLASSES]
        ax.bar(
            [x + (i - 1) * bar_width for x in x_positions],
            values,
            width=bar_width,
            label=split,
            color=SPLIT_COLORS[split],
        )

    ax.set_xticks(list(x_positions))
    ax.set_xticklabels(CLASSES, rotation=45, ha="right")
    ax.set_ylabel("image count")
    ax.set_title("Dataset class distribution per split")
    ax.legend()
    fig.tight_layout()

    out_path = args.report_dir / "class_distribution.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[dataset_stats] chart saved to {out_path}")


if __name__ == "__main__":
    main()
