"""
train.py — two-stage transfer-learning training.

STAGE 1 (head training)
    Backbone frozen. Only the new Dense head learns. Few epochs, higher LR.
STAGE 2 (fine-tuning)
    Unfreeze the top `FINE_TUNE_LAYERS` backbone layers (model.py), tiny LR
    (STAGE2_LR = 1e-5) so pretrained weights stay useful.

Other pieces (per project spec):
    * class weights  — computed from the training split to counterbalance
                       the heavy PlantVillage tomato classes.
    * early stopping — on val_accuracy with patience; restores best weights.
    * checkpointing  — saves the best model by val_accuracy per stage.
    * history.json   — full Keras history + stage metadata saved to
                       ml/reports/, so the final report uses REAL numbers.

USAGE (GPU expected — Kaggle/Colab; CPU works but slow)
    python ml/src/train.py --train data/processed/train --val data/processed/val \
        [--epochs-head 5 --epochs-ft 15 --batch 32 --out-model ml/reports/best_model.keras]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf  # module level: make_callbacks() needs it at call time

from augmentations import build_datasets
from classes import missing_from_dataset
from model import STAGE1_LR, STAGE2_LR, build_model, unfreeze_for_finetune

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "ml" / "reports"

# ---------------------------------------------------------------------------
# Early stopping / patience settings
# ---------------------------------------------------------------------------
PATIENCE_HEAD = 3               # stop head training quickly if no progress
PATIENCE_FT = 5                 # fine-tuning is slower; allow more patience


def compute_class_weights(class_names: list[str], train_dir: Path) -> dict[int, float]:
    """Compute class weights from the number of images per class folder.

    Formula: n_samples / (n_classes * n_class_samples) — the sklearn
    'balanced' convention. Keeps rare classes (rice diseases) from being
    drowned out by tomato's ~18k images.

    Args:
        class_names: the model's real output order (the Keras folder order),
            NOT classes.py order — otherwise the weight for index 0 would be
            applied to the wrong class and quietly bias the model.
        train_dir: processed training split, one sub-folder per class.

    Returns:
        {output_index: weight} ready to pass to `model.fit(class_weight=...)`.
    """
    counts: list[int] = []
    for class_name in class_names:
        class_dir = train_dir / class_name
        n = sum(
            1
            for p in class_dir.glob("*")
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
        )
        counts.append(n)

    total = sum(counts)
    n_classes = len(class_names)
    # Avoid exactly-uniform weights when data is perfectly balanced.
    weights = {
        i: (total / (n_classes * count)) if count > 0 else 1.0
        for i, count in enumerate(counts)
    }
    return weights


def make_callbacks(stage: str, report_dir: Path, checkpoint_path: Path):
    """Callbacks shared by both stages: checkpoint best-on-val, stop early.
    `stage` is only used to tag logs, so the history file tells the two
    runs apart."""
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=PATIENCE_HEAD if stage == "head" else PATIENCE_FT,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(report_dir / f"train_log_{stage}.csv"), append=False),
    ]


def merge_histories(h1, h2) -> dict[str, list[float]]:
    """Merge two Keras History objects into one flat dict (merged epoch axis)."""
    merged: dict[str, list[float]] = {}
    for key in h1.history:
        merged[key] = list(h1.history[key]) + list(h2.history.get(key, []))
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=REPO_ROOT / "data" / "processed" / "train")
    parser.add_argument("--val", type=Path, default=REPO_ROOT / "data" / "processed" / "val")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--out-model", type=Path, default=DEFAULT_REPORT_DIR / "best_model.keras")
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-ft", type=int, default=15)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()

    args.report_dir.mkdir(parents=True, exist_ok=True)

    # ---- 1. Data ----------------------------------------------------------
    train_ds, val_ds, class_names = build_datasets(
        str(args.train), str(args.val), batch_size=args.batch, seed=42
    )
    # ---- 1b. Label sanity check -----------------------------------------
    # Keras labels the classes in ALPHABETICAL FOLDER order, and the trained
    # model's output index follows that order. classes.py groups by crop, so
    # the two orders legitimately differ. What must NOT happen is a folder
    # that is not a known class (silent extra label) or a labels.json written
    # in a different order than training used.
    #   -> we verify the folder SET here, and record the exact order that
    #      training used in ml/reports/class_order.json so export_onnx.py can
    #      replay it when it writes backend/app/data/labels.json.
    missing = missing_from_dataset(class_names)
    if missing:
        print(
            "[train] NOTE: the dataset has no images for "
            f"{len(missing)} canonical class(es): {missing}\n"
            f"[train] Training will cover {len(class_names)} classes only. "
            "Add data for the missing ones later and re-run prepare_data.py."
        )
    class_order_path = args.report_dir / "class_order.json"
    class_order_path.write_text(
        json.dumps(
            {
                "class_names": class_names,
                "n_classes": len(class_names),
                "source": "Keras image_dataset_from_directory (alphabetical folder order)",
                "note": "The model's output index IS this list order. export_onnx.py reads this file.",
            },
            indent=2,
        )
    )
    print(f"[train] label order ({len(class_names)} classes) saved to {class_order_path}")

    class_weights = compute_class_weights(class_names, args.train)
    print("[train] class weights (top/low):",
          sorted(class_weights.items(), key=lambda kv: -kv[1])[:3],
          "...", sorted(class_weights.items(), key=lambda kv: kv[1])[:3])

    # ---- 2. Stage 1: train the frozen model -------------------------------
    # Output width MUST match the number of classes the dataset actually has,
    # otherwise the head produces dead logits for missing classes and softmax
    # probabilities are spread over labels the model can never learn.
    model = build_model(len(class_names))
    print("[train] STAGE 1 — head training (backbone frozen)")
    h1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_head,
        class_weight=class_weights,
        callbacks=make_callbacks("head", args.report_dir, args.out_model),
    )

    # ---- 3. Stage 2: fine-tune the top backbone layers --------------------
    model = unfreeze_for_finetune(model)
    print(f"[train] STAGE 2 — fine-tuning top layers at lr={STAGE2_LR}")
    h2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_ft,
        class_weight=class_weights,
        callbacks=make_callbacks("finetune", args.report_dir, args.out_model),
    )

    # ---- 4. Save merged history + final model location --------------------
    merged = merge_histories(h1, h2)
    best_val_acc = max(merged.get("val_accuracy", [0.0]))
    history_payload = {
        "class_names": class_names,   # real order used by the trained model
        "class_weights": class_weights,
        "stage1_epochs_run": len(h1.history["loss"]),
        "stage2_epochs_run": len(h2.history["loss"]),
        "stage1_lr": STAGE1_LR,
        "stage2_lr": STAGE2_LR,
        "best_val_accuracy": best_val_acc,
        "best_model_path": str(args.out_model),
        "history": merged,
    }
    history_path = args.report_dir / "history.json"
    history_path.write_text(json.dumps(history_payload, indent=2))
    print(f"[train] history written to {history_path}")
    print(f"[train] best val accuracy: {best_val_acc:.4f} (REAL number — do not edit)")


if __name__ == "__main__":
    main()
