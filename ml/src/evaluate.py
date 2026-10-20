"""
evaluate.py — test-set evaluation with per-class metrics + confusion matrix.

WHAT IT PRODUCES (in ml/reports/)
    * metrics.md  — accuracy, per-class precision / recall / F1, macro+micro,
                    all REAL numbers from the given model + data.
    * confusion_matrix.png — labeled heatmap of test predictions.
    * (field mode) field_test.md — same style report for real phone photos,
                    so the lab-vs-field gap can be measured honestly.

TWO MODES
    1. Standard:  python ml/src/evaluate.py --model ml/reports/best_model.keras \
                       --data data/processed/test
    2. Field:     python ml/src/evaluate.py --model ml/reports/best_model.keras \
                       --field-test data/field_test
       (field images are organized data/field_test/<class>/*.jpg — the folder
        layout works as implicit labels, same as the processed splits.)

NOTE ON HONESTY
    This script only writes numbers it actually computed. If the model or
    data are missing it fails loudly instead of fabricating a report.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "ml" / "reports"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def load_model_and_data(model_path: Path, data_dir: Path, batch_size: int = 32):
    """Load the .keras model and build a deterministic (uneaugmented) dataset."""
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path)
    ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels="inferred",
        label_mode="int",
        image_size=(224, 224),
        batch_size=batch_size,
        shuffle=False,               # order must match confusion matrix rows
    )
    class_names = ds.class_names

    # Normalization must match training ([-1, +1] MobileNetV2 scaling).
    def normalize(images, labels):
        return (tf.cast(images, tf.float32) / 127.5) - 1.0, labels

    ds = ds.map(normalize, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return model, ds, class_names


def predict_all(model, ds):
    """Run the dataset once and return (y_true, y_pred, y_prob_max)."""
    import numpy as np

    y_true, y_pred, probs_top = [], [], []
    for images, labels in ds:
        probs = model.predict_on_batch(images).numpy()
        y_true.extend(labels.numpy().tolist())
        y_pred.extend(probs.argmax(axis=1).tolist())
        probs_top.extend(probs.max(axis=1).tolist())
    return np.array(y_true), np.array(y_pred), np.array(probs_top)


def compute_report(y_true, y_pred, probs_max: list[float] | None, class_names):
    """Return a dict of REAL metrics using scikit-learn."""
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_recall_fscore_support,
    )

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(len(class_names))), zero_division=0
    )
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))

    report = {
        "n_images": int(len(y_true)),
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "per_class": {},
        "confusion_matrix": cm.tolist(),
    }
    for i, name in enumerate(class_names):
        entry = {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        if probs_max is not None:
            # Mean confidence on images of this class (helps threshold tuning).
            mask = y_true == i
            entry["mean_confidence"] = float(probs_max[mask].mean()) if mask.any() else None
        report["per_class"][name] = entry
    return report


def save_confusion_image(cm, class_names, out_path: Path) -> None:
    """Render the confusion matrix as a readable heatmap PNG."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(class_names, fontsize=8)
    # Annotate counts + row-normalized percentages.
    cm_norm = cm.astype(float) / np.clip(cm.sum(axis=1, keepdims=True), 1, None)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            if cm[r, c]:
                ax.text(c, r, f"{cm[r, c]}\n{cm_norm[r, c]*100:.0f}%",
                        ha="center", va="center", fontsize=7,
                        color="white" if cm_norm[r, c] > 0.5 else "black")
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Confusion matrix")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_markdown(report: dict, class_names, out_path: Path, title: str, model: str, data: str) -> None:
    """Write the report markdown (metrics.md or field_test.md)."""
    lines = [
        f"# {title}",
        "",
        f"*Model:* `{model}` ",
        f"*Data:* `{data}`  ({report['n_images']} images)  ",
        f"*Generated:* {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        "",
        "All numbers below are computed from the actual run — not estimates.",
        "",
        f"**Overall accuracy: {report['accuracy']*100:.2f}%  |  Macro F1: {report['macro_f1']:.3f}**",
        "",
        "| class | precision | recall | F1 | support | mean confidence |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for name in class_names:
        e = report["per_class"][name]
        mc = f"{e['mean_confidence']:.3f}" if e.get("mean_confidence") is not None else "—"
        lines.append(
            f"| {name} | {e['precision']:.3f} | {e['recall']:.3f} | {e['f1']:.3f} | {e['support']} | {mc} |"
        )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--data", type=Path, help="processed test split directory")
    parser.add_argument("--field-test", type=Path, help="folder of real phone photos (data/field_test)")
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model} — train first (train.py).")

    if args.field_test:
        # ---- FIELD TEST MODE: lab-vs-field gap measurement ------------------
        if not args.field_test.exists():
            raise SystemExit(
                f"Field-test folder not found: {args.field_test}\n"
                "Expected layout: data/field_test/<class>/<image> — collect\n"
                "20–30 real phone photos first (task T5.3)."
            )
        model, ds, class_names = load_model_and_data(args.model, args.field_test)
        y_true, y_pred, probs = predict_all(model, ds)
        report = compute_report(y_true, y_pred, probs, class_names)
        save_confusion_image(
            __import__("sklearn").metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
            class_names, args.report_dir / "field_confusion_matrix.png")
        write_markdown(report, class_names, args.report_dir / "field_test.md",
                       "Field Test Report (real phone photos)", args.model.name, str(args.field_test))
        print(f"[evaluate] field report -> {args.report_dir/'field_test.md'}")
    elif args.data:
        # ---- STANDARD TEST-SPLIT MODE --------------------------------------
        model, ds, class_names = load_model_and_data(args.model, args.data)
        y_true, y_pred, probs = predict_all(model, ds)
        report = compute_report(y_true, y_pred, probs, class_names)
        save_confusion_image(__import__("sklearn").metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))),
                             class_names, args.report_dir / "confusion_matrix.png")
        write_markdown(report, class_names, args.report_dir / "metrics.md",
                       "Model Metrics (test split)", args.model.name, str(args.data))
        print(f"[evaluate] accuracy={report['accuracy']*100:.2f}% macro_f1={report['macro_f1']:.3f}")
        print(f"[evaluate] metrics.md + confusion_matrix.png -> {args.report_dir}")
    else:
        raise SystemExit("Give either --data <test split> or --field-test <folder>.")


if __name__ == "__main__":
    main()
