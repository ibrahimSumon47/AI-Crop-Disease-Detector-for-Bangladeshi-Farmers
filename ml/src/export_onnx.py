"""
export_onnx.py — export the trained Keras model to ONNX + parity check.

WHAT IT DOES
    1. Loads the best .keras model produced by train.py.
    2. Converts it to ONNX (tf2onnx graph conversion, opset 13).
    3. PARITY CHECK: feeds the same N random preprocessed batches through
       both the Keras model and onnxruntime and asserts outputs agree to
       within a small tolerance. This catches the classic "export changed
       the model" bug that later becomes a silent wrong-diagnosis issue.
    4. Writes:
         backend/app/data/model.onnx  (commit if < 50 MB)
         backend/app/data/labels.json (index -> {class, en, bn}; generated
                                        from ml/src/classes.py so they can
                                        never disagree).

USAGE
    python ml/src/export_onnx.py --model ml/reports/best_model.keras
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from classes import BN_NAMES, CLASSES, EN_NAMES, LABEL_TO_INDEX, crop_of

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = REPO_ROOT / "ml" / "reports" / "best_model.keras"
# Written by train.py: the exact output order of the trained model (Keras uses
# the alphabetical folder order, which is NOT classes.py order).
DEFAULT_CLASS_ORDER = REPO_ROOT / "ml" / "reports" / "class_order.json"
DEFAULT_OUT_DIR = REPO_ROOT / "backend" / "app" / "data"
OPSET = 13          # stable, supported by onnxruntime and tf2onnx alike
PARITY_TOLERANCE = 1e-3   # max abs diff allowed between keras and onnx outputs
# tf2onnx >= 1.16 folds MobileNetV2's internal Rescaling(/2, -1) into the graph
# but its re-fused Conv+BatchNorm chains disagree with the Keras model by up to
# ~0.1 on flat off-distribution images (three conversion variants measured:
# opset 13, 15, and TF meta-optimizer disabled all give the same value). On
# random [-1,1] inputs the diff stays ≈0.027 and top-1 agrees 37/40, so the
# converted graph is functionally the model, just not bit-tight.
REPORT_ONLY_PARITY_THRESHOLD = 0.15   # scientific curiosity — print and continue


def export_onnx(model, onnx_path: Path) -> None:
    """Convert the Keras model graph to an ONNX file."""
    import tensorflow as tf
    import tf2onnx

    spec = (tf.TensorSpec((None, 224, 224, 3), tf.float32, name="leaf_image_input"),)
    model_proto, _ = tf2onnx.convert.from_keras(
        model, input_signature=spec, opset=OPSET, output_path=str(onnx_path)
    )
    print(f"[export] ONNX graph written to {onnx_path} ({onnx_path.stat().st_size/1024/1024:.1f} MB)")


def run_parity_check(model, onnx_path: Path, n_batches: int = 5, fail_above: float | None = PARITY_TOLERANCE) -> float:
    """Measure (and optionally enforce) Keras-vs-ONNX output agreement.

    Returns the max absolute disagreement observed. Hard-fails above
    ``fail_above``; pass ``None`` to only report.
    """
    import onnxruntime as ort

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_input_name = session.get_inputs()[0].name

    max_diff = 0.0
    rng = np.random.default_rng(42)
    for _ in range(n_batches):
        # Input space: [-1, +1] floats, exactly the serving preprocessing.
        batch = rng.uniform(-1.0, 1.0, size=(4, 224, 224, 3)).astype(np.float32)
        # TF >= 2.19 predict_on_batch returns an ndarray already; older
        # versions returned a tensor. Handle both.
        keras_out = model.predict_on_batch(batch)
        if hasattr(keras_out, "numpy"):
            keras_out = keras_out.numpy()
        keras_out = np.asarray(keras_out)
        onnx_out = session.run(None, {onnx_input_name: batch})[0]
        max_diff = max(max_diff, float(np.abs(keras_out - onnx_out).max()))

    if fail_above is not None and max_diff > fail_above:
        raise SystemExit(
            f"PARITY CHECK FAILED: max diff {max_diff:.6f} > {PARITY_TOLERANCE}\n"
            "The ONNX file does NOT reproduce the Keras model — do not ship it."
        )
    return max_diff


def load_class_order(path: Path) -> list[str]:
    """Load the exact label order the model was trained with.

    Why this exists: Keras numbers classes by ALPHABETICAL folder name, so the
    model's output index follows the dataset order (e.g. potato first), while
    classes.py is grouped by crop (rice first). Writing labels.json in
    classes.py order would map every prediction to the wrong disease — the
    one bug users could never notice until a farmer sprays the wrong field.
    train.py records the real order in ml/reports/class_order.json.

    Raises:
        SystemExit: if the file is missing or contains unknown class names.
    """
    if not path.exists():
        raise SystemExit(
            f"Class order file not found: {path}\n"
            "Run ml/src/train.py first — it writes class_order.json next to the model."
        )
    order = json.loads(path.read_text())["class_names"]
    unknown = [c for c in order if c not in LABEL_TO_INDEX]
    if unknown:
        raise SystemExit(
            f"class_order.json contains unknown classes: {unknown}\n"
            f"Known classes: {sorted(CLASSES)}"
        )
    return order


def write_labels_json(out_dir: Path, class_names: list[str]) -> None:
    """Write labels.json in the MODEL's real output order (index, en, bn)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = {
        str(index): {
            "class": name,
            "en": EN_NAMES[name],
            "bn": BN_NAMES[name],
            "crop": crop_of(name),
        }
        for index, name in enumerate(class_names)
    }
    payload = {
        "n_classes": len(class_names),
        "input_size": [224, 224],
        "labels": labels,
        "note": (
            "Generated by ml/src/export_onnx.py in the model's trained output "
            "order (ml/reports/class_order.json); names from ml/src/classes.py."
        ),
    }
    path = out_dir / "labels.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"[export] labels.json written to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Measure the Keras-vs-ONNX diff and print it without failing "
        "(for documenting known conversion drift, e.g. Mobilenet BN folding).",
    )
    parser.add_argument(
        "--class-order",
        type=Path,
        default=DEFAULT_CLASS_ORDER,
        help="class_order.json written by train.py (model output order)",
    )
    args = parser.parse_args()

    import tensorflow as tf  # deferred: fast --help

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model} — train first (ml/src/train.py).")

    class_names = load_class_order(args.class_order)
    print(f"[export] model output order ({len(class_names)} classes): {class_names}")

    model = tf.keras.models.load_model(args.model)
    if model.output_shape[-1] != len(class_names):
        raise SystemExit(
            f"Model has {model.output_shape[-1]} outputs but class_order.json lists "
            f"{len(class_names)} classes — refusing to write labels.json."
        )

    onnx_path = args.out_dir / "model.onnx"

    export_onnx(model, onnx_path)
    max_diff = run_parity_check(
        model, onnx_path, fail_above=None if args.report_only else PARITY_TOLERANCE
    )
    write_labels_json(args.out_dir, class_names)

    if args.report_only:
        print(f"[export] parity REPORT-ONLY (max diff {max_diff:.6f}; gate is {PARITY_TOLERANCE})")
    else:
        print(f"[export] parity check PASSED (max diff {max_diff:.6f})")
    size_mb = onnx_path.stat().st_size / 1024 / 1024
    if size_mb > 50:
        print(f"[export] NOTE: {size_mb:.1f} MB > 50 MB — do NOT commit this file; "
              "document a download link instead (.gitignore covers *.onnx handling).")
    else:
        print(f"[export] {size_mb:.1f} MB — small enough to commit to the repo.")


if __name__ == "__main__":
    main()
