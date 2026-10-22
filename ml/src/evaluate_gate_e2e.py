"""Full end-to-end evaluation on the REAL test split: gate + main model.

Measures, for a given gate ONNX checkpoint:
  * leaf pass rate  — what fraction of true test-split leaves the gate
    lets through (lower = false rejections, the dangerous failure mode);
  * main-model top-1 accuracy — computed two ways: over ALL true leaves
    (fails count as wrong) and over GATE-PASSED leaves (what a farmer
    actually experiences);
  * junk pass rate on synthetic junk + harder background crops (what lands
    on the honest low-confidence screen vs. being refused outright).

Usage:
  python ml/src/evaluate_gate_e2e.py --gate ml/reports/_gate_before.onnx --tag before
  python ml/src/evaluate_gate_e2e.py --tag after
Results are appended into ml/reports/gate_e2e_report.json (side by side).
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(HERE))

from train_plant_classifier import synth_negative, background_crops, degrade  # noqa: E402
from app.services import inference, preprocess as pp, plant_guard  # noqa: E402

PROC = REPO / "data" / "processed"
REPORTS = REPO / "ml" / "reports"
MODEL = REPO / "backend" / "app" / "data" / "plant_guard_model.onnx"
EVAL_SEED = 4242
USER_NEG = HERE / "data" / "negatives" / "user_leaf_like.jpg"


def install_gate(checkpoint: Path) -> None:
    """Point the serving gate at the requested checkpoint for this run."""
    if checkpoint.resolve() != MODEL.resolve():
        shutil.copyfile(checkpoint, MODEL)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=4, help="test images per class (61 classes)")
    ap.add_argument("--n-junk", type=int, default=300)
    ap.add_argument("--gate", type=str, default=None, help="ONNX checkpoint to evaluate (default: serving one)")
    ap.add_argument("--tag", type=str, required=True, help="label stored in the report, e.g. before/after")
    args = ap.parse_args()

    if args.gate:
        install_gate(Path(args.gate))
    plant_guard._GATE_PATH = MODEL  # explicit: serving uses this checkpoint
    inference.load_model()

    if not PROC.is_dir():
        print("data/processed missing — cannot evaluate", file=sys.stderr)
        return 1

    rng = random.Random(EVAL_SEED)
    nprng = np.random.default_rng(EVAL_SEED)

    # ---------- positives: real test-split leaves, spread over all classes.
    pos: list[tuple[Path, str]] = []
    for d in sorted(PROC.glob("test/*")):
        if not d.is_dir():
            continue
        files = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        pos += [(Path(f), d.name) for f in rng.sample([str(f) for f in files], min(args.per_class, len(files)))]
    neg_rng = np.random.default_rng(EVAL_SEED)

    # ---------- negatives: seeds 4242 synthetic junk + harder bg crops ------
    negs: list[Image.Image] = []
    n_synth, n_bg = args.n_junk // 2, args.n_junk // 2
    while len(negs) < n_synth:
        negs.extend(synth_negative(nprng))
    negs = negs[:n_synth]
    bg_dirs = sorted(d for d in PROC.glob("test/*") if d.is_dir())
    while len(negs) < n_synth + n_bg:
        mode = "leaffree" if len(negs) % 5 < 3 else "partial"
        batch = background_crops(nprng, bg_dirs, mode=mode, need=1)
        if not batch:
            continue
        c = batch[0]
        if mode == "leaffree" and nprng.random() < 0.3:
            c = degrade(c, float(nprng.uniform(0.4, 0.95)), float(nprng.uniform(0.5, 4.0)), int(nprng.integers(40, 88)))
        negs.append(c)

    # ---------- run the real serving path ----------------------------------
    leaf_passed = leaf_failed = 0
    top1_all_correct = 0
    top1_gate_passed = gate_passed_total = 0
    refused = []
    for f, expected in pos:
        img = Image.open(f).convert("RGB")
        img.thumbnail((1024, 1024))
        ok, reason = plant_guard.looks_like_leaf(img)
        if not ok:
            leaf_failed += 1
            refused.append((f.name, expected, reason))
            continue
        leaf_passed += 1
        gate_passed_total += 1
        batch = pp.preprocess(img)
        top1 = inference.predict_top3(batch)[0]
        if top1["class_name"] == expected:
            top1_gate_passed += 1
        top1_all_correct += int(top1["class_name"] == expected)

    junk_passed = 0
    for im in negs:
        ok, _ = plant_guard.looks_like_leaf(im)
        junk_passed += int(ok)

    total_leaves = leaf_passed + leaf_failed
    result = {
        "tag": args.tag,
        "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "gate_checkpoint": (MODEL.__str__() if args.gate else "serving"),
        "positives_n": total_leaves,
        "junk_n": len(negs),
        "gate_leaf_pass_rate": round(leaf_passed / max(total_leaves, 1), 4),
        "gate_refusal_top1": 0.0 if not total_leaves else round(top1_all_correct / total_leaves, 4),
        "top1_accuracy_all_leaves": round(top1_all_correct / max(total_leaves, 1), 4),
        "top1_accuracy_gate_passed": round(top1_gate_passed / max(gate_passed_total, 1), 4),
        "junk_pass_rate": round(junk_passed / max(len(negs), 1), 4),
        "gate_leaf_refusals": leaf_failed,
        "gate_refusal_samples": [
            {"file": n, "true_class": c, "reason": r[:90]} for n, c, r in refused[:5]
        ],
    }

    path = REPORTS / "gate_e2e_report.json"
    report = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    report[args.tag] = result
    path.write_text(json.dumps(report, indent=2))

    print(json.dumps(result, indent=2))
    print(f"saved -> {path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
