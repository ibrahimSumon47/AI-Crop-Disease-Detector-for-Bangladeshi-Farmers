"""
evaluate_plant_gate.py — hold real evaluation of the SERVING gate.

Trains on one distribution, so numbers can only be believed from data the
trainer never saw. This script imports the SERVING feature extractor
(backend/app/services/plant_guard._classifier_features) so we measure the
exact code path farmers hit, then evaluates on:

Positives (fresh sample, NOT the trainer's val set):
  random leaves from data/processed/test (all classes).

Negatives (all unseen by training):
  * synthetic photo-junk generator with a DIFFERENT seed than training
    (the training run used seed 42 for everything, which also made its own
    val negatives all identical — that bug is why this script exists);
  * background crops cut from real data/processed/test leaves (nearly
    leaf-free windows);
  * the real user-submitted non-leaf photo + augmentations.

Output: P(leaf) distribution for both classes, a threshold sweep, and the
recommended GATE_THRESHOLD. Numbers are appended to
ml/reports/plant_classifier_metrics.json under "serving_holdout".

Usage:  python ml/src/evaluate_plant_gate.py [--n-pos 600] [--n-neg 600]
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "backend"))
sys.path.insert(0, str(HERE))  # reuse the synthetic/background generators

from app.services import plant_guard  # noqa: E402  (serving code — the thing we measure)
from train_plant_classifier import synth_negative, augment, background_crop  # noqa: E402

REPORTS = REPO / "ml" / "reports"
PROC = REPO / "data" / "processed"
USER_NEG = Path(r"C:/Users/ibrah/Downloads/istockphoto-1248776797-612x612.jpg")
EVAL_SEED = 4242  # deliberately different from training


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-pos", type=int, default=600)
    ap.add_argument("--n-neg", type=int, default=600)
    args = ap.parse_args()

    # --- positives: fresh test-split leaves -------------------------------
    test_dirs = sorted(d for d in (PROC / "test").glob("*") if d.is_dir())
    files = [f for d in test_dirs for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    rng = random.Random(EVAL_SEED)
    rng.shuffle(files)
    pos_files = files[: args.n_pos]

    # --- negatives ---------------------------------------------------------
    neg_rng = np.random.default_rng(EVAL_SEED)
    negs: list[Image.Image] = []
    want_synth, want_bg = args.n_neg - args.n_neg // 3, args.n_neg // 3
    while len(negs) < want_synth:
        negs.extend(synth_negative(neg_rng))
    negs = negs[:want_synth]
    while len(negs) < want_synth + want_bg:
        c = background_crop(neg_rng, test_dirs)
        if c is not None:
            negs.append(c)
    if USER_NEG.exists():
        for v in augment(Image.open(USER_NEG).convert("RGB")):
            negs.append(v)

    def prob(im: Image.Image) -> float:
        p = plant_guard._trained_leaf_probability(im)
        if p is None:
            raise RuntimeError("gate model not loadable — run export first")
        return p

    pos_probs = np.asarray([prob(Image.open(f).convert("RGB")) for f in pos_files])
    neg_probs = np.asarray([prob(im) for im in negs])

    print(f"pos n={len(pos_probs)}  min={pos_probs.min():.3f}  p1={np.percentile(pos_probs, 1):.3f}  "
          f"p5={np.percentile(pos_probs, 5):.3f}  median={np.median(pos_probs):.3f}")
    print(f"neg n={len(neg_probs)}  max={neg_probs.max():.3f}  p99={np.percentile(neg_probs, 99):.3f}  "
          f"median={np.median(neg_probs):.3f}")

    sweep = {}
    best = None
    for t in (0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.5, 0.6):
        leaf_reject = float((pos_probs < t).mean())
        nonleaf_catch = float((neg_probs < t).mean())
        sweep[str(t)] = {"leaf_reject_rate": leaf_reject, "nonleaf_detection_rate": nonleaf_catch}
        print(f"  t={t:<4} leaf-reject {leaf_reject * 100:>5.1f}%   nonleaf-catch {nonleaf_catch * 100:>5.1f}%")
        # prefer: zero/low leaf rejection first, then maximal junk catching
        key = (leaf_reject, -nonleaf_catch)
        if best is None or key < best[0]:
            best = (key, t)
    rec = best[1]
    print(f"recommended GATE_THRESHOLD = {rec}")

    # End-to-end serving numbers: the real gate (strong-leaf override +
    # trained model + heuristic fallback), not just the raw classifier.
    e2e_pos = [plant_guard.looks_like_leaf(im) for im in
               (Image.open(f).convert("RGB") for f in pos_files)]
    e2e_neg = [plant_guard.looks_like_leaf(im) for im in negs]
    e2e = {
        "leaf_pass_rate": float(sum(ok for ok, _ in e2e_pos) / len(e2e_pos)),
        "junk_pass_rate": float(sum(ok for ok, _ in e2e_neg) / len(e2e_neg)),
        "note": "junk passes land on the honest low-confidence/crop-unclear path",
    }
    print(f"end-to-end serving gate: leaf-pass {e2e['leaf_pass_rate'] * 100:.1f}%  "
          f"junk-pass {e2e['junk_pass_rate'] * 100:.1f}% (-> honest low-confidence path)")

    metrics_path = REPORTS / "plant_classifier_metrics.json"
    report = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.exists() else {}
    report["serving_holdout"] = {
        "evaluated_at": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
        "positives_n": int(len(pos_probs)),
        "negatives": "synthetic (seed 4242) + test-split background crops + user photo",
        "pos_probs_p1": float(np.percentile(pos_probs, 1)),
        "pos_probs_p5": float(np.percentile(pos_probs, 5)),
        "neg_probs_p99": float(np.percentile(neg_probs, 99)),
        "threshold_sweep": sweep,
        "serving_end_to_end": e2e,
    }
    metrics_path.write_text(json.dumps(report, indent=2))
    print(f"saved -> {metrics_path.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
