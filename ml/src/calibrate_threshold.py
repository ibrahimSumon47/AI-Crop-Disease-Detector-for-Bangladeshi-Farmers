"""
calibrate_threshold.py — pick the confidence threshold for /predict.

WHY
    A softmax probability of 0.51 is NOT a trustworthy answer for a farmer.
    The backend refuses to diagnose when the top probability is below a
    calibrated threshold and instead returns "ইস_uncertain: consult an
    agriculture officer". This script finds that threshold on VALIDATION
    data and writes it to ml/reports/threshold.json.

HOW IT CHOOSES
    Sweep thresholds from 0.30 to 0.98. For each candidate threshold:
      * a val image is "covered" when max(probs) >= threshold
      * precision is computed ONLY over covered images
    We pick the LOWEST threshold that still keeps precision >= TARGET_PRECISION
    (default 0.90). Lower threshold => more images get an answer, but each
    must stay accurate. This is a coverage/precision lockstep, tuned for
    where a wrong diagnosis costs the farmer a season.

OUTPUT
    ml/reports/threshold.json — {"threshold": 0.xx, "target_precision": ...,
    "precision_at_threshold": ..., "coverage": ...} — pasted into the
    backend's CONFIDENCE_THRESHOLD env var by the user.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import precision_score

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "ml" / "reports"
DEFAULT_TARGET_PRECISION = 0.90
THRESHOLD_GRID = np.arange(0.30, 0.98, 0.01)  # candidate thresholds


def get_val_predictions(model_path: Path, val_dir: Path, batch_size: int = 32):
    """Reuse evaluate.py's loading + prediction over the validation split."""
    # Importing from our sibling module keeps preprocessing EXACTLY the same
    # as the metrics run (single source of truth for [-1,1] scaling etc.).
    import evaluate as _ev

    model, ds, class_names = _ev.load_model_and_data(model_path, val_dir, batch_size)
    y_true, y_pred, probs_max = _ev.predict_all(model, ds)
    return y_true, y_pred, probs_max, class_names


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--val", type=Path, default=REPO_ROOT / "data" / "processed" / "val")
    parser.add_argument("--out", type=Path, default=DEFAULT_REPORT_DIR / "threshold.json")
    parser.add_argument("--target-precision", type=float, default=DEFAULT_TARGET_PRECISION)
    args = parser.parse_args()

    y_true, y_pred, probs_max, _ = get_val_predictions(args.model, args.val)

    correct = (y_pred == y_true).astype(int)  # 1 when the answer was right

    best: dict | None = None
    sweep = []
    for t in THRESHOLD_GRID:
        covered = probs_max >= t
        n_covered = int(covered.sum())
        if n_covered == 0:
            sweep.append({"threshold": round(float(t), 2), "n_covered": 0,
                          "precision": None, "coverage": 0.0})
            continue
        precision = float(correct[covered].mean())
        row = {
            "threshold": round(float(t), 2),
            "n_covered": n_covered,
            "precision": precision,
            "coverage": round(n_covered / len(correct), 3),
        }
        sweep.append(row)
        # Keep the LOWEST threshold that still meets the target precision.
        if precision >= args.target_precision:
            best = row

    # If no threshold meets the target, report the highest-precision one and
    # flag it — the user will need to decide (retrain vs lower the bar).
    if best is None:
        scored = [r for r in sweep if r["precision"] is not None]
        best = max(scored, key=lambda r: r["precision"]) if scored else sweep[-1]
        note = ("WARNING: no threshold reached the target precision; returned the "
                "best-effort threshold — consider retraining before production.")
    else:
        note = (f"Lowest threshold on the 0.01 grid that holds precision >= "
                f"{args.target_precision:.2f} on the validation split.")

    payload = {
        "threshold": best["threshold"],
        "target_precision": args.target_precision,
        "precision_at_threshold": best["precision"],
        "coverage": best["coverage"],
        "note": note,
        "sweep": sweep,
        "model": str(args.model),
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"[calibrate] threshold={best['threshold']} precision={best['precision']} coverage={best['coverage']}")
    print(f"[calibrate] wrote {args.out}")
    print("[calibrate] copy this value into backend CONFIDENCE_THRESHOLD after training succeeds.")


if __name__ == "__main__":
    main()
