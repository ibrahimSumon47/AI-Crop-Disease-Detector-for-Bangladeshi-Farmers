"""
Pre-inference sanity gate: refuse obvious NON-leaf photos before they reach
the ONNX model.

Why: with a 61-class model, any random photo (a sprouted potato on white,
a product shot, a flat color image) still produces a softmax answer — e.g.
"2% cauliflower insect holes" — which the UI would render as misleading
low-confidence advice. Serving an answer to a photo that contains no plant
material is worse than refusing.

What it checks (calibrated 2026-10-22 on 160 sampled images from the real
`data/processed/test` split + non-leaf probes):

1. **Flat/blank images** — grayscale std < 3.0 means there is no photo detail
   at all (solid color, an empty upload mixed with padding). Real phone
   photos, even blurry ones, always exceed this.
2. **Plain-background non-plant objects** — the "stock photo of a tuber on
   white" signature: very little green plant material AND an overwhelming
   plain white/gray background. Measured on the real split:
   every tested leaf image has green fraction >= 0.09 *or* background
   fraction < 0.09 (PlantVillage-style dark backgrounds), while product-style
   non-leaf photos sit at green ≈ 0.02 / white ≈ 0.64+.

Deliberately NOT rejected (documented in README limitations): humans/skin,
documents, collages — these pass the gate and rely on the honest low
confidence path; leaf material in less-than-ideal light still passes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image

# --- thresholds (see module docstring for the calibration data) -------------
FLAT_STD_MIN = 3.0        # grayscale std below this = no photographic detail
GREEN_MIN = 0.08          # fraction of vegetation-green pixels (ExG test)
WHITE_MAX = 0.50          # plain-background fraction allowed alongside low green

REJECT_FLAT = (
    "ছবিতে কোনো দৃশ্য নেই (একদম ফাঁকা/একরঙা)। সন্দেহভাজন পাতার একটি স্পষ্ট ছবি "
    "তুলে আবার চেষ্টা করুন।"
)
REJECT_NO_LEAF = (
    "ছবিতে গাছের পাতা ধরা পড়েনি। সন্দেহভাজন পাতাটি কাছ থেকে, দিন-আলোয় তুলুন "
    "(ফসলের ফল/কন্দ, প্লাস্টিক প্যাকেট বা অন্য কোনো বস্তু নয়) — তারপর আবার "
    "স্ক্যান করুন।"
)


@dataclass
class PlantStats:
    """Raw measurements, kept for logging/tests rather than user display."""

    green_frac: float
    white_frac: float
    gray_std: float


def _features(img: Image.Image) -> PlantStats:
    """Compute the color/texture statistics the rule needs."""
    small = img.convert("RGB")
    small.thumbnail((256, 256))  # stats are scale-free; saves work
    arr = np.asarray(small, dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Excess-green vegetation index — the standard remote-sensing cue.
    exg = 2.0 * g - r - b
    green_frac = float(((exg > 25.0) & (g > 60.0)).mean())

    mx = arr.max(-1)
    mn = arr.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    val = mx / 255.0
    white_frac = float(((sat < 0.12) & (val > 0.75)).mean())

    gray = arr.mean(-1)
    gray_std = float(gray.std())
    return PlantStats(green_frac, white_frac, gray_std)


def looks_like_leaf(img: Image.Image) -> tuple[bool, str]:
    """Return (ok, message). message is empty when ok, else Bangla guidance."""
    stats = _features(img)

    if stats.gray_std < FLAT_STD_MIN:
        return False, REJECT_FLAT

    # Object-on-plain-background signature (sprouted potato / product shots).
    if stats.green_frac < GREEN_MIN and stats.white_frac > WHITE_MAX:
        return False, REJECT_NO_LEAF

    return True, ""
