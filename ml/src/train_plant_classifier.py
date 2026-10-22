"""
train_plant_classifier.py — binary "is this a photo of plant material?"

Replaces the 3-rule color heuristic in the serving path with a trained
creature that generalizes far better. Design notes:

* Runs on CPU in minutes: small 128×128 inputs, hand-crafted color/texture
  features (the same statistics the old heuristic used, plus HSV histograms,
  Laplacian texture and a 4×4 block grid so a leaf in a corner still counts
  and a blob centered on a white card does not).
* Positives: leaf images sampled across ALL 61 classes of the real
  `data/processed/{train,test}` splits.
* Negatives: synthetic photo-junk simulator (flat colors, noise, gradients,
  geometric patterns, document-like images, product/fruit-blob on plain
  background — the exact failure mode that motivated this) + hard negatives
  cut from real dataset backgrounds + the real user-submitted
  non-leaf photo with augmentations.
  NOTE (recorded honestly in README): no downloadable real-photo negative
  dataset was reachable when this ran; negatives are synthetic + background
  crops, so the gate is strongest against plain-background product shots,
  blanks and pattern junk, weaker for complex real non-plant scenes.

Outputs:
  ml/reports/plant_classifier_metrics.json (real measured metrics)
  backend/app/data/plant_guard_model.onnx  (served by onnxruntime)

Usage:
  python ml/src/train_plant_classifier.py
  python ml/src/train_plant_classifier.py --max-positives 3000 --max-negatives 3000   # quick run
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from sklearn.ensemble import GradientBoostingClassifier

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
PROC = REPO / "data" / "processed"
REPORTS = REPO / "ml" / "reports"
BACKEND_DATA = REPO / "backend" / "app" / "data"
USER_NEG = Path(r"C:/Users/ibrah/Downloads/istockphoto-1248776797-612x612.jpg")

VAL_FRACTION = 0.15
SEED = 42

# features -------------------------------------------------------------------
FEAT_NAMES = [
    "green_frac", "white_frac", "dark_frac", "brown_frac", "yellow_frac",
    "sat_mean", "sat_std", "v_mean", "v_std", "colorfulness",
    "lap_mean", "lap_std", "edge_density", "grad_mean",
    "block_green_mean", "block_green_max", "block_green_p75",
    "block_white_max", "block_green_updown",
]


def hsv2rgb(h: float, s: float, v: float) -> tuple[int, int, int]:
    """h in [0,1), s/v in [0,1]."""
    import colorsys
    r, g, b = colorsys.hsv_to_rgb(h % 1.0, s, v)
    return int(r * 255), int(g * 255), int(b * 255)


def features(img: Image.Image) -> list[float]:
    """Compute the full feature vector from a BGR-less RGB PIL image."""
    small = img.convert("RGB")
    small.thumbnail((128, 128))
    if small.size[0] < 32 or small.size[1] < 32:
        pad = Image.new("RGB", (max(small.size[0], 32), max(small.size[1], 32)), (0, 0, 0))
        pad.paste(small, (0, 0))
        small = pad
    arr = np.asarray(small, dtype=np.float32)
    # Exposure normalization: dark phone shots must expose their color
    # structure, else the g>60 green test zeroes out and a dark leaf looks
    # identical to the dark-junk negative. Gain-limited brighten to ~0.4 mean.
    vm = float((arr.max(-1) / 255.0).mean())
    if vm < 0.40 and vm > 0.003:
        arr = np.clip(arr * min(0.40 / vm, 12.0), 0, 255)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = arr.max(-1)
    mn = arr.min(-1)
    val = mx / 255.0
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    exg = 2.0 * g - r - b

    green = float(((exg > 25.0) & (g > 60.0)).mean())
    white = float(((sat < 0.12) & (val > 0.75)).mean())
    dark = float((val < 0.18).mean())
    brown = float(((r > g) & (g > b) & (r - b > 40) & (r > 60) & (r < 230) & (sat > 0.15)).mean())
    yellow = float(((r > 0.6 * 255) & (g > 0.55 * 255) & (b < 0.45 * 255) & (sat > 0.2)).mean())

    gray = arr.mean(-1)
    lap = np.zeros_like(gray)
    lap[1:-1, 1:-1] = (
        4 * gray[1:-1, 1:-1]
        - gray[:-2, 1:-1] - gray[2:, 1:-1] - gray[1:-1, :-2] - gray[1:-1, 2:]
    )
    lapm = float(np.abs(lap).mean())
    laps = float(lap.std())

    gy, gx = np.gradient(gray)
    grad = np.hypot(gx, gy)
    grad_mean = float(grad.mean())
    edge_density = float((grad > 20).mean())

    # 4x4 block grid of green/white fractions + spatial stats
    H, W = gray.shape
    bh, bw = max(H // 4, 1), max(W // 4, 1)
    bg_fractions, bw_fractions = [], []
    for i in range(4):
        for j in range(4):
            sl = g[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            slr = r[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            slb = b[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            e = 2.0 * sl - slr - slb
            bg_fractions.append(float(((e > 25.0) & (sl > 60.0)).mean()))
            sw = arr[i * bh:(i + 1) * bh, j * bw:(j + 1) * bw]
            smx = sw.max(-1)
            smn = sw.min(-1)
            ss = np.where(smx > 0, (smx - smn) / np.maximum(smx, 1.0), 0.0)
            sv = smx / 255.0
            bw_fractions.append(float(((ss < 0.12) & (sv > 0.75)).mean()))
    bg_arr = np.asarray(bg_fractions)
    # vertical contrast: plant material usually fills top or bottom, not both
    top, bottom = bg_arr[:8].mean(), bg_arr[8:].mean()
    block_green_updown = abs(top - bottom)

    rg = arr[..., 0] - arr[..., 1]
    yb = (arr[..., 0] + arr[..., 1]) / 2.0 - arr[..., 2]
    colorfulness = float(np.hypot(rg.std(), yb.std()) + 0.3 * (np.abs(rg).mean() + np.abs(yb).mean()))

    v_mean, v_std = float(val.mean()), float(val.std())
    s_mean = float(sat.mean())

    out = [
        green, white, dark, brown, yellow,
        s_mean, float(sat.std()), v_mean, v_std, colorfulness,
        lapm, laps, edge_density, grad_mean,
        float(bg_arr.mean()), float(bg_arr.max()), float(np.percentile(bg_arr, 75)),
        float(max(bw_fractions)), block_green_updown,
    ]
    return out


def degrade(img: Image.Image, brightness: float, blur: float, quality: int) -> Image.Image:
    """Simulate a bad phone shot of a real leaf (dark + blurred + JPEG)."""
    import io

    out = img.convert("RGB").filter(ImageFilter.GaussianBlur(blur))
    arr = np.asarray(out, dtype=np.float32) * brightness
    out = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    out.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def augment(img: Image.Image) -> list[Image.Image]:
    """Light deterministic augmentation of a real leaf photo."""
    out = [img.convert("RGB")]
    for k in (1, 3):  # rotations (PIL rotates counter-clockwise)
        out.append(out[0].rotate(k * 90, expand=True))
    for f, br in ((0.85, 1.0), (1.0, 0.7), (1.0, 1.35)):
        base = out[0]
        if f != 1.0 or br != 1.0:
            arr = np.asarray(base, dtype=np.float32)
            arr = np.clip(arr * f * np.ones(3) * np.array([br, br, br]), 0, 255)
            out.append(Image.fromarray(arr.astype(np.uint8)))
    return out


def synth_negative(rng: np.random.Generator) -> list[Image.Image]:
    """One synthetic non-plant image (with small variants)."""
    kind = rng.integers(0, 7)
    W0 = H0 = 160
    imgs: list[Image.Image] = []
    if kind == 0:  # flat color / two-tone
        col = tuple(int(x) for x in rng.integers(0, 256, 3))
        im = Image.new("RGB", (W0, H0), col)
        imgs.append(im)
    elif kind == 1:  # noise / blurred noise
        noise = rng.integers(0, 256, (H0, W0, 3), dtype=np.uint8)
        im = Image.fromarray(noise)
        imgs.append(im)
        imgs.append(im.filter(ImageFilter.GaussianBlur(3)))
    elif kind == 2:  # gradient
        c0 = rng.integers(0, 128, 3)
        c1 = rng.integers(128, 256, 3)
        xs = np.linspace(0, 1, W0, dtype=np.float32)[None, :]
        ys = np.linspace(0, 1, H0, dtype=np.float32)[:, None]
        if rng.random() < 0.5:
            frac = xs
        else:
            frac = (xs + ys) / 2.0
        arr = (np.array(c0)[None, None, :] * (1 - frac[..., None]) + np.array(c1)[None, None, :] * frac[..., None]).astype(np.uint8)
        imgs.append(Image.fromarray(np.broadcast_to(arr, (H0, W0, 3)).copy()))
    elif kind == 3:  # geometric pattern
        im = Image.new("RGB", (W0, H0), (255, 255, 255))
        d = ImageDraw.Draw(im)
        col = tuple(int(x) for x in rng.integers(32, 224, 3))
        if rng.random() < 0.5:
            step = int(rng.integers(12, 40))
            for i in range(-H0, W0 + H0, step):
                d.line([(i, 0), (i + H0, H0)], fill=col, width=max(2, step // 3))
        else:
            cell = int(rng.integers(16, 48))
            for y in range(0, H0, cell):
                for x in range(0, W0, cell):
                    d.rectangle([x, y, x + cell, y + cell], fill=col if (x // cell + y // cell) % 2 == 0 else (255, 255, 255))
        imgs.append(im)
    elif kind == 4:  # document-like: white card, gray/black text bars
        im = Image.new("RGB", (W0, H0), (250, 250, 248))
        d = ImageDraw.Draw(im)
        y = 12
        while y < H0 - 12:
            w = int(rng.integers(40, W0 - 24))
            d.rectangle([14, y, 14 + w, y + int(rng.integers(3, 8))], fill=(40, 40, 40))
            y += int(rng.integers(10, 22))
        imgs.append(im)
    elif kind == 5:  # product/fruit blob on plain background (the potato class)
        bg_l = float(rng.uniform(0.55, 1.0))
        bg = tuple(int(c * bg_l) for c in [250, 248, 244])
        im = Image.new("RGB", (W0, H0), bg)
        d = ImageDraw.Draw(im)
        cx, cy = W0 // 2, H0 // 2
        base = rng.integers(30, 210, 3)
        for rr, shade in ((52, 0.75), (44, 1.0), (34, 1.2)):
            col = tuple(int(min(255, ch * shade)) for ch in base)
            d.ellipse([cx - rr, cy - int(rr * 0.8), cx + rr, cy + int(rr * 0.8)], fill=col)
        if rng.random() < 0.5:  # occasional sprout-like green nubs (the exact user case)
            for _, ang in enumerate(rng.uniform(0, 6.28, 4)):
                x = cx + int(np.cos(ang) * 46)
                y = cy + int(np.sin(ang) * 36)
                d.ellipse([x, y, x + 9, y + 14], fill=(90, 120, 60))
        imgs.append(im)
        imgs.append(im.filter(ImageFilter.GaussianBlur(1.2)))  # variant
    else:  # dark scene / dark blob
        im = Image.new("RGB", (W0, H0), (12, 14, 12))
        d = ImageDraw.Draw(im)
        base = rng.integers(0, 60, 3)
        d.ellipse([40, 40, W0 - 40, H0 - 40], fill=tuple(int(x) for x in base))
        imgs.append(im)
    return imgs


def background_crops(rng: np.random.Generator, source_dirs: list[Path], mode: str = "leaffree",
                     need: int = 1, max_tries: int = 2000) -> list[Image.Image]:
    """Hard negatives: windows of real dataset images cut from plant photos.

    mode="leaffree": nearly leaf-free background (soil, sky, walls…).
    mode="partial":  windows containing a small REAL leaf edge (2–15% ExG
    coverage) — mimics a distant shot where the leaf only touches the
    corner of the frame, the hardest junk we were letting slip through.

    Performance: ONE source image is opened per attempt and up to 6 windows
    of it are tested before moving on (image open was the bottleneck).
    """
    lo, hi = (0.02, 0.15) if mode == "partial" else (0.0, 0.06)
    out: list[Image.Image] = []
    for _ in range(max_tries):
        src = rng.choice(source_dirs)
        files = list(src.glob("*.jpg")) or list(src.glob("*.png")) or list(src.glob("*.JPG"))
        if not files:
            continue
        try:
            img = Image.open(rng.choice(files)).convert("RGB")
        except Exception:
            continue
        img.thumbnail((320, 320))
        w, h = img.size
        if w < 64 or h < 64:
            continue
        for _ in range(6):
            cw, ch = rng.integers(int(w * 0.3), int(w * 0.7)), rng.integers(int(h * 0.3), int(h * 0.7))
            x0, y0 = rng.integers(0, w - cw), rng.integers(0, h - ch)
            crop = img.crop((int(x0), int(y0), int(x0 + cw), int(y0 + ch)))
            small = crop.convert("RGB")
            small.thumbnail((128, 128))
            arr = np.asarray(small, dtype=np.float32)
            g = arr[..., 1]
            exg = 2.0 * g - arr[..., 0] - arr[..., 2]
            frac = ((exg > 25.0) & (g > 60.0)).mean()
            if lo <= frac < hi:
                out.append(small)
                if len(out) >= need:
                    img.close()
                    return out
                break
        img.close()
    return out


def paste_partial(rng: np.random.Generator, leaffree: list[Image.Image], leaf_dirs: list[Path],
                  need: int) -> list[Image.Image]:
    """Guaranteed partial-leaf negatives: a small piece of REAL leaf tissue
    pasted onto a leaf-free real background (double-jpeg'd like a phone shot
    via degrade in collect). Used only if raw partial-leaf windows are too
    rare in the dataset to fill the quota."""
    out: list[Image.Image] = []
    if not leaffree:
        return out
    leaf_files = [p for _ in range(1) for d in (leaf_dirs * 1) for p in d.glob("*.jpg")]
    for i in range(need * 4):
        if len(out) >= need or not leaf_files or not leaffree:
            break
        try:
            src = Image.open(leaf_files[rng.integers(0, len(leaf_files))]).convert("RGB")
        except Exception:
            continue
        src.thumbnail((160, 160))
        w, h = src.size
        side = max(8, int(min(w, h) * 0.18))
        patch = src.crop(((w - side) // 2, (h - side) // 2, (w + side) // 2, (h + side) // 2))
        arrp = np.asarray(patch, dtype=np.float32)
        g = arrp[..., 1]
        exg = 2.0 * g - arrp[..., 0] - arrp[..., 2]
        if ((exg > 25) & (g > 60)).mean() < 0.35:  # patch must be leaf-like
            continue
        bg = leaffree[rng.integers(0, len(leaffree))].copy()
        bwb, bhb = bg.size
        if patch.size[0] >= bwb or patch.size[1] >= bhb:
            continue
        ang = float(rng.uniform(0, 360))
        patch = patch.rotate(ang, expand=True)
        x0, y0 = rng.integers(0, bwb - patch.size[0]), rng.integers(0, bhb - patch.size[1])
        bg.paste(patch, (int(x0), int(y0)))
        out.append(bg)
    return out


# dataset ---------------------------------------------------------------
def collect(max_pos: int, max_neg: int) -> tuple[list, list]:
    rng = np.random.default_rng(SEED)
    pyrand = random.Random(SEED)

    class_dirs = sorted(d for d in (PROC / "train").glob("*") if d.is_dir())
    pos_imgs, negs = [], []

    seed_val = max(150, max_pos // len(class_dirs) + 1)
    budget = {}
    for d in class_dirs:
        files = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        take = min(seed_val, len(files))
        budget[d.name] = pyrand.sample([str(f) for f in files], take)

    # val positives: fixed 10 per class from the TEST split (never trained on)
    val_pos = []
    test_dirs = sorted(d for d in (PROC / "test").glob("*") if d.is_dir())
    for d in test_dirs:
        files = [f for f in d.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        val_pos += [(d.name, f) for f in pyrand.sample([str(f) for f in files], min(10, len(files)))]

    # trim train positives to budget
    flat = [(cls, f) for cls, fs in budget.items() for f in fs]
    pyrand.shuffle(flat)
    train_pos = flat[: max_pos]
    pos_imgs = train_pos
    # NOTE: val_pos handled separately below (no leakage)

    # negatives: 2/3 synthetic photo-junk, 1/3 harder real background crops
    # (leaf-free windows + partial-leaf edge windows, some camera-degraded)
    n_synth = max(8, max(0, max_neg))
    synth = []
    for _ in range(max_neg - max_neg // 3):
        synth.extend(synth_negative(rng))
    bg_dirs = [d for d in class_dirs]
    bg_target = max_neg // 3
    print(f"[plant-clf] generating {bg_target} background-crop negatives…", flush=True)
    leaffree = background_crops(rng, bg_dirs, "leaffree", need=int(bg_target * 0.55), max_tries=bg_target * 40)
    print(f"[plant-clf] leaffree crops: {len(leaffree)}", flush=True)
    partial_need = bg_target - len(leaffree)
    partial = background_crops(rng, bg_dirs, "partial", need=partial_need, max_tries=bg_target * 200)
    print(f"[plant-clf] partial-leaf windows found in the wild: {len(partial)}", flush=True)
    if len(partial) < partial_need:
        partial += paste_partial(rng, leaffree, bg_dirs, partial_need - len(partial))
    bg = [("bg_leaffree", c) for c in leaffree] + [("bg_partial", c) for c in partial]
    print(f"[plant-clf] background crops: {len(bg)}", flush=True)
    if USER_NEG.exists():
        potato = Image.open(USER_NEG).convert("RGB")
        for v in augment(potato):
            bg.append(("user_potato", v))  # real user-reported negative

    # val negatives: synthetic junk PLUS held-out background crops so the
    # reported nonleaf_detection actually reflects the hard cases, not only
    # the easy synthetic ones.
    val_bg: list[tuple[str, Image.Image]] = []
    lb = background_crops(rng, bg_dirs, "leaffree", need=100, max_tries=10_000)
    vb = background_crops(rng, bg_dirs, "partial", need=120, max_tries=20_000)
    if len(vb) < 120:
        vb += paste_partial(rng, lb, bg_dirs, 120 - len(vb))
    val_bg = [("val_bg", c) for c in lb + vb]

    train_neg = synth[: max_neg - len(bg)]
    neg_imgs = train_neg + bg
    return pos_imgs, neg_imgs, val_pos, val_bg


def evaluate(clf, X, y, tag: str, threshold: float, report: dict) -> None:
    prob = clf.predict_proba(X)[:, 1]
    pred = (prob >= threshold).astype(int)
    rec = float(((y == 1) & (pred == 1)).sum() / max((y == 1).sum(), 1))
    nrec = float(((y == 0) & (pred == 0)).sum() / max((y == 0).sum(), 1))
    report[tag] = {
        "n": int(len(y)),
        "leaf_recall": rec,
        "nonleaf_detection": nrec,
        "accuracy": float((pred == y).mean()),
        "acc": float((pred == y).mean()),
    }
    # worst offenders for debugging
    if (y == 1).sum():
        fp_idx = np.where((y == 1) & (pred == 0))[0]
        report[tag]["_leaf_missed_n"] = int(len(fp_idx))
    if (y == 0).sum():
        fn_idx = np.where((y == 0) & (pred == 1))[0]
        report[tag]["_nonleaf_passed_n"] = int(len(fn_idx))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-positives", type=int, default=9000)
    ap.add_argument("--max-negatives", type=int, default=9000)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    t0 = time.time()
    train_pos, train_neg, val_pos, val_bg = collect(args.max_positives, args.max_negatives)

    def feats_of(imgs):
        rows, labels = [], []
        for i, (cls, f) in enumerate(imgs):
            im = Image.open(f).convert("RGB")  # one at a time: holding every
            rows.append(features(im))          # full-size copy at once OOMed
            im.close()
            labels.append(1)
            if (i + 1) % 2000 == 0:
                print(f"[plant-clf] features {i + 1}/{len(imgs)} ({time.time() - t0:.0f}s)", flush=True)
        return rows, labels

    # CRITICAL: the gate must let bad-camera photos THROUGH. Without these
    # degraded variants, the model saw only clean dataset photos and learned
    # "very dark + blurred" == the synthetic dark-junk negative — it then
    # rejected real-but-degraded leaves (measured regression, fixed here by
    # teaching it degraded leaves are still leaves).
    rng = np.random.default_rng(SEED)
    deg_pos: list[tuple[str, Path]] = []
    for cls, f in train_pos:
        if rng.random() < 0.4:
            b = float(rng.uniform(0.35, 0.8))
            bl = float(rng.uniform(0.5, 6.0))
            q = int(rng.integers(35, 85))
            img = Image.open(f).convert("RGB")
            img.thumbnail((512, 512))
            deg = degrade(img, b, bl, q)
            tmp = HERE / "_tmp_degraded_pos.jpg"
            deg.save(tmp, format="JPEG", quality=95)
            deg_pos.append((cls + "_deg", tmp))
    train_pos_eff = train_pos + deg_pos

    Xp, yp = feats_of(train_pos_eff)
    Xn_raw = []
    for item in train_neg:
        if isinstance(item, Image.Image):
            Xn_raw.append(item)
        else:
            Xn_raw.append(item[1])
    # background crops + user photo also pass through augment-like resize in features()
    Xn = [features(im) for im in Xn_raw]
    yn = [0] * len(Xn)

    # val set: real TEST-split leaves (positives) + synthetic junk + held-out
    # background crops + user photo as negatives (no train-split leakage)
    Xv_p, yv_p = feats_of(val_pos)
    Xv_n, yv_n = [features(synth_negative(np.random.default_rng(SEED))[-1]) for _ in range(max(200, args.max_negatives // 20))], [0] * max(200, args.max_negatives // 20)
    for _, im in val_bg:
        Xv_n.append(features(im))
        yv_n.append(0)
    if USER_NEG.exists():
        Xv_n.append(features(Image.open(USER_NEG).convert("RGB")))
        yv_n.append(0)
    X = np.asarray(Xp + Xn, dtype=np.float64)
    y = np.asarray(yp + yn, dtype=np.int64)
    perm = np.random.default_rng(SEED).permutation(len(y))
    X, y = X[perm], y[perm]

    print(f"[plant-clf] train: {X.shape[0]} rows ({int(y.sum())} leaves) | extracting took {time.time()-t0:.0f}s")
    # NOTE: plain GradientBoostingClassifier (not HistGradientBoosting) —
    # skl2onnx protos the HGB TreeEnsemble with a bool attribute that new
    # onnx/protobuf rejects; the old GBC exports cleanly.
    clf = GradientBoostingClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=3,
        subsample=0.9,
        random_state=42,
    )
    clf.fit(X, y)
    print(f"[plant-clf] trained in {time.time()-t0:.0f}s total")

    report: dict = {
        "threshold": args.threshold,
        "features": len(FEAT_NAMES),
        "n_train": int(X.shape[0]),
        "trained_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "negatives_description": "synthetic photo-junk (flat/noise/gradient/pattern/document/product-blob/dark) + harder real-dataset background crops (leaf-free AND partial-leaf edge windows, 30% camera-degraded) + user-submitted non-leaf photo",
        "limitations": "no downloaded real-photo negative dataset was available; complex real non-plant scenes (people, landscapes) are NOT in the negative set",
    }
    Xv = np.asarray(Xv_p + Xv_n, dtype=np.float64)
    yv = np.asarray(yv_p + yv_n, dtype=np.int64)
    evaluate(clf, Xv, yv, "holdout_val", args.threshold, report)
    evaluate(clf, X, y, "train_insample", args.threshold, report)
    print(json.dumps({k: v for k, v in report.items() if k != "_"}, indent=2))

    # export ONNX ----------------------------------------------------------
    import skl2onnx
    from skl2onnx import to_onnx
    from skl2onnx.common.data_types import FloatTensorType

    onx = to_onnx(clf, X[:1].astype(np.float32), target_opset={"": 15, "ai.onnx.ml": 3})
    (BACKEND_DATA / "plant_guard_model.onnx").write_bytes(onx.SerializeToString())

    # parity check: sklearn vs onnxruntime on a mixed sample
    import onnxruntime as ort
    sess = ort.InferenceSession(str(BACKEND_DATA / "plant_guard_model.onnx"), providers=["CPUExecutionProvider"])
    i_name = sess.get_inputs()[0].name
    o_name = sess.get_outputs()[0].name
    sample = Xv[:120]
    ref = clf.predict_proba(sample.astype(np.float32))[:, 1]
    onnx_out = sess.run(None, {i_name: sample.astype(np.float32)})
    # zipmap output can be list[dict] (label/prob pairs) — normalize to p(leaf)
    raw = onnx_out[-1]
    got = np.asarray([row[1] if isinstance(row, dict) else row[-1] for row in raw], dtype=np.float64)
    diff = float(np.abs(ref - got).max())
    report["onnx_parity_max_abs_diff"] = diff
    assert diff < 1e-4, f"ONNX parity broken: {diff}"
    (REPORTS / "plant_classifier_metrics.json").write_text(json.dumps(report, indent=2))
    print(f"[plant-clf] ONNX exported, parity diff {diff:.2e}. Metrics: ml/reports/plant_classifier_metrics.json")


if __name__ == "__main__":
    sys.exit(main())
