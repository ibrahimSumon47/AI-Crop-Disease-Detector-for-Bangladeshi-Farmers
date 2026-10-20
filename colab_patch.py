"""
colab_patch.py — Colab-এ ঠিক করা ml/src কোড বসানোর এক ফাইলের প্যাচ।

কেন: GitHub-এ এখনো পুরোনো (ভাঙা) কোড আছে, তাই `git clone` করলে ভাঙা
কোডই আসে। এই স্ক্রিপ্ট সেই ফোল্ডারে গিয়ে ৫টা ফাইল ঠিক করা ভার্সন দিয়ে
বদলায় — Colab সেশন বা ডাউনলোড করা ডেটা মুছবে না।

Colab-এ চালানোর নিয়ম (২ ধাপ):
    from google.colab import files
    files.upload()               # এই ফাইলটা বেছে নিন
    !python colab_patch.py       # এটাই মূল কাজ করবে
"""

import base64
import py_compile
import sys
import tempfile
import zlib
from pathlib import Path

FOLDER_HINTS = ["AI-Crop-Disease-Detector-for-Bangladeshi-Farmers"]

_PAYLOAD = {
    'augmentations.py': (
        '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'
        '/C9/fWxU'
    ),
    'classes.py': (
        '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'
        'aGDVgGv9H0FrLQ8='
    ),
    'train.py': (
        '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'
        'R9/nUaLv00zJ9x1T/ZsB0+C/BzoGpw=='
    ),
    'export_onnx.py': (
        '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'
        'vCiiFBRF3thhg/LR4P+GRbRI'
    ),
    'model.py': (
        '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'
        'L1Z2Fok='
    ),
}

REQUIRED = {
    'augmentations.py': ['tf.raw_ops.ImageProjectiveTransformV3', 'tf.image.crop_to_bounding_box'],
    'classes.py': ['CLASSES', 'def validate'],
    'train.py': ['class_order.json', 'build_model(len(class_names))'],
    'export_onnx.py': ['class_order.json'],
    'model.py': ['Dense(n_classes'],
}

FORBIDDEN = {
    'augmentations.py': ['tf.keras.layers.Random', 'keras.layers.GaussianBlur'],
    'train.py': ['FATAL: dataset folder order differs'],
}


def find_repo() -> Path:
    """রিপো ফোল্ডার খুঁজি — নামের শেষে -1, -2 থাকলেও ধরবে।"""
    roots = [Path("/content"), Path.cwd()]
    candidates = []
    for root in roots:
        for hint in FOLDER_HINTS:
            candidates.append(root / hint)
        if root.is_dir():
            candidates += sorted(p.parents[2] for p in root.glob("*/ml/src/train.py"))
        candidates += [root, root.parent]
    for c in candidates:
        try:
            if (c / "ml" / "src" / "train.py").is_file():
                return c.resolve()
        except OSError:
            continue
    raise SystemExit(
        "[patch] রিপো ফোল্ডার পাওয়া যায়নি। আগে cell 1a (git clone) রান করুন।"
    )


def main() -> int:
    repo = find_repo()
    src = repo / "ml" / "src"
    print(f"[patch] repo : {repo}")

    written = []
    for name, blob in _PAYLOAD.items():
        text = zlib.decompress(base64.b64decode(blob)).decode("utf-8")
        target = src / name
        target.write_text(text, encoding="utf-8")
        written.append((name, target, text))

    print("[patch] ফাইল লেখা হয়েছে:")
    for name, target, text in written:
        print(f"        {name:<18} {len(text.splitlines()):>4} lines")

    problems = []
    for name, target, _ in written:
        try:
            py_compile.compile(str(target), doraise=True, cfile=tempfile.mktemp())
        except py_compile.PyCompileError as exc:
            problems.append(f"{name}: syntax error -> {exc.msg}")
    for name, _, text in written:
        for needle in REQUIRED.get(name, []):
            if needle not in text:
                problems.append(f"{name}: expected code missing -> {needle!r}")
        for needle in FORBIDDEN.get(name, []):
            if needle in text:
                problems.append(f"{name}: old buggy code still present -> {needle!r}")

    if problems:
        print("\n[patch] যাচাই ব্যর্থ:")
        for p in problems:
            print("        -", p)
        return 1

    print("\n[patch] সব ঠিক আছে — ৫টা ফাইল বদলানো ও যাচাই সম্পন্ন।")
    print("[patch] এখন cell 4a (training) আবার রান করুন।")
    return 0


if __name__ == "__main__":
    sys.exit(main())
