#!/usr/bin/env python3
"""Generate colab_patch.py — a standalone script the user uploads to Colab.

The script contains the four (five, incl. model.py) fixed ml/src files
zlib+base64-embedded, writes them into the cloned repo inside the Colab
runtime, then self-verifies (py_compile + content assertions).

Run from the repo root:  python tools/build_colab_patch.py
"""

import base64
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ml" / "src"
OUT = ROOT / "colab_patch.py"

FILES = [
    "augmentations.py",
    "classes.py",
    "train.py",
    "export_onnx.py",
    "model.py",
]

REQUIRED = {
    "augmentations.py": [
        "tf.raw_ops.ImageProjectiveTransformV3",
        "tf.image.crop_to_bounding_box",
    ],
    "classes.py": ["CLASSES", "def validate"],
    "train.py": ["class_order.json", "build_model(len(class_names))"],
    "export_onnx.py": ["class_order.json"],
    "model.py": ["Dense(n_classes"],
}
FORBIDDEN = {
    "augmentations.py": ["tf.keras.layers.Random", "keras.layers.GaussianBlur"],
    "train.py": ["FATAL: dataset folder order differs"],
}

TEMPLATE = '''"""
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
@PAYLOAD@
}

REQUIRED = @REQUIRED@

FORBIDDEN = @FORBIDDEN@


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
        print("\\n[patch] যাচাই ব্যর্থ:")
        for p in problems:
            print("        -", p)
        return 1

    print("\\n[patch] সব ঠিক আছে — ৫টা ফাইল বদলানো ও যাচাই সম্পন্ন।")
    print("[patch] এখন cell 4a (training) আবার রান করুন।")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''


def fmt_payload() -> str:
    chunks = []
    for f in FILES:
        data = (SRC / f).read_bytes()
        blob = base64.b64encode(zlib.compress(data, 9)).decode()
        lines = []
        for i in range(0, len(blob), 76):
            lines.append('        %r' % blob[i:i + 76])
        chunks.append(f"    {f!r}: (\n" + "\n".join(lines) + "\n    ),")
    return "\n".join(chunks)





def dict_repr(d, indent=4):
    pad = " " * indent
    entries = [f"{pad}{k!r}: {v!r}," for k, v in d.items()]
    return "{\n" + "\n".join(entries) + "\n" + " " * (indent - 4) + "}"


out = (
    TEMPLATE
    .replace("@PAYLOAD@", fmt_payload())
    .replace("@REQUIRED@", dict_repr(REQUIRED))
    .replace("@FORBIDDEN@", dict_repr(FORBIDDEN))
)
OUT.write_text(out, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")
