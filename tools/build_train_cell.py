#!/usr/bin/env python3
"""Emit a train.py-only repair cell (cell_train_fix.txt) + verify it end-to-end.

Why train.py only: the user's paste of the combined cell half-decoded
(augmentations.py landed fine — 277 lines confirmed — but the train.py blob
arrived corrupted, zlib raised 'incorrect data check'). A smaller blob is far
less likely to get mangled by the chat/copy path, and the cell now checks a
sha256 of the received base64 and explains in Bangla what a mismatch means.

Verification:
  1) run the exact cell text against a simulated clone built from the OLD
     commit (143b37b) -> exit 0, train.py byte-identical to the fixed version
  2) corrupt one character of the blob -> friendly Bangla message, no crash
"""

import base64
import hashlib
import io
import subprocess
import tarfile
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ml" / "src"
OUT = ROOT / "cell_train_fix.txt"
TARGET = "train.py"

blob = base64.b64encode(zlib.compress((SRC / TARGET).read_bytes(), 9)).decode()
sha = hashlib.sha256(blob.encode()).hexdigest()[:16]  # hash of the b64 TEXT itself
size_kb = len(blob) / 1024

lines = [f'    "{TARGET}": (']
for i in range(0, len(blob), 72):
    lines.append(f'        "{blob[i:i+72]}"')
lines.append("    ),")
payload = "\n".join(lines)

TEMPLATE = '''import glob, pathlib, py_compile, base64, zlib, hashlib

candidates = []
for root in [pathlib.Path("/content"), pathlib.Path.cwd(), pathlib.Path.cwd().parent]:
    candidates += sorted(glob.glob(str(root / "*AI-Crop-Disease-Detector*/ml/src")))
    if (root / "ml" / "src" / "train.py").is_file():
        candidates.append(str(root / "ml" / "src"))
assert candidates, "রিপো ফোল্ডার পাওয়া যায়নি — Colab-এ cell 1a (git clone) রান হয়েছে তো?"
SRC = pathlib.Path(candidates[0])
print("লিখছি ফোল্ডারে:", SRC)

# ---- ঠিক করা train.py (base64+zlib এমবেড) ----
EMBEDDED = {
@PAYLOAD@
}

b64 = EMBEDDED["train.py"]

# কপি-পেস্টে কোনো অক্ষর বদলে গেলে আগেই ধরি — রহস্যময় zlib error-এর বদলে স্পষ্ট বার্তা
recv_digest = hashlib.sha256(b64.encode()).hexdigest()[:16]
if recv_digest != "@SHA@":
    raise SystemExit(
        "কপি-পেস্টে ডেটা কিছুটা ভেঙেছে (অক্ষর বদলে গেছে)।\\n"
        "এই কোডটা বড়, তাই কপি করার সময় একটু পেস্ট মিস হয়ে থাকতে পারে।\\n"
        "সমাধান: উপরে থেকে অত্যন্ত খেয়াল করে পুরোপুরি নতুন করে কপি করে\\n"
        "ঠিক এই cell-এ আবার পেস্ট করে চালান। (প্রত্যাশিত কোড " + "@SHA@" + ", পেয়েছি " + recv_digest + ")"
    )
print("ডেটা-যাচাই ঠিক আছে (sha256 check passed)")

data = zlib.decompress(base64.b64decode(b64))
new_text = data.decode("utf-8")
(SRC / "train.py").write_text(new_text, encoding="utf-8")
print(f"বদলানো হলো: ml/src/train.py ({len(new_text.splitlines())} lines)")

py_compile.compile(str(SRC / "train.py"), doraise=True)

t = (SRC / "train.py").read_text(encoding="utf-8")
assert "FATAL: dataset folder order" not in t, "পুরোনো guard এখনো আছে!"
assert "class_order.json" in t
assert "build_model(len(class_names))" in t
assert "keras.layers.Random" not in t

print("DONE — train.py ঠিক করা মারা গেছে। এখন cell 4a আবার চালান।")
'''

cell = TEMPLATE.replace("@PAYLOAD@", payload).replace("@SHA@", sha)
OUT.write_text(cell, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, blob {size_kb:.0f} KB, sha {sha})")

# ---------- verification 1: clean cell on old clone ----------
sim = Path("/tmp/trainsim")
subprocess.run(["rm", "-rf", str(sim)], check=True)
repo_dir = sim / "AI-Crop-Disease-Detector-for-Bangladeshi-Farmers"
(repo_dir / "ml" / "src").mkdir(parents=True)
raw = subprocess.run(
    ["git", "-C", str(ROOT), "archive", "143b37b", "ml/src"],
    check=True, capture_output=True,
).stdout
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    tar.extractall(repo_dir)
old_model = subprocess.run(
    ["git", "-C", str(ROOT), "show", "143b37b:ml/src/model.py"],
    check=True, capture_output=True,
).stdout
(repo_dir / "ml" / "src" / "model.py").write_bytes(old_model)
old_text = (repo_dir / "ml" / "src" / "train.py").read_text()
assert "FATAL: dataset folder order differs" in old_text

proc = subprocess.run(
    ["python", "-c", cell], capture_output=True, text=True,
    cwd=sim, env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(sim)},
)
print(proc.stdout)
assert proc.returncode == 0, f"cell failed: {proc.stderr}"
assert "DONE" in proc.stdout and "ডেটা-যাচাই" in proc.stdout
patched = (repo_dir / "ml" / "src" / "train.py").read_bytes()
assert patched == (SRC / TARGET).read_bytes(), "not byte-identical!"
print("VERIFY-1: clean paste -> exit 0, train.py byte-identical — OK")

# ---------- verification 2: corrupted paste -> friendly message ----------
corrupt = cell.replace("eNq1", "eNq0", 1)  # one letter flipped, like a bad copy
proc2 = subprocess.run(
    ["python", "-c", corrupt], capture_output=True, text=True,
    cwd=sim, env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(sim)},
)
print("--- corrupted paste output ---")
print(proc2.stdout)
combined = proc2.stdout + proc2.stderr
assert "কপি-পেস্ট" in combined, combined
assert "Traceback" not in combined and "zlib.error" not in combined and "incorrect data check" not in combined, combined
print("VERIFY-2: corrupted paste -> friendly Bangla message (no crash) — OK")

subprocess.run(["rm", "-rf", str(sim)], check=True)
