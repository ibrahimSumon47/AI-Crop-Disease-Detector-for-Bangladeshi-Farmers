#!/usr/bin/env python3
"""Emit a classes.py-only repair cell (cell_classes_fix.txt) + verify it.

Rationale: the user's train.py fix landed, and 4a now fails at
  ImportError: cannot import name 'missing_from_dataset' from 'classes'
because Colab's clone still carries the OLD classes.py (old commit has no
missing_from_dataset — it was added in the label-order fix). Ship the fixed
classes.py the same way that just worked: small embedded blob + sha256
paste guard + py_compile + content asserts.

Verification:
  1) clean cell on a simulated clone from old commit 143b37b -> exit 0,
     classes.py byte-identical to the fixed repo version, and
     missing_from_dataset() returns exactly the 2 canonical classes missing
     from the user's real 11-folder list.
  2) corrupted blob -> friendly Bangla re-copy message, no crash.
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
OUT = ROOT / "cell_classes_fix.txt"
TARGET = "classes.py"

blob = base64.b64encode(zlib.compress((SRC / TARGET).read_bytes(), 9)).decode()
sha = hashlib.sha256(blob.encode()).hexdigest()[:16]

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

# ---- ঠিক করা classes.py (base64+zlib এমবেড) ----
EMBEDDED = {
@PAYLOAD@
}

b64 = EMBEDDED["classes.py"]

# কপি-পেস্টে কোনো অক্ষর বদলে গেলে আগেই ধরি
recv_digest = hashlib.sha256(b64.encode()).hexdigest()[:16]
if recv_digest != "@SHA@":
    raise SystemExit(
        "কপি-পেস্টে ডেটা কিছুটা ভেঙেছে (অক্ষর বদলে গেছে)।\\n"
        "সমাধান: উপরে থেকে খেয়াল করে পুরোপুরি নতুন করে কপি করে\\n"
        "ঠিক এই cell-এ আবার পেস্ট করে চালান। (প্রত্যাশিত " + "@SHA@" + ", পেয়েছি " + recv_digest + ")"
    )
print("ডেটা-যাচাই ঠিক আছে (sha256 check passed)")

data = zlib.decompress(base64.b64decode(b64))
new_text = data.decode("utf-8")
(SRC / "classes.py").write_text(new_text, encoding="utf-8")
print(f"বদলানো হলো: ml/src/classes.py ({len(new_text.splitlines())} lines)")

py_compile.compile(str(SRC / "classes.py"), doraise=True)

import sys
sys.path.insert(0, str(SRC))
import classes

assert "missing_from_dataset" in new_text
assert "N_CLASSES" in new_text and "EN_NAMES" in new_text and "BN_NAMES" in new_text
classes.validate()
assert classes.N_CLASSES == 13

# আপনার আসল ১১-ফোল্ডার লিস্টে ২টা ক্লাস সত্যিই missing কি না
missing = classes.missing_from_dataset(
    ["rice_brown_spot", "rice_blast", "rice_healthy", "potato_early_blight",
     "potato_late_blight", "potato_healthy", "tomato_bacterial_spot",
     "tomato_early_blight", "tomato_late_blight", "tomato_leaf_mold",
     "tomato_healthy"]
)
assert missing == ["rice_bacterial_leaf_blight", "rice_tungro"], missing
print("missing_from_dataset() পরীক্ষিত — ভালো কাজ করছে:", missing)

print("DONE — classes.py ঠিক হয়ে গেছে। এখন cell 4a আবার চালান।")
'''

cell = TEMPLATE.replace("@PAYLOAD@", payload).replace("@SHA@", sha)
OUT.write_text(cell, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB, blob {len(blob) / 1024:.0f} KB, sha {sha})")

# ---------- 1: clean cell on old clone ----------
sim = Path("/tmp/classsim")
subprocess.run(["rm", "-rf", str(sim)], check=True)
repo_dir = sim / "AI-Crop-Disease-Detector-for-Bangladeshi-Farmers"
(repo_dir / "ml" / "src").mkdir(parents=True)
raw = subprocess.run(
    ["git", "-C", str(ROOT), "archive", "143b37b", "ml/src"],
    check=True, capture_output=True,
).stdout
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    tar.extractall(repo_dir)

# the user's Colab: train.py + augmentations.py already repaired by earlier cells
for f in ["train.py", "augmentations.py"]:
    (repo_dir / "ml" / "src" / f).write_bytes((SRC / f).read_bytes())
old_model = subprocess.run(
    ["git", "-C", str(ROOT), "show", "143b37b:ml/src/model.py"],
    check=True, capture_output=True,
).stdout
(repo_dir / "ml" / "src" / "model.py").write_bytes(old_model)
# old classes.py must genuinely lack missing_from_dataset
old_c = (repo_dir / "ml" / "src" / "classes.py").read_text()
assert "missing_from_dataset" not in old_c

proc = subprocess.run(
    ["python", "-c", cell], capture_output=True, text=True,
    cwd=sim, env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(sim)},
)
print(proc.stdout)
assert proc.returncode == 0, f"cell failed:\n{proc.stdout}\n{proc.stderr}"
assert "DONE" in proc.stdout
patched = (repo_dir / "ml" / "src" / "classes.py").read_bytes()
assert patched == (SRC / TARGET).read_bytes(), "not byte-identical!"
print("VERIFY-1: clean paste -> exit 0, classes.py byte-identical — OK")

# ---------- 2: corrupted paste ----------
corrupt = cell.replace("eNq", "fNq", 1)
proc2 = subprocess.run(
    ["python", "-c", corrupt], capture_output=True, text=True,
    cwd=sim, env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": str(sim)},
)
combined = proc2.stdout + proc2.stderr
print("--- corrupted paste output ---")
print(combined)
assert "কপি-পেস্ট" in combined, combined
assert "Traceback" not in combined and "incorrect data check" not in combined, combined
print("VERIFY-2: corrupted paste -> friendly Bangla message — OK")

subprocess.run(["rm", "-rf", str(sim)], check=True)
