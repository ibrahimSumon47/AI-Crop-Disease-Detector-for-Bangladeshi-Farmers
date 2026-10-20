#!/usr/bin/env python3
"""Emit one self-contained Colab cell (cell_fix.txt) + verify it end-to-end.

The cell embeds the FIXED augmentations.py and train.py (base64+zlib) and
rewrites them in the cloned repo inside the Colab runtime, then self-verifies.
We simulate the Colab clone (old commit 143b37b + old model.py) and run the
exact cell text to prove: exit 0, files byte-identical to the fixed repo
versions, and the old FATAL guard / keras.layers.Random code gone.
"""

import base64
import subprocess
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "ml" / "src"
OUT = ROOT / "cell_fix.txt"

TEMPLATE = '''import glob, pathlib, py_compile, base64, zlib

candidates = []
for root in [pathlib.Path("/content"), pathlib.Path.cwd(), pathlib.Path.cwd().parent]:
    candidates += sorted(glob.glob(str(root / "*AI-Crop-Disease-Detector*/ml/src")))
    if (root / "ml" / "src" / "train.py").is_file():
        candidates.append(str(root / "ml" / "src"))
assert candidates, "রিপো ফোল্ডার পাওয়া যায়নি — আগে cell 1a (git clone) রান করুন"
SRC = pathlib.Path(candidates[0])
print("লিখছি ফোল্ডারে:", SRC)

# ---- ঠিক করা ফাইলগুলো (base64+zlib এমবেড করা, হুবহু রিপো ভার্সন) ----
EMBEDDED = {
@PAYLOAD@
}

# ---- লিখো + যাচাই করো ----
for name in EMBEDDED:
    data = zlib.decompress(base64.b64decode(EMBEDDED[name]))
    (SRC / name).write_text(data.decode("utf-8"), encoding="utf-8")
    print(f"বদলানো হলো: ml/src/{name} ({len(data.splitlines())} lines)")

for name in EMBEDDED:
    py_compile.compile(str(SRC / name), doraise=True)

t = (SRC / "train.py").read_text(encoding="utf-8")
a = (SRC / "augmentations.py").read_text(encoding="utf-8")
assert "FATAL: dataset folder order" not in t, "পুরোনো guard এখনো আছে!"
assert "class_order.json" in t
assert "build_model(len(class_names))" in t
assert "keras.layers.Random" not in a

print("DONE — ফাইল লেখা ও যাচাই সম্পন্ন। এখন cell 4a আবার চালান।")
'''


def payload_for(name: str) -> str:
    blob = base64.b64encode(zlib.compress((SRC / name).read_bytes(), 9)).decode()
    lines = [f'    "{name}": (']
    for i in range(0, len(blob), 72):
        lines.append(f'        "{blob[i:i+72]}"')
    lines.append('    ),')
    return "\n".join(lines)


DEF_FILES = ["augmentations.py", "train.py"]

# 1) build the dict placeholder
dict_body = "\n".join(payload_for(f) for f in DEF_FILES)
cell = TEMPLATE.replace("@PAYLOAD@", dict_body)
OUT.write_text(cell, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size} bytes)")

# 2) end-to-end verification in a simulated clone
sim = Path("/tmp/cellsim")
subprocess.run(["rm", "-rf", str(sim)], check=True)
repo_dir = sim / "AI-Crop-Disease-Detector-for-Bangladeshi-Farmers"
(repo_dir / "ml" / "src").mkdir(parents=True)

import io
import tarfile

raw = subprocess.run(
    ["git", "-C", str(ROOT), "archive", "143b37b", "ml/src"],
    check=True, capture_output=True,
).stdout
with tarfile.open(fileobj=io.BytesIO(raw)) as tar:
    tar.extractall(repo_dir)
# Colab clone also carries the (old) model.py untouched by later fixes
old_model = subprocess.run(
    ["git", "-C", str(ROOT), "show", "143b37b:ml/src/model.py"],
    check=True, capture_output=True,
).stdout
(repo_dir / "ml" / "src" / "model.py").write_bytes(old_model)

# sanity: old clone really is old
old_text = (repo_dir / "ml" / "src" / "train.py").read_text()
assert "FATAL: dataset folder order differs" in old_text, "simulation setup wrong"
assert "build_model()" in old_text

proc = subprocess.run(
    ["python", "-c", OUT.read_text()],
    capture_output=True, text=True, cwd=sim,
    env={"PATH": "/usr/bin:/bin:/usr/local/bin", "HOME": "/tmp/cellsim"},
)
print(proc.stdout)
assert proc.returncode == 0, f"cell failed: {proc.stderr}"
assert "DONE" in proc.stdout

for f in DEF_FILES:
    fixed = (repo_dir / "ml" / "src" / f).read_bytes()
    want = (SRC / f).read_bytes()
    assert fixed == want, f"{f} not byte-identical!"
t_ok = (repo_dir / "ml" / "src" / "train.py").read_text()
a_ok = (repo_dir / "ml" / "src" / "augmentations.py").read_text()
assert "FATAL" not in t_ok and "keras.layers.Random" not in a_ok
print("VERIFY: patched files byte-identical to fixed repo versions — OK")
