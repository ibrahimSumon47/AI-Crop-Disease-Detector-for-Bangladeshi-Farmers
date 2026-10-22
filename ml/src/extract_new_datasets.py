"""
extract_new_datasets.py — one-time helper (read-only on the archives).

Extracts ONLY the leaf-disease class folders we need from the datasets the
user downloaded into "New folder (2)", rewriting class-folder names to the
canonical names in classes.py. This keeps the raw/ tree clean:

    ml/data/raw/<source>/<canonical_class>/img...

Why rename to canonical at extract time?
    Many datasets use bare generic folders ("Healthy", "Anthracnose") that
    also exist in OTHER datasets under raw/ — prepare_data.py matches on the
    immediate parent folder name, so a bare "Healthy" would be ambiguous
    (rice wins pass-1 of the alias map). Renaming here makes every extracted
    folder unambiguous and means prepare_data.py needs zero new aliases.

Sources and how they are opened:
    plan name            archive type   extractor
    mln1.zip             ordinary zip   Python zipfile
    Okra/spv PD etc.     ordinary zip   Python zipfile
    msv1.zip             BROKEN zip*    7-Zip (local-header reader)
    Rice Leaf Images.7z  7z             7-Zip

    * msv1.zip's md5 matches the Zenodo record checksum exactly, so the
    download is intact; the file simply has no zip central directory.
    Python zipfile and unzip cannot open it, but 7-Zip reads it fine via
    the local file headers.

Sources extracted (round 1):
    plantvillage     <- PlantVillage Dataset.zip (corn/grape/potato/tomato only)
    maize_tz         <- mln1.zip (Maize Lethal Necrosis)
    grape_ngld       <- NGLD.zip (Downy/Powdery Mildew + Bacterial Leaf Spot)
    cauliflower_bd   <- Cauliflower Leaf Diseases Original Image.zip
    cucumber_mr      <- Cucumber Disease Recognition ... Original Image.zip
    papaya_bd        <- BDPapayaLeaf.zip (Original Images only)
    mango_bd         <- MangoLeafBD Dataset.zip
    okra_diseasenet  <- Okra DiseaseNet Dataset.zip (Train+Test+Val merged)

Sources extracted (round 2 — added 2026-10-21):
    chilli_bd        <- Chili Leaf Disease Original Dataset.zip
    rice_leafbd      <- RiceLeafBD ... Original Images.zip
    rice_samples     <- Rice Leaf Disease Images.7z
    maize_tz         <- msv1.zip (Maize Streak Virus, via 7-Zip)

CHILI GROWTH STAGE zips are intentionally NOT extracted: "Dry chili /
Red Chili / Flower / ..." are growth-stage photos, NOT leaf diseases —
out of scope for this app.

Usage:
    python ml/src/extract_new_datasets.py            # process all planned archives
    python ml/src/extract_new_datasets.py --only msv1.zip
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ZIPS_DIR = REPO_ROOT / "New folder (2)"
OUT = REPO_ROOT / "ml" / "data" / "raw"
STAGING = REPO_ROOT / "data" / "_tmp_extract"

SEVEN_ZIP = Path("C:/Program Files/7-Zip/7z.exe")

# Archives that must go through 7-Zip instead of Python zipfile:
#   - "msv1.zip": Zenodo file without a zip central directory (md5 verified,
#     but zipfile refuses) — 7-Zip reads it via local headers.
#   - the rice .7z: 7z archive, not a zip.
NEEDS_7ZIP: set[str] = {"msv1.zip", "Rice Leaf Disease Images.7z"}

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# archive name -> list of (class-folder-key, source_dir_out, canonical_class)
# class-folder-key is matched against the archive's folder names (lowercase,
# stripped); for msv1.zip the archive contains ONLY loose images in one
# folder ("msv1"), so the key is that folder name.
PLANS: dict[str, dict[str, tuple[str, str]]] = {}


def _plan(zip_name: str, source: str, mapping: dict[str, str]) -> None:
    PLANS[zip_name] = {k.lower().strip(): (source, v) for k, v in mapping.items()}


_plan("PlantVillage Dataset.zip", "plantvillage", {
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "maize_gray_leaf_spot",
    "Corn_(maize)___Common_rust_": "maize_common_rust",
    "Corn_(maize)___Northern_Leaf_Blight": "maize_northern_leaf_blight",
    "Corn_(maize)___healthy": "maize_healthy",
    "Grape___Black_rot": "grape_black_rot",
    "Grape___Esca_(Black_Measles)": "grape_esca",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "grape_leaf_blight",
    "Grape___healthy": "grape_healthy",
    "Potato___Early_blight": "potato_early_blight",
    "Potato___Late_blight": "potato_late_blight",
    "Potato___healthy": "potato_healthy",
    "Tomato___Bacterial_spot": "tomato_bacterial_spot",
    "Tomato___Early_blight": "tomato_early_blight",
    "Tomato___Late_blight": "tomato_late_blight",
    "Tomato___Leaf_Mold": "tomato_leaf_mold",
    "Tomato___healthy": "tomato_healthy",
})

# Tanzania maize: zip content is a folder mln1/ containing loose images
# (no class subfolders) — everything inside is MLN diseased leaves.
_plan("mln1.zip", "maize_tz", {"mln1": "maize_lethal_necrosis"})

_plan("Niphad Grape Leaf Disease Dataset (NGLD).zip", "grape_ngld", {
    "Bacterial Leaf Spot": "grape_bacterial_leaf_spot",
    "Downy Mildew": "grape_downy_mildew",
    "Powdery Mildew": "grape_powdery_mildew",
    "Healthy Leaves": "grape_healthy",
})

_plan("Cauliflower Leaf Diseases Original Image.zip", "cauliflower_bd", {
    "Black Rot": "cauliflower_black_rot",
    "Insect Hole": "cauliflower_insect_holes",
    "Healthy": "cauliflower_healthy",
})

_plan("Cucumber Disease Recognition Dataset Original Image.zip", "cucumber_mr", {
    "Anthracnose": "cucumber_anthracnose",
    "Bacterial Wilt": "cucumber_bacterial_wilt",
    "Belly Rot": "cucumber_belly_rot",
    "Downy Mildew": "cucumber_downy_mildew",
    "Fresh Cucumber": "cucumber_healthy",
    "Fresh Leaf": "cucumber_healthy",
    "Gummy Stem Blight": "cucumber_gummy_stem_blight",
    "Pythium Fruit Rot": "cucumber_pythium_fruit_rot",
})

_plan("BDPapayaLeaf.zip", "papaya_bd", {
    "Anthracnose": "papaya_anthracnose",
    "BacterialSpot": "papaya_bacterial_spot",
    "Curl": "papaya_curl",
    "RingSpot": "papaya_ring_spot",
    "Healthy": "papaya_healthy",
})

_plan("MangoLeafBD Dataset.zip", "mango_bd", {
    "Anthracnose": "mango_anthracnose",
    "Bacterial Canker": "mango_bacterial_canker",
    "Cutting Weevil": "mango_cutting_weevil",
    "Die Back": "mango_die_back",
    "Gall Midge": "mango_gall_midge",
    "Powdery Mildew": "mango_powdery_mildew",
    "Sooty Mould": "mango_sooty_mould",
    "Healthy": "mango_healthy",
})

# Okra DiseaseNet: class folders look like "Class 0 - Alternaria Leaf Spot"
# and appear under Training/Testing/Validation with inconsistent spellings.
# Matched below by a lowercase substring of the folder name; the substring
# keys must be checked with a different matcher (see _match_plan_key).
_plan("Okra DiseaseNet Dataset.zip", "okra_diseasenet", {
    "alternaria": "okra_alternaria_leaf_spot",
    "cercospora": "okra_cercospora_leaf_spot",
    "downy": "okra_downy_mildew",
    "healthy": "okra_healthy",
    "curly": "okra_leaf_curl_virus",
    "phyllosticta": "okra_phyllosticta_leaf_spot",
})

_plan("Chili Leaf Disease Original Dataset.zip", "chilli_bd", {
    "Bacterial Spot": "chilli_bacterial_spot",
    "Cercospora Leaf Spot": "chilli_cercospora_leaf_spot",
    "Curl Virus": "chilli_curl_virus",
    "Healthy Leaf": "chilli_healthy",
    "Nutrition Deficiency": "chilli_nutrient_deficiency",
    "White spot": "chilli_white_spot",
})

_plan("RiceLeafBD A Real-Field Image Dataset for Rice Leaf Disease Detection and Classification in Bangladesh Original Images.zip", "rice_leafbd", {
    "Bacterial Leaf Blight": "rice_bacterial_leaf_blight",
    "Brown Spot": "rice_brown_spot",
    "Healthy Leaf": "rice_healthy",
    "Tungro Virus": "rice_tungro",
})

_plan("Rice Leaf Disease Images.7z", "rice_samples", {
    "Bacterialblight": "rice_bacterial_leaf_blight",
    "Blast": "rice_blast",
    "Brownspot": "rice_brown_spot",
    "Tungro": "rice_tungro",
})

_plan("msv1.zip", "maize_tz", {"msv1": "maize_streak_virus"})


def _match_plan_key(zip_name: str, plan: dict, folder: str) -> str | None:
    """Return the plan key (or None) matching a folder inside archive.

    Okra uses substring matching ("Class 0 - Alternaria leaf spot").
    Every other archive uses an exact (lowercased) match on a folder name.
    """
    key = folder.strip().lower()
    if zip_name == "Okra DiseaseNet Dataset.zip":
        for candidate in plan:
            if candidate in key:
                return candidate
        return None
    return key if key in plan else None


def _save_image(src_file: Path, source: str, canonical: str,
                counts: dict) -> None:
    out_dir = OUT / source / canonical
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = src_file.stem
    dst = out_dir / f"{stem}{src_file.suffix.lower()}"
    i = 1
    while dst.exists():
        dst = out_dir / f"{stem}_{i}{src_file.suffix.lower()}"
        i += 1
    shutil.copy2(src_file, dst)
    counts[(source, canonical)] = counts.get((source, canonical), 0) + 1


def extract_via_7zip(zip_path: Path) -> None:
    """Extract a whole archive with 7-Zip into STAGING, then bucket files."""
    if not SEVEN_ZIP.exists():
        raise SystemExit(
            f"7-Zip not found at {SEVEN_ZIP} — needed for '{zip_path.name}'.\n"
            "Install 7-Zip (https://www.7-zip.org) or put 7z.exe there."
        )
    STAGING.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [str(SEVEN_ZIP), "x", str(zip_path), f"-o{STAGING}", "-y", "-bd"],
        capture_output=True, text=True, timeout=1800,
    )
    # 7z code 2 = "unexpected end of archive" — most files may still be fine;
    # per-file counts below tell us what we really got. Hard failures (>=4).
    if result.returncode >= 4:
        shutil.rmtree(STAGING, ignore_errors=True)
        raise SystemExit(
            f"7-Zip failed on {zip_path.name} (code {result.returncode}):\n"
            f"{result.stdout[-1500:]}\n{result.stderr[-800:]}"
        )
    plan = PLANS[zip_path.name]
    counts: dict[tuple[str, str], int] = {}
    img_paths = [
        p for p in STAGING.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    ]
    for img in sorted(img_paths):
        rel_parts = img.relative_to(STAGING).parts
        canonical, source = None, None
        for part in rel_parts[:-1]:
            key = _match_plan_key(zip_path.name, plan, part)
            if key:
                source, canonical = plan[key]
                break
        if zip_path.name == "msv1.zip" and canonical is None:
            # the whole archive is one flat class (maize_streak_virus)
            source, canonical = PLAN_MSV_FALLBACK
        if canonical is None:
            print(f"    skipped dir: {'/'.join(rel_parts[:-1])}")
            continue
        _save_image(img, source, canonical, counts)
    shutil.rmtree(STAGING, ignore_errors=True)
    return counts


# special-case fallback for msv1.zip: flat images, no matching folder needed
PLAN_MSV_FALLBACK: tuple[str, str] = ("maize_tz", "maize_streak_virus")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only", nargs="*", metavar="NAME",
        help="Process only these archive names (default: all planned archives).",
    )
    args = parser.parse_args()

    if not ZIPS_DIR.exists():
        raise SystemExit(f"Zips folder not found: {ZIPS_DIR}")

    counts: dict[tuple[str, str], int] = {}
    audit_lines: list[str] = []
    unmatched_dirs: set[str] = set()

    for zip_name, plan in PLANS.items():
        if args.only and zip_name not in args.only:
            continue
        path = ZIPS_DIR / zip_name
        if not path.exists():
            print(f"!! missing archive: {zip_name} (not downloaded yet)")
            audit_lines.append(f"{zip_name}, MISSING")
            continue
        print(f"\n=== {zip_name} ===")
        if zip_name in NEEDS_7ZIP:
            counts7 = extract_via_7zip(path)
            counts.update(counts7)
            continue

        # --- ordinary zip handled with Python zipfile ---------------------
        with zipfile.ZipFile(path) as z:
            for m in z.infolist():
                name = m.filename
                if m.is_dir() or Path(name).suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                parts = name.split("/")
                src_dir, canonical = None, None
                for part in parts[:-1]:
                    key = _match_plan_key(zip_name, plan, part)
                    if key:
                        src_dir, canonical = plan[key]
                        break
                if canonical is None:
                    if len(parts) > 1:
                        unmatched_dirs.add("/".join(parts[:-1]))
                    continue
                out_dir = OUT / src_dir / canonical
                out_dir.mkdir(parents=True, exist_ok=True)
                # unique file name: original stem (+numeric suffix on clash)
                stem = Path(name).stem
                dst = out_dir / f"{stem}{Path(name).suffix.lower()}"
                i = 1
                while dst.exists():
                    dst = out_dir / f"{stem}_{i}{Path(name).suffix.lower()}"
                    i += 1
                with z.open(m) as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
                counts[(src_dir, canonical)] = counts.get((src_dir, canonical), 0) + 1

    print("\n================ EXTRACT SUMMARY ================")
    per_source: dict[str, int] = {}
    for (source, canonical), n in sorted(counts.items()):
        per_source[source] = per_source.get(source, 0) + n
        print(f"  {source}/{canonical}: {n}")
    print("-------------------------------------------------")
    for source, n in sorted(per_source.items()):
        print(f"  TOTAL {source}: {n}")

    print("\n================ AUDIT ================")
    print("\nUnmatched folders (may be skipped intentionally, e.g. apple):")
    for d in sorted(unmatched_dirs)[:40]:
        print(f"    - {d}")
    if len(unmatched_dirs) > 40:
        print(f"    ... and {len(unmatched_dirs) - 40} more")

    audit_path = REPO_ROOT / "ml" / "reports" / "extract_audit.txt"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text("\n".join(audit_lines), encoding="utf-8")
    print(f"\naudit written to {audit_path}")


if __name__ == "__main__":
    main()
