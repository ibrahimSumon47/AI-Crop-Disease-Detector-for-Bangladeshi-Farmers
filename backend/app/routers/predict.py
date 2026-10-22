"""
POST /predict — the core endpoint of the app.

Flow (matches docs/architecture.md):
    upload -> preprocess -> ONNX top-3 -> calibrated confidence threshold ->
    Bangla guideline lookup (optional LLM rewrite) -> persist Scan row ->
    response JSON.

Threshold semantics: if the top-1 confidence is below CONFIDENCE_THRESHOLD
(calibrated at 0.97 from the validation sweep), we STILL return the top-3
but set is_uncertain=True — the frontend then renders the honest
"নিশ্চিত না, কৃষি অফিসারের সাথে কথা বলুন" screen instead of guessing.
"""

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, status
from sqlalchemy.orm import Session

from ..config import settings
from ..db import get_db
from ..models import Scan
from ..schemas import Advice, PredictionItem, PredictionResponse
from ..services import guidelines as gl
from ..services import inference, plant_guard, processing
from ..services.preprocess import ImageValidationError, preprocess, validate_image

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
async def predict_leaf(
    file: UploadFile = File(..., description="Leaf photo (JPG/PNG, max 5 MB)"),
    db: Session = Depends(get_db),
) -> PredictionResponse:
    """Diagnose a leaf photo and return crop, disease, confidence, advice."""
    data = await file.read()

    try:
        img = validate_image(data)
    except ImageValidationError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    # Sanity gate BEFORE inference: a photo with no plant material (a tuber
    # on a plain background, a flat/blank image) must be refused with
    # guidance instead of a misleading 2% "diagnosis".
    leaf_ok, leaf_reason = plant_guard.looks_like_leaf(img)
    if not leaf_ok:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, leaf_reason)

    # Advisory photo-quality notes (blur/dark). Not blocking: bad camera
    # photos still get a full diagnosis, the response just carries Bangla
    # notes on how to take a better shot.
    photo_warnings = plant_guard.photo_warnings(img)

    batch = preprocess(img)

    top3 = inference.predict_top3(batch)
    family_note = processing.attach_family_note(top3)
    crop_unclear = processing.is_crop_unclear(top3)
    best = top3[0]
    is_uncertain = best["confidence"] < settings.confidence_threshold

    advice_raw = await gl.get_advice(best["class_name"])
    advice = Advice(
        symptoms=advice_raw["symptoms"] if advice_raw else None,
        treatment=advice_raw["treatment"] if advice_raw else None,
        organic=advice_raw["organic"] if advice_raw else None,
        prevention=advice_raw["prevention"] if advice_raw else None,
        expert_threshold=advice_raw["expert_threshold"] if advice_raw else None,
        source=advice_raw["source"] if advice_raw else "none",
    )

    # Persist the scan so /history can reproduce it later.
    scan = Scan(
        created_at=datetime.utcnow(),
        filename=file.filename,
        crop=best["crop"],
        disease=best["class_name"],
        confidence=best["confidence"],
        top3=_top3_to_json(top3),
        is_uncertain=is_uncertain,
    )
    db.add(scan)
    db.commit()  # scan.id is populated after flush

    return PredictionResponse(
        crop=best["crop"],
        disease=best["class_name"],
        disease_bn=advice_raw["disease_bn"] if advice_raw else best["class_name_bn"],
        confidence=best["confidence"],
        top3=[PredictionItem(**p) for p in top3],
        is_uncertain=is_uncertain,
        advice=advice,
        family_note=family_note,
        crop_unclear=crop_unclear,
        photo_warnings=photo_warnings,
        scan_id=scan.id,
        created_at=scan.created_at,
    )


def _top3_to_json(top3: list[dict]) -> str:
    """Compact JSON for the Scan.top3 text column (no scalars lost)."""
    import json
    return json.dumps(
        [
            {
                "class": p["class_name"],
                "class_bn": p["class_name_bn"],
                "confidence": round(p["confidence"], 4),
            }
            for p in top3
        ],
        ensure_ascii=False,
    )
