"""
Pydantic response schemas — the API contract the Next.js frontend codes against.

Field naming is camel-free on purpose: the frontend consumes Bangla strings
and snake_case keys equally and the team is small; consistency across the
project beats short keys.
"""

from datetime import datetime

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """One ranked class inside a prediction (used for top-1 and top-3)."""

    rank: int = Field(..., ge=1, description="1 = most likely class")
    class_name: str = Field(..., description="Canonical name, e.g. rice_blast")
    class_name_bn: str = Field(..., description="Bangla display name for the UI")
    crop: str = Field(..., description="rice | potato | tomato")
    confidence: float = Field(..., ge=0.0, le=1.0)


class Advice(BaseModel):
    """Bangla treatment advice, grounded in the static guidelines JSON."""

    symptoms: str | None = None
    treatment: str | None = None
    organic: str | None = None
    prevention: str | None = None
    expert_threshold: str | None = None  # when to call an agriculture officer
    source: str = Field("guideline", description="guideline | llm | none")


class PredictionResponse(BaseModel):
    """Full JSON returned by POST /predict."""

    crop: str
    disease: str
    disease_bn: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    top3: list[PredictionItem] = Field(..., description="Best 3 candidate classes")
    is_uncertain: bool = Field(
        ...,
        description=(
            "True when confidence < CONFIDENCE_THRESHOLD; the UI must then show "
            "'নিশ্চিত না' and recommend the agriculture officer."
        ),
    )
    advice: Advice
    family_note: str | None = Field(
        None,
        description=(
            "Present when the model is torn between the same disease on potato "
            "and tomato (shared blight family), or when a rice photo needs a "
            "leaf close-up. The frontend displays this note prominently; the "
            "attached advice applies as described in the note."
        ),
    )
    crop_unclear: bool = Field(
        False,
        description=(
            "True when even the top-1 confidence is below 0.25 — the crop field "
            "itself is a guess. The UI hides the most-likely advice and coaches "
            "the farmer to retake a proper leaf close-up in good light."
        ),
    )
    scan_id: int | None = Field(None, description="DB id of the saved scan row")
    created_at: datetime


class HistoryItem(BaseModel):
    """One row of scan history (GET /history)."""

    id: int
    created_at: datetime
    filename: str | None
    crop: str
    disease: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_uncertain: bool


class HistoryResponse(BaseModel):
    """Paginated history envelope."""

    items: list[HistoryItem]
    total: int
