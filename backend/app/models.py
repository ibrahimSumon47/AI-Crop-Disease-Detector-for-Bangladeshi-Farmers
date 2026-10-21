"""
ORM models. One table for now: Scan (each diagnosis request gets a row).

Design notes:
- top3 is stored as a JSON text column containing
  [{"class", "class_bn", "confidence"}, ...] so a single row fully
  reproduces the UI response later (audit/debug friendly).
- is_uncertain records whether the calibrated threshold forced the
  "consult an officer" answer — useful statistics for improving the model
  with real deployment traffic later.
- crop/disease stored as canonical machine names (potato_late_blight...);
  the Bangla display name lives in guidelines/labels JSON, not in the DB.
"""

from datetime import datetime

from sqlalchemy import Boolean, Column, DateTime, Float, Integer, String, Text

from .db import Base


class Scan(Base):
    """One stored leaf-scan diagnosis."""

    __tablename__ = "scans"

    id = Column(Integer, primary_key=True, index=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True, nullable=False)

    # Original uploaded filename (display only; not used for classification).
    filename = Column(String(255), nullable=True)

    # Prediction result (canonical names, e.g. "tomato_late_blight").
    crop = Column(String(64), nullable=False)
    disease = Column(String(64), nullable=False)
    confidence = Column(Float, nullable=False)

    # JSON string: [{"class": ..., "class_bn": ..., "confidence": ...}] x3
    top3 = Column(Text, nullable=False)

    # True when confidence < CONFIDENCE_THRESHOLD -> UI must show
    # "নিশ্চিত না, কৃষি অফিসারের সাথে কথা বলুন".
    is_uncertain = Column(Boolean, default=False, nullable=False)

    def to_history_item(self) -> dict:
        """Minimal dict for the history endpoint (no heavy JSON parse)."""
        return {
            "id": self.id,
            "created_at": self.created_at,
            "filename": self.filename,
            "crop": self.crop,
            "disease": self.disease,
            "confidence": self.confidence,
            "is_uncertain": self.is_uncertain,
        }
