"""
GET /history — scan history support for the frontend.

Simple admin-free history of stored scans (this is a single-user MVP
for the capstone). Pagination via ?limit / ?offset.
"""

import logging

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..db import get_db
from ..models import Scan
from ..schemas import HistoryItem, HistoryResponse

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
def get_history(
    limit: int = Query(20, ge=1, le=100, description="Rows per page"),
    offset: int = Query(0, ge=0, description="Rows to skip"),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    """Return the most recent scans, newest first."""
    q = db.query(Scan).order_by(Scan.created_at.desc())
    return HistoryResponse(
        items=[HistoryItem(**s.to_history_item()) for s in q.offset(offset).limit(limit).all()],
        total=db.query(func.count(Scan.id)).scalar() or 0,
    )
