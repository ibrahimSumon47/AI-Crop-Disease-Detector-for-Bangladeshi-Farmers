"""
Vercel serverless entrypoint — hosts the WHOLE backend under /api/* so the
entire product (Next.js frontend + FastAPI backend + ONNX model) can live on
a single Vercel project.

How it works:
- `backend/` files ship into the Python function bundle via vercel.json
  -> functions.includeFiles (backend/app includes model.onnx ~9 MB).
- This wrapper mounts the REAL backend routers under "/api" — same
  preprocess/inference/guideline services, no duplicated logic.
- DATABASE_URL must point at `/tmp` on Vercel (env var) since the deployed
  filesystem is read-only outside /tmp; note that SQLite there is not
  durable across cold starts / redeploys (documented in docs/deploy.md).
- Same-origin /api/* calls mean NO CORS is involved in the full-Vercel mode.

Ops note: set MAX_UPLOAD_MB=4 on Vercel because serverless request bodies
are capped at ~4.5 MB.
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

# Make the repo root importable so `backend.app...` resolves inside the
# serverless bundle (backend/** is included via vercel.json).
_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI  # noqa: E402

from backend.app.db import init_db  # noqa: E402
from backend.app.routers import chat, history, predict  # noqa: E402
from backend.app.services import guidelines, inference  # noqa: E402


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Same startup work as the standalone backend (db, guidelines, model)."""
    init_db()
    guidelines.load_guidelines()
    inference.load_model()
    app.state.model_loaded = True
    yield


app = FastAPI(
    title="AI Crop Disease Detector API (Vercel mode)",
    description="Same API as the standalone backend, mounted under /api/*.",
    version="0.1.0",
    lifespan=lifespan,
)

# Reuse the real routers — /api/predict + /api/history.
app.include_router(chat.router, prefix="/api")
app.include_router(history.router, prefix="/api")
app.include_router(predict.router, prefix="/api")


@app.get("/api/health", tags=["system"])
def health_check() -> dict:
    """Mirror of the standalone /health (kept under /api for symmetry)."""
    return {
        "status": "ok",
        "service": "crop-disease-detector",
        "mode": "vercel-serverless",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
    }
