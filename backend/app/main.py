"""
FastAPI entrypoint for the AI Crop Disease Detector backend.

Request flow for the main endpoint (see docs/architecture.md):
    image upload -> validation/preprocess -> ONNX inference -> confidence
    threshold -> guideline lookup (Bangla) -> save scan -> JSON response

Run locally:
    cd backend
    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import settings

# Plain logging; keep it simple and greppable in hosting dashboards.
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown hook: load the model once, keep it in memory.

    Loading the ONNX session at startup (instead of per-request) keeps
    inference latency stable. The heavy work lives in services/inference.py.
    """
    from .services import inference  # delayed: keeps app import cheap (T3.7 wires it)

    inference.load_model()  # warm-up: reads labels.json + creates the session
    app.state.model_loaded = True
    logger.info("Model loaded and ready.")
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="AI Crop Disease Detector API",
    description=(
        "Bangla-first disease diagnosis for Bangladeshi farmers. "
        "Upload a leaf photo; get crop, disease, confidence and treatment advice."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS so the Next.js frontend (and only it, per ALLOWED_ORIGINS) can call us.
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", tags=["system"])
def health_check() -> dict:
    """Lightweight probe for monitoring and CI.

    Returns basic service info. `model_loaded` reflects the real state set
    by the lifespan startup (False while the model is still loading).
    """
    return {
        "status": "ok",
        "service": "crop-disease-detector",
        "model_loaded": bool(getattr(app.state, "model_loaded", False)),
    }


# Routers are attached when they exist; keep /health above so monitoring
# never depends on model code paths.
