# Production image for the AI Crop Disease Detector backend.
#
# Works on both target hosts:
#   - Hugging Face Spaces (expects port 7860; set PORT=7860 in Space config)
#   - Render (injects PORT automatically; defaults here keep dev localhost)
#
# Build & run locally:
#   cd backend && docker build -t crop-disease-api . && docker run -p 8000:8000 crop-disease-api

FROM python:3.11-slim

# Environment first: no bytecode, unbuffered logs, deterministic behavior.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libgomp1: onnxruntime needs the OpenMP runtime on Debian-slim.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy metadata first for better layer caching.
COPY requirements.txt .
RUN pip install -r requirements.txt

# App code + data (model.onnx ~9 MB, labels.json, guidelines_bn.json).
COPY app ./app

# Non-root user (HF Spaces and Render both like this).
RUN useradd -m appuser && chown -R appuser:appuser /app
USER appuser

# Hosts inject PORT (Render: 10000-ish, HF Spaces: 7860); default 8000 locally.
ENV PORT=8000
EXPOSE 8000

# SQLite file lives in /app (writable in-container; note it resets on redeploy —
# acceptable for scan history on free tiers; documented in docs/deploy.md).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
