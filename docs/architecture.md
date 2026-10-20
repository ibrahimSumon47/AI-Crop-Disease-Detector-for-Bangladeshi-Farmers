# Architecture Overview

This document describes how a single scan request travels through the system,
which components exist, and how failure / uncertainty is handled.

## 1. High-level components

```
┌─────────────────────┐         ┌──────────────────────────────────────┐
│  Frontend (Vercel)  │  HTTPS  │   Backend (HF Spaces Docker / Render)│
│  Next.js + Tailwind │ ──────► │   FastAPI + uvicorn                  │
│  Bangla-first UI    │         │                                      │
└─────────────────────┘         │  /predict  /history  /health         │
                                │        │                             │
                                │  ┌─────▼──────────────────────────┐  │
                                │  │ preprocess → ONNX inference    │  │
                                │  │ guideline JSON → optional LLM  │  │
                                │  │ SQLite (scan history)          │  │
                                │  └────────────────────────────────┘  │
                                └──────────────────────────────────────┘
```

- **Frontend:** static/SSR Next.js app. All ML and data logic lives behind the API;
  the browser only uploads an image and renders JSON.
- **Backend:** single FastAPI service that owns the ONNX model, the Bangla
  guideline JSON, and the SQLite scan history.
- **Model:** MobileNetV2 (or EfficientNet-B0) fine-tuned on leaf-disease datasets,
  exported to ONNX and loaded **once** at startup with ONNX Runtime.

## 2. Request flow — `POST /predict`

```mermaid
sequenceDiagram
    participant F as Frontend
    participant A as FastAPI /predict
    participant P as preprocess.py
    participant I as inference.py (ONNX)
    participant T as threshold check
    participant G as advice.py (guidelines_bn.json → optional LLM)
    participant D as SQLite (Scan table)

    F->>A: multipart image upload
    A->>A: validate (JPEG/PNG, ≤5 MB, min dimensions)
    A->>P: read bytes
    P->>P: resize to 224×224, RGB, normalize exactly as in training
    P->>I: preprocessed tensor
    I->>I: softmax → 10 class probabilities
    I->>T: (label, confidence, top-3)
    alt confidence ≥ CONFIDENCE_THRESHOLD
        T->>G: class label
        G->>G: guideline lookup; if LLM_API_KEY set, rewrite in Bangla<br/>(grounded ONLY in the JSON, no invented doses)
        G->>A: symptoms / treatment / prevention text
    else confidence < threshold
        T->>A: is_uncertain = true
    end
    A->>D: INSERT Scan (metadata only, image is NOT stored)
    A-->>F: JSON {crop, disease, confidence, top3, advice, is_uncertain}
```

Step by step:

1. **Upload** — the browser sends the image as `multipart/form-data`.
2. **API validation** — type (JPEG/PNG/WebP), max size (5 MB), min dimensions;
   invalid inputs return a clean JSON error (never a stack trace).
3. **Preprocessing** (`services/preprocess.py`) — resize, color and
   normalisation **identical** to the training pipeline (this is the #1 cause
   of silent accuracy loss).
4. **ONNX inference** (`services/inference.py`) — one forward pass, softmax
   over all classes; returns the top-3 (label, probability) pairs.
5. **Threshold gate** — if `max(prob) < CONFIDENCE_THRESHOLD` (calibrated in
   `ml/src/calibrate_threshold.py`), the API does **not** diagnose. It returns
   `is_uncertain: true` with the Bengali message
   "আমি নিশ্চিত নই — কৃষি কর্মকর্তার সাথে পরামর্শ করুন".
6. **Advice lookup** (`services/advice.py`) — reads `guidelines_bn.json`
   (disease name, symptoms, treatment, organic option, prevention, when to
   call an expert). If `LLM_API_KEY` is configured, an LLM **rewrites only**
   the JSON text in simpler Bangla; any failure silently falls back to the
   static JSON.
7. **Persistence** — scan metadata (timestamp, filename, crop, disease,
   confidence, top-3 JSON, is_uncertain) is saved to SQLite. **The image
   itself is never stored.**
8. **Response** — a single JSON body built by `routers/predict.py`.

## 3. Other endpoints

| Endpoint | Method | Purpose |
| --- | --- | --- |
| `/health` | GET | Liveness + model-loaded flag (used for free-tier keep-alive checks) |
| `/predict` | POST | Full flow above, with rate limiting |
| `/history` | GET | Recent scans (paged) |
| `/history/{id}` | GET | One scan in detail |
| `/history/{id}` | DELETE | Remove one scan |

## 4. Design decisions worth noting

- **Honest uncertainty is a first-class feature.** Lying to a farmer costs a
  real crop; low confidence must lead to human escalation, not a guess.
- **Static-first advice.** The LLM layer is optional decoration on top of a
  reviewed, curated JSON — it can never produce new doses or products.
- **ONNX, not a Python framework, in production.** The backend only needs
  runtime inference; training stays in the `ml/` notebooks on Kaggle/Colab.
- **SQLite first.** Simple, zero-config, enough for the scan-history use case;
  `DATABASE_URL` keeps the door open for Postgres later.
- **Rate limiting + safe errors** on the public endpoints, because free-tier
  hosting is exposed to the whole internet.
