# Project Proposal — AI Crop Disease Detector for Bangladeshi Farmers

**Student name:** ibrahimSumon47
**Repository:** https://github.com/ibrahimSumon47/AI-Crop-Disease-Detector-for-Bangladeshi-Farmers
**Live app:** https://ai-crop-disease-detector-for-bangla-three.vercel.app

---

## 1. Problem Statement

Bangladeshi farmers lose a large share of rice, potato and tomato yield every
season to leaf diseases (early/late blight, blast, brown spot, bacterial spot,
leaf mold). Two practical barriers make it worse:

1. **No timely diagnosis.** The nearest agriculture officer may be far away;
   by the time a farmer gets expert advice, the disease has spread.
2. **Language barrier.** Existing plant-disease tools are English-first, so
   their output is not actionable in the field.

**Goal:** a farmer points a phone camera at a leaf and, within seconds, gets
(1) the crop and the most likely disease, (2) a confidence score with top-3
alternatives, (3) treatment advice written in simple Bangla, and (4) — when
the model is not confident — an honest "not sure" answer with a best-guess
advice labeled low-confidence instead of a dangerous guess.

## 2. Proposed Solution

A mobile-first web app (installable as a PWA on Android/iPhone):

- **Frontend:** Next.js 16 + Tailwind CSS 4, Bangla-first UI (Noto Sans
  Bengali), camera capture + upload, result card, uncertain screen, scan
  history page.
- **Backend:** FastAPI serving an ONNX model — endpoints `/predict` (upload →
  diagnosis JSON), `/history` (scan log), `/health`. SQLite stores scan
  history. A curated Bangla guideline JSON provides expert advice per class;
  an optional grounded LLM layer can rephrase it (off by default).
- **AI/ML component:** a computer-vision classifier built by transfer
  learning — MobileNetV2 pretrained on ImageNet, fine-tuned in two stages on
  ~34,800 labeled leaf images, exported to ONNX (9 MB) and served with CPU
  onnxruntime. A calibrated confidence threshold decides when to answer and
  when to say "consult the officer".

## 3. AI Approach

| Step | Method |
| --- | --- |
| Data | PlantVillage (potato/tomato), a rice-leaf dataset, PlantDoc references; stratified 70/15/15 split, seed 42 |
| Preprocessing | Resize 224×224, normalize to [-1, +1] — identical at training and serving |
| Model | MobileNetV2 (ImageNet) + new dense head; two-stage training (head 5 epochs lr 1e-3, fine-tune 20 epochs lr 1e-5), class-weighted loss, early stopping |
| Evaluation | Test-split accuracy, per-class precision/recall/F1, confusion matrix |
| Calibration | Threshold sweep on validation (precision >= 0.90 objective); deployment threshold selected for real field photos (0.30) with UI-side honest labeling |
| Export | Keras → ONNX with numerical parity check against onnxruntime |
| Known limitation | Lab-photo training causes low confidence on field photos; measured and reported (test acc 61.34%, macro F1 0.526). Retraining with field photos is the documented next step. |

## 4. Tech Stack

| Layer | Technology |
| --- | --- |
| Frontend | Next.js 16 (TypeScript, App Router), Tailwind CSS 4, PWA manifest |
| Backend | Python 3.12, FastAPI, SQLAlchemy + SQLite, onnxruntime, Pillow |
| ML | TensorFlow/Keras (training on Colab GPU), MobileNetV2, ONNX export |
| Deployment | Single Vercel project — Next.js service + Python serverless function (same-origin `/api/*`), free tier |
| CI | GitHub Actions — backend pytest suite (15 tests) on every push |
| Version control | Git on GitHub, linear history on `main` |

## 5. Deliverables Map

| Required deliverable | Where |
| --- | --- |
| Proposal (this document) | `docs/proposal.md` |
| Code repository (frontend + backend + ML) | GitHub repo (see structure in README) |
| Deployed application | the Vercel URL above |
| README with setup, model description, usage, screenshots | `README.md` |
| Final report (3–5 pages) | `docs/final-report.md` |
