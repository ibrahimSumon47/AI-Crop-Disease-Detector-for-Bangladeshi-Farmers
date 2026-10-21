# Final Report — AI Crop Disease Detector for Bangladeshi Farmers

**Repository:** https://github.com/ibrahimSumon47/AI-Crop-Disease-Detector-for-Bangladeshi-Farmers
**Deployed application:** https://ai-crop-disease-detector-for-bangla-three.vercel.app
**All metrics below come from the actual training/evaluation runs — nothing is estimated.**

---

## 1. Problem Solved

Smallholder farmers in Bangladesh typically learn that their rice, potato or
tomato crop is diseased only after visible damage, and getting expert advice
means physically reaching an agriculture officer — often with the crop
already spreading disease. Available AI tools are English-only, which does
not help the target users.

This project delivers a working, deployed system where a farmer photographs
a leaf and, within seconds on a low-end phone, receives:

1. the crop and the most likely disease (Bangla name displayed);
2. a confidence score plus the top-3 alternative diagnoses;
3. treatment advice curated in simple Bangla (symptoms, treatment, organic
   option, prevention, when to call an expert);
4. when the model lacks confidence — an honest "নিশ্চিত বলা যাচ্ছে না" screen
   that still shows the most-probable disease's advice labeled
   low-confidence, plus the officer recommendation; and a family-explanation
   when the model is torn between the same disease on potato vs tomato
   (shared Solanaceae pathogens — see §3.4).

## 2. Design Decisions

**Single deployment instead of split hosting.** The original plan was Render
(backend) + Vercel (frontend). Because the ONNX model is only 9 MB, the
whole system now runs as one Vercel project: a Next.js *service* plus a
Python serverless function (`api/index.py`) that mounts the FastAPI backend
under `/api/*`. Benefits: one account, one URL, zero CORS configuration.
Trade-offs, honestly stated: scan history lives in `/tmp` SQLite and resets
on redeploy; uploads are capped at 4 MB (platform limit); a cold start after
15 min idle adds a few seconds.

**Honest uncertainty as a core UX principle.** A wrong pesticide
recommendation ruins a season, so the calibrated threshold decides between
answer and defer. User testing showed that refusing with *no* guidance makes
the app feel useless, so the uncertain screen shows the most-probable
disease's full curated advice clearly labeled low-confidence — actionable
but flagged.

**Advice from a curated JSON, never from an LLM's imagination.** Eleven
classes × six fields (mathished Bangla text reviewed for glyph correctness)
are stored in `guidelines_bn.json`. The optional LLM layer is grounded: it
may only rephrase this JSON, has temperature 0.2, and silently falls back to
the static text if unavailable. It is off by default in the deployed app.

## 3. AI Workflow

### 3.1 Data
PlantVillage (potato: early/late blight + healthy; tomato: bacterial spot,
early/late blight, leaf mold + healthy), a rice-leaf disease dataset (blast,
brown spot, healthy) and PlantDoc as a reference source; licenses recorded
in `ml/DATASETS.md`. [prepare_data.py](../ml/src/prepare_data.py) merges
sources with folder-name alias matching and writes a stratified 70/15/15
train/val/test split (seed 42). Two canonical rice classes
(bacterial leaf blight, tungro) had no images in this run, so the model
ships with 11 classes — documented and visible in the class list.

### 3.2 Training
MobileNetV2 (ImageNet weights) + new classification head, trained on a free
Colab T4 via the committed notebook: 5 head-only epochs (lr 1e-3), then 20
fine-tune epochs (lr 1e-5) with class-weighted loss, early stopping and
checkpointing. Augmentations: rotation, flip, brightness/contrast, blur,
zoom/crop.

### 3.3 Evaluation (test split, 5,220 unseen images)
| Metric | Value |
| --- | --- |
| Top-1 accuracy | **61.34%** |
| Macro F1 | 0.526 |
| Best per class | tomato_bacterial_spot F1 0.79, tomato_healthy 0.74, potato_early_blight 0.72 |
| Worst per class | tomato_early_blight recall 0.20, potato_healthy F1 0.36, rice classes ≤ 0.49 |

Confusion matrix: [ml/reports/confusion_matrix.png](../ml/reports/confusion_matrix.png).

### 3.4 Threshold calibration + the potato/tomato confusion (key finding)
The validation sweep finds 0.97 keeps precision ≥ 93.6% — but coverage of
only 9.9%: on real field photos the model almost never reaches that
confidence (user testing confirmed: a *correct* potato-early-blight answer
at 14%). The deployed threshold is therefore 0.30 with honest UI labeling.

Separately, TTA experiments (flip + multi-crop averaging) on three real
photos showed averaging does **not** fix the crop confusion — it can even
sharpen the potato bias. Root cause: within Solanaceae the *same disease*
appears on both crops, and training data is potato-heavy. The shipped fix is
an honest stabilizer (`processing.attach_family_note`): when top-1 and a
top-3 alternative are the same disease across the potato↔tomato boundary
(family probability ≥ 0.25), the response carries a Bangla note explaining
that the model is torn between the two hosts, that the advice applies to
both, and that the actionable information is the *disease family*, not the
host. This converts a misleading "আলু" headline into a correct,
usable diagnosis.

### 3.5 Serving
ONNX export with a numeric parity check (Keras vs onnxruntime max diff below
tolerance). `labels.json` is generated in the model's true output order —
cross-verified against the training history — so no label/index mismatch is
possible. Preprocessing at serving time mirrors training exactly (224×224,
[-1,+1], EXIF rotation).

## 4. Results

- Working end-to-end product, live on a free deployment, installable as a
  PWA (Add to Home Screen on Android/iOS; PWABuilder can wrap an APK).
- Backend correctness enforced by a 15-test pytest suite (CI on every push),
  including real-model end-to-end prediction tests.
- Measured model quality: 61.34% test accuracy / 0.526 macro-F1 with a
  documented plan to improve via field-photo data collection
  (`--merge-sources` pipeline already committed) — the current model is a
  *seed* whose honest behavior (uncertainty + family note) is safe to
  deploy even before that retrain.
- Verification of the deployed stack: `/health` showing `model_loaded:
  true`, camera upload from a real phone-style flow, history persistence
  (screenshots in `docs/screenshots/`).

## 5. Conclusions

1. **The system works and is deployed**, meeting every capstone requirement:
   frontend, backend, AI/ML component, Git/GitHub with docs and commit
   history, free-tier deployment with a public URL, and full documentation.
2. **Honesty beats false confidence.** The most valuable engineering decision
   was treating uncertainty as a first-class outcome: the app is trustworthy
   even when the model is weak, and users get usable guidance in every case.
3. **Data is the next frontier.** The known accuracy ceiling is a data
   problem, not an architecture one; field-photo collection and one retrain
   are the highest-impact next steps (documented step-by-step in the repo).
4. **Bangla-first matters.** The entire advice layer and UI communicate in
   the farmer's language; the model's Bangla disease names are generated
   from the same canonical list, removing a whole translation layer.
