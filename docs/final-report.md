# Final Report — AI Crop Disease Detector for Bangladeshi Farmers

**Repository:** https://github.com/ibrahimSumon47/AI-Crop-Disease-Detector-for-Bangladeshi-Farmers
**Deployed application:** https://ai-crop-disease-detector-for-bangla-three.vercel.app
**All metrics below come from the actual evaluation runs in `ml/reports/` — nothing is estimated.**

---

## 1. Problem Solved

Smallholder farmers in Bangladesh usually discover that a rice, potato, tomato
or other crop is diseased only after visible damage. Getting expert advice
means reaching an agriculture officer, who may be far away, and by then the
disease may have spread. Most available plant-disease tools are English-only,
so their output is hard to act on in the field.

This project delivers a deployed system. A farmer photographs a leaf and,
within seconds on a low-end phone, receives:

- the crop and the most likely disease, with the Bangla name;
- a confidence score and the top-3 alternative diagnoses;
- treatment advice written in simple Bangla (symptoms, treatment, organic
  option, prevention, and when to call an officer);
- an honest "নিশ্চিত বলা যাচ্ছে না" (not certain) screen when the model is not
  confident, which still shows the most-probable disease's advice labelled
  low-confidence;
- a clear message instead of a diagnosis when the photo does not show a plant
  leaf; blurred or very dark photos still get a diagnosis, with a quality
  warning.

## 2. Design Decisions

### 2.1 One deployment instead of split hosting

The original plan was a separate backend host plus Vercel for the frontend.
Because the ONNX model is small, the whole system now runs as one Vercel
project: a Next.js service and a Python serverless function (`api/index.py`)
that mounts the FastAPI backend under `/api/*`. This gives one account, one URL
and no CORS configuration. Trade-offs: scan history lives in SQLite under
`/tmp` and resets on redeploy, uploads are capped at 4 MB on that deployment
(the app default is 5 MB; Vercel limits request bodies to about 4.5 MB), and a
cold start after idle adds a few seconds.

### 2.2 Honest uncertainty as a core principle

A wrong pesticide recommendation can ruin a season, so the calibrated
confidence threshold decides between answering and deferring. Refusing with no
guidance made the app feel useless in testing, so the uncertain screen still
gives the most-probable disease's advice, flagged as low-confidence.

### 2.3 Advice from a curated file, not from a language model

Treatment advice is stored in `backend/app/data/guidelines_bn.json`, written in
Bangla and reviewed for glyph correctness. An optional LLM layer may only
rephrase this file. It runs at temperature 0.2, falls back to the static text
on any error, and is off by default in the deployed app.

### 2.4 A plant gate in front of the classifier

A separate binary model (`plant_guard`) first decides whether the photo shows a
leaf. Non-leaf photos get clear Bangla guidance before any disease inference
runs. Section 4.3 measures its effect.

### 2.5 Cross-crop family note

Potato and tomato share pathogens (early blight, late blight). When the top-1
disease and a top-3 alternative are the same disease across that boundary, the
response carries a Bangla note saying the model is torn between the two hosts,
that the advice applies to both, and that the useful information is the disease
family rather than the host.

## 3. AI Workflow

### 3.1 Data

The classifier covers 61 classes across potato, tomato, rice, maize, grape,
mango, chilli, cucumber, cauliflower, okra and papaya, including healthy
classes. Sources and licences are recorded in `ml/DATASETS.md`.
`ml/src/prepare_data.py` merges sources with folder-name alias matching and
writes a stratified 70/15/15 train/validation/test split with seed 42. The test
split has 14,641 images.

Known data gap: the training photos are mostly lab or close-up images, while
farmers photograph leaves in fields. This is the main reason for the accuracy
ceiling in Section 4.

### 3.2 Classifier training

MobileNetV2 with ImageNet weights and a new classification head, trained on a
free Colab GPU through `ml/notebooks/train_kaggle.ipynb`. Stage 1 trains only
the head for 5 epochs at learning rate 1e-3. Stage 2 fine-tunes for up to 20
epochs at learning rate 1e-5, with class-weighted loss, early stopping and
checkpointing. Stage 1 ran 5 epochs and stage 2 ran 11 before early stopping.
Preprocessing is 224×224 with inputs scaled to [-1, +1], identical at training
and serving time.

### 3.3 Plant gate

A small binary classifier (`plant_guard`, 19 features) separates leaves from
non-leaf photos. It was trained on synthetic photo-junk, leaf-free background
crops and partial-leaf edge windows from the real dataset (30%
camera-degraded), plus one user-submitted non-leaf photo. Held-out validation:
leaf recall 95.7%, non-leaf detection 95.2%, accuracy 95.5%. Its stated
limitation is that complex real non-plant scenes such as people or landscapes
are not in the negative set.

### 3.4 Calibration and the deployment threshold

The validation sweep (`ml/reports/threshold.json`) shows that a high threshold
of 0.97 keeps precision above 93%, but it answers only about 4% of images. The
deployed threshold is 0.30. At this value the sweep covers about 68% of
validation images at roughly 53% precision, and the UI labels lower-confidence
answers honestly. The value is set in `backend/app/config.py`, so the deployed
default does not depend on environment variables.

### 3.5 Serving and export

The model is exported to ONNX. The parity check against onnxruntime gives a
maximum absolute difference of 8.3e-8 for the plant gate. Labels are read from
`backend/app/data/labels.json`, ordered to match the model output. A serving
fix removed a double softmax: the exported graph already outputs probabilities,
so applying softmax again flattened the confidence scores.

## 4. Results

### 4.1 Classifier on the test split (14,641 unseen images)

| Metric | Value |
| --- | --- |
| Top-1 accuracy | **42.80%** |
| Macro F1 | 0.299 |
| Classes | 61 |

A 61-way problem on mostly lab-style photos is much harder than the 11-class
earlier model (61.34% on 5,220 images). The numbers are not directly
comparable, because the class set, the test set and the data mix all changed.
The earlier model is not the shipped model.

### 4.2 Per-class highlights

- Strongest: chilli_curl_virus F1 0.68, maize_streak_virus F1 0.67,
  maize_common_rust F1 0.76, mango_cutting_weevil F1 0.76.
- Weakest: cucumber_anthracnose and cucumber_downy_mildew (F1 0.00),
  grape_bacterial_leaf_spot (F1 0.00), okra_leaf_curl_virus (F1 0.00), and
  several classes with 24 to 35 test images.
- Potato early blight F1 0.43, late blight F1 0.34; tomato late blight F1 0.48.

Small test counts make per-class F1 noisy. Per-class results are a guide to
where the data is thin, not a precise ranking.

### 4.3 Effect of the plant gate (end-to-end, full test split)

| Metric | Before gate retrain | After gate retrain |
| --- | --- | --- |
| Gate leaf pass rate | 98.91% | 98.58% |
| Leaf refusals | 159 / 14,641 | 208 / 14,641 |
| Junk photos passed into the diagnosis | 28.33% | 14.67% |
| Top-1 accuracy, all leaves | 44.07% | 43.83% |
| Top-1 accuracy, gate-passed leaves | 44.56% | 44.46% |

Retraining the gate on harder negatives roughly halved the junk that reached
diagnosis, for a cost of about 0.3 percentage points more false refusals on
real leaves and about 0.2 points of top-1 accuracy. This is a trade-off the
project accepts, because a wrong diagnosis of a non-leaf photo is worse than a
refusal.

Both runs use the same test split and seed (4242). The earlier small-sample run
(244 leaves, 300 junk) shows the same direction: junk passed fell from 30.7% to
14.7%.

### 4.4 Uncertainty and the family note in practice

Below the 0.30 threshold, the app shows the honest uncertain screen with the
best-guess advice labelled low-confidence. When the potato/tomato family note
applies, it appears alongside the result. Both are covered by backend tests
(`test_family_note.py` and `test_api.py`).

### 4.5 Engineering verification

- Backend test suite: 39 pytest tests, run in GitHub Actions on every push to
  `main`.
- Deployed application: Vercel, with `/health` reporting `model_loaded: true`.
- Chat endpoint: served at `/api/chat` and verified in the browser; it answers
  across all 61 curated classes from the guideline JSON, with no external API
  key.
- Git: `main` is in sync with `origin/main` on GitHub, and the repository
  history shows the incremental changes described in this report.

## 5. Limitations

- Accuracy is limited: 42.8% top-1 on the 61-class test split. Many classes
  have few test images, and several have F1 near zero.
- Training data is mostly lab-style photos. Field photos are the main gap, and
  collecting them is the highest-impact next step.
- The plant gate was not tested on complex real non-plant scenes such as people
  or landscapes.
- Scan history is stored in SQLite under `/tmp` on Vercel and resets on
  redeploy.
- Uploads are limited to 5 MB by the app, and 4 MB on the Vercel deployment
  because the platform caps request bodies at about 4.5 MB. The LLM advice
  layer is off by default.

## 6. Conclusions

- The system is deployed and working end to end: frontend, backend, a trained
  computer-vision classifier, a plant gate, Bangla advice, scan history and a
  PWA, with a public URL.
- Honesty is the central design choice. The app says when it is not sure, and
  it still gives usable guidance in those cases.
- The measured accuracy is modest and is reported as measured. The main limit
  is data, not the architecture. Field-photo collection and a retrain are the
  documented next steps.
- Measurement changed decisions: the threshold was set from validation data,
  the gate was retrained after the before/after study showed junk leaking
  through, and a double-softmax bug was found and fixed.

## 7. System Architecture

A farmer's photo goes through these steps:

1. **Upload and validation:** the file must be JPG or PNG, at most 5 MB, and at
   least 64×64 pixels.
2. **Plant gate:** a trained classifier decides whether the photo shows a plant
   leaf. Non-leaf photos get a clear Bangla message and no diagnosis.
3. **Photo-quality check:** blurry or very dark photos still get a diagnosis,
   with a Bangla tip to retake the photo.
4. **Preprocessing:** resize to 224×224, apply EXIF rotation, and scale to
   [-1, +1], matching training.
5. **ONNX inference** on the CPU, producing class probabilities.
6. **Calibrated threshold (0.30):** above it, the app returns the diagnosis with
   Bangla advice. Below it, the app shows the uncertain screen with the
   best-guess advice labelled low-confidence and a note to consult the
   agriculture officer.
7. **Response:** the scan is saved to history, and the JSON response returns.

The frontend is a Next.js app and the backend is a FastAPI app, both deployed
on one Vercel project under the same origin.

## 8. Data Sources

The training data comes from 11 public datasets, merged into canonical class
folders with a stratified 70/15/15 split (seed 42): PlantVillage (potato,
tomato, maize, grape), a rice leaf disease dataset including tungro, RiceLeafBD
(field photos from Bangladesh), a chilli leaf disease dataset from Bangladesh, a
Tanzanian maize dataset, the NGLD grape dataset, a Bangladeshi cauliflower
dataset, a cucumber disease dataset, BDPapayaLeaf, MangoLeafBD and an Okra
disease dataset. Each source's licence is recorded in `ml/DATASETS.md`. Some
licences are non-commercial only, so this project is suitable for research and
education, and a commercial product would need permission or different data.

## 9. Advice Review

The Bangla treatment advice is stored in
`backend/app/data/guidelines_bn.json`. Of the 61 entries, 11 were reviewed by an
agriculture officer. The other 50 were written carefully but are still marked
as drafts in the data file, until a Bangla agriculture officer verifies them.
Pesticide names and doses must be checked against local regulations before
commercial use.

## 10. Future Work

- Collect 50 to 100 real field photos per class, retrain with
  `--merge-sources`, and check the 50 draft advice entries with an agriculture
  officer.
- Add eggplant (brinjal), the first unsupported crop, using the dataset
  candidates in `ml/src/download_data.md`.
- Consider an opt-in feature to store anonymised upload thumbnails, to build a
  real Bangladeshi field dataset over time.
- Move scan history to a persistent database, since the current `/tmp` store
  resets on redeploy.
