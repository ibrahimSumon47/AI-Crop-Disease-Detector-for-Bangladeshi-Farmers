# ML Workflow — AI Crop Disease Detector

This folder contains everything needed to train, evaluate, calibrate and export
the crop-disease classifier that powers the backend.

## The pipeline at a glance

```
download datasets (Kaggle / Mendeley / GitHub)
        │
        ▼
ml/src/prepare_data.py        → data/processed/{train,val,test}/<class>/
        │                      (stratified 70/15/15, seed=42)
        ▼
ml/src/dataset_stats.py       → ml/reports/class_distribution.png
        │
        ▼
ml/src/train.py               → trained model + ml/reports/history.json
   (two-stage transfer learning: MobileNetV2 + new head)
        │
        ▼
ml/src/evaluate.py            → ml/reports/metrics.md + confusion_matrix.png
        │
        ▼
ml/src/calibrate_threshold.py → ml/reports/threshold.json
        │
        ▼
ml/src/export_onnx.py         → backend/app/data/model.onnx + labels.json
```

## Where to run it

**Recommended: Kaggle or Google Colab (free GPU).** Use
[`ml/notebooks/train_kaggle.ipynb`](notebooks/train_kaggle.ipynb) — it runs the
whole pipeline end to end. Local training works but a laptop CPU will take
hours per epoch.

The target machine where datasets live: attach the dataset as a Kaggle
"Dataset" input, or mount Google Drive in Colab. Output artifacts
(`metrics.md`, `threshold.json`, `labels.json`, `model.onnx`) are small and
should be copied back into this repository (ONNX file also only if < 50 MB).

## Class list (13 classes)

Each supported crop gets its own "healthy" class, so the model learns crop +
disease in a single multi-class head:

- **rice** (ধান): bacterial_leaf_blight, brown_spot, blast, tungro, healthy
- **potato** (আলু): early_blight, late_blight, healthy
- **tomato** (টম্যাটো): bacterial_spot, early_blight, late_blight, leaf_mold, healthy

This list is defined in [`ml/src/classes.py`](src/classes.py) as the single
source of truth (order, index mapping, human/Bangla names).

## Files in this folder

| File | Purpose |
| --- | --- |
| `requirements.txt` | Python dependencies for training |
| `DATASETS.md` | Dataset names, sources, links and licenses |
| `src/classes.py` | Canonical class list + index mappings |
| `src/download_data.md` | Step-by-step dataset download instructions |
| `src/prepare_data.py` | Merge + stratified split into `data/processed/` |
| `src/dataset_stats.py` | Class counts per split + bar chart |
| `src/augmentations.py` | Training-time augmentation pipeline |
| `src/model.py` | MobileNetV2 transfer-learning model definition |
| `src/train.py` | Two-stage training with class weights + early stopping |
| `src/evaluate.py` | Accuracy / per-class metrics / confusion matrix / field test |
| `src/calibrate_threshold.py` | Pick the confidence threshold on validation data |
| `src/export_onnx.py` | Export to ONNX + numerical parity check |
| `notebooks/train_kaggle.ipynb` | One-click notebook for Kaggle/Colab |
| `reports/` | Output artifacts (metrics, charts, threshold) |

## Honesty rules

- Metrics reported in `ml/reports/` come only from real training runs.
- If a dataset cannot be downloaded (no Kaggle token etc.), write the script
  anyway and document exactly the steps the human must run — never fabricate
  results.
