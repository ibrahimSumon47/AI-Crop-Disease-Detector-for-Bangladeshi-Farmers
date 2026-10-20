# Datasets — sources, licenses and usage

The model is trained on three public datasets covering the 13 target classes
(rice / potato / tomato diseases + healthy).

| Dataset | Contents used | Class mapping | Source | License |
| --- | --- | --- | --- | --- |
| **PlantVillage** (Kaggle mirror `abdallahalidev/plantvillage-dataset`) | Potato Early Blight, Potato Late Blight, Potato Healthy (~2,200 images); Tomato Bacterial Spot, Tomato Early Blight, Tomato Late Blight, Tomato Leaf Mold, Tomato Healthy | `potato_*`, `tomato_*` | Kaggle: <https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset> — original: <https://plantvillage.psu.edu/> | **CC-BY-NC-SA-4.0** (verified 2026-10-20 from the Kaggle dataset page). Non-commercial use with attribution — fine for this educational capstone; do NOT sell a commercial product with this data. |
| **Rice Leaf Diseases** (`shayanriyaz/riceleafs`) | Bacterial Leaf Blight, Brown Spot, Blast, Tungro + Healthy (the dataset ships `RiceLeafs` with healthy folders) | `rice_bacterial_leaf_blight`, `rice_brown_spot`, `rice_blast`, `rice_tungro`, `rice_healthy` | Kaggle: <https://www.kaggle.com/datasets/shayanriyaz/riceleafs> | **CC0-1.0** (verified 2026-10-20 from the Kaggle dataset page) — public domain, no restriction. |
| **PlantDoc Dataset** (OPTIONAL — field photos) | Extra field-condition photos for all classes | blend-in field shots | Mendeley DOI 10.5281/zenodo.4082314, GitHub <https://github.com/pratikm141/PlantDoc> | CC BY 4.0 (paper: Singh et al., 2019 — DOI 10.1145/3371150.3371168). Clone often fails; training works fine without it because riceleafs includes healthy rice leaves. |

## Notes on licenses

- **PlantVillage:** the dataset has been redistributed widely on Kaggle. The
  original PlantVillage images are open-access; the commonly referenced
  license is CC BY-SA. Verify the Kaggle mirror's stated license before
  redistributing any subset.
- **PlantDoc:** unambiguous — CC BY 4.0 (authors encourage reuse with
  attribution). Cite the paper.
- **Rice leaf datasets:** several near-duplicate versions exist. Pick one,
  record its exact Kaggle URL and license line in the training notes.
- **Attribution requirement:** this project (academic capstone) must cite all
  three datasets in the final report and model card. See
  `docs/model_card.md` after training.

## Images are NOT committed

Raw datasets stay out of git (see `.gitignore`: `/data/`, `ml/data/` is
ignored). Only `prepare_data.py` reads them and writes the stratified split
to `data/processed/`.
