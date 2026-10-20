# Datasets — sources, licenses and usage

The model is trained on three public datasets covering the 13 target classes
(rice / potato / tomato diseases + healthy).

| Dataset | Contents used | Class mapping | Source | License |
| --- | --- | --- | --- | --- |
| **PlantVillage** (spoonm/plantvillage on Kaggle, or the original Penn State release) | Potato Early Blight, Potato Late Blight, Potato Healthy (~2,200 images); Tomato Bacterial Spot, Tomato Early Blight, Tomato Late Blight, Tomato Leaf Mold, Tomato Healthy (subsets of the ~18k tomato images) | `potato_*`, `tomato_*` | Kaggle: <https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset> (mirror) — original: <https://plantvillage.psu.edu/> | CC BY-SA 4.0 (per the PlantVillage public dataset page documentation; the original images are color lab photos) |
| **Rice Leaf Diseases** (UCI/Mendeley, often distributed as `rice-leaf-diseases` on Kaggle) | Bacterial Leaf Blight, Brown Spot, Leaf Blast / Blast, Tungro — plus Healthy if the chosen version includes it (most versions do not; healthy rice leaf images are added from the PlantDoc rice class) | `rice_bacterial_leaf_blight`, `rice_brown_spot`, `rice_blast`, `rice_tungro` | Kaggle: <https://www.kaggle.com/datasets/shayanriyaz/riceleafs> (or Mendeley DOI 10.2478/acph-2020-0002 version) | Research use / open academic release — verify the exact license on the chosen dataset page before publishing trained artifacts publicly |
| **PlantDoc Dataset** (photos of real field conditions, phone-style photos) | Rice healthy/lesion photos, extra field-condition potato & tomato images, useful as extra validation data / field-domain blend-in | `rice_healthy` + blend-in field shots for other classes | GitHub: <https://github.com/pratikm141/PlantDoc> (official repo with CSV of image URLs) — Mendeley DOI 10.5281/zenodo.4082314 | CC BY 4.0 (paper: "PlantDoc: A Dataset for Visual Plant Disease Detection", Singh et al., 2019 — DOI 10.1145/3371150.3371168, CC BY 4.0) |

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
