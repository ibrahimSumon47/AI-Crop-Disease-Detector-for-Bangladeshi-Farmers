# How to download the datasets — step by step

Because Kaggle downloads need an account token, this step is documented for
the **human** to run once. Everything is free.

## 1. Create the raw dataset folders

```
ml/data/raw/plantvillage/        <- PlantVillage images (already split by class folder)
ml/data/raw/rice_leaf/           <- rice dataset (one folder per class)
ml/data/raw/plantdoc/            <- PlantDoc (a CSV + image files, or folder structure)
```

`prepare_data.py` (T2.3b) reads from these three folders and writes the
merged, stratified split to `data/processed/{train,val,test}/<class>/`.

## 2. Download PlantVillage (potato + tomato)

Option A — Kaggle (needs a free Kaggle account):

1. Sign up at <https://www.kaggle.com>.
2. Go to <https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset>.
3. Click **Download** (accept the rules). You get a large `.zip`.
4. Unzip; from inside it, copy the folders for the 8 potato/tomato classes
   we use into `ml/data/raw/plantvillage/`:
   - `Potato___Early_blight`, `Potato___Late_blight`, `Potato___healthy`
   - `Tomato_Bacterial_spot`, `Tomato_Early_blight`, `Tomato_Late_blight`,
     `Tomato_Leaf_Mold`, `Tomato_healthy`

Option B — command line with the Kaggle CLI (`pip install kaggle`):

1. Kaggle -> Account -> API -> Create New API Token; you get `kaggle.json`.
2. Put it at `~/.kaggle/kaggle.json` (Linux/Mac) or
   `C:/Users/<you>/.kaggle/kaggle.json` (Windows).
3. Run:

   ```
   kaggle datasets download -d abdallahalidev/plantvillage-dataset -p ml/data/raw
   cd ml/data/raw && unzip plantvillage-dataset.zip
   ```

## 3. Download the rice leaf dataset

1. Go to one of the Kaggle mirrors, for example
   <https://www.kaggle.com/datasets/shayanriyaz/riceleafs> or search
   "rice leaf disease" on Kaggle datasets.
2. Download and unzip into `ml/data/raw/rice_leaf/` so that folders exist
   like: `bacterial_leaf_blight/`, `brown_spot/`, `blast/`, `tungro/`,
   and if available `healthy/`.
3. Record the exact URL you used in `ml/DATASETS.md`.

## 4. Download PlantDoc (field photos, rice healthy etc.)

Option A — Mendeley/Zenodo (direct download):

1. Go to <https://zenodo.org/record/4082314> (PlantDoc) or the GitHub repo
   <https://github.com/pratikm141/PlantDoc>.
2. PlantDoc ships as a CSV (`plantdoc.csv` or similar) with image URLs plus
   a downloadable archive. If you use the CSV variant, run
   `python ml/src/prepare_data.py --download-plantdoc` to fetch the images
   (uses only public URLs).
3. Unzip into `ml/data/raw/plantdoc/` keeping its class folder structure.

## 5. Sanity check

After downloading, ensure each of the folders contains real image files
(expect thousands of JPEG/PNGs; PlantVillage alone is ~55k images). Then run:

```
python ml/src/prepare_data.py
python ml/src/dataset_stats.py
```

and check `ml/reports/class_distribution.png` before training.

## 6. Also do NOT commit

Raw datasets and `data/processed/` are git-ignored. Only small report
artifacts get committed.

## Field-test photos (`data/field_test/`)

Later, 20-30 real photos taken with an actual phone are needed to measure
the lab-vs-field gap (task T5.3). Put them into `data/field_test/<class>/`
(following the same class naming) and run
`python ml/src/evaluate.py --field-test data/field_test`.
