# Malicious URL Defence for Adversarial Attacks (FGSM & PGD)

Research-focused code for **malicious URL classification** and **robustness evaluation** under gradient-based adversarial attacks—primarily **FGSM** and **PGD**—with feature-centric defensive training and evaluation outputs.

> ⚠️ **Ethical use notice**
> This repository is intended for **defensive security research**, robustness testing, and academic experimentation on **your own data/models**.  
> Do **not** use it to bypass real-world security systems or to generate evasive malicious content.

---

## What this repo does

- Trains **baseline models** for malicious URL classification:
  - Traditional ML models (scripted pipeline)
  - Deep Learning models (scripted pipeline)
- Builds **defensive feature datasets** and trains **defensive models** (including a LightGBM-based defensive model).
- Runs **adversarial robustness evaluation** using:
  - **FGSM** (single-step gradient sign perturbation)
  - **PGD** (iterative projected gradient perturbation)
- Stores experiment outputs in organized `results_*` directories.

---

## Repository layout (high level)

### Training & experiments
- `TrainAllModels_ML.py`  
  Train baseline **machine learning** models.

- `TrainAllModels_DL.py`  
  Train baseline **deep learning** models.

- `TrainAllFeatureDefensiveModels.py`  
  Train models using **defensive/robust feature sets**.

- `train_DefensiveLightGBM.py`  
  Train a **defensive LightGBM** model (and/or run LGBM-focused defence experiments).

### Adversarial evaluation
- `attackModels.py`  
  Runs adversarial evaluation (FGSM/PGD) against trained models for robustness testing.

### Dataset preparation / cleaning
- `remove_dups_defensive.py`  
  Removes duplicates / performs dataset cleanup for defensive workflows.

- `prep_defensive_dataset_features.py`  
  Feature extraction / transformation for defensive dataset creation.

- `prep_defence_dataset_features_for_training.py`  
  Prepares the defensive-feature dataset specifically for training.

### Results folders
- `results_base3/`
- `results_3class/`
- `results_3class_plus_detector/`
- `results_defence_features_3class/`
- `results_defense_feat_stream/`
- `results_defense_features/Defense-LGBM/`
- `results_evaluation/`

These folders typically contain saved metrics, logs, predictions, and/or plots produced by the above scripts.

---

## Setup

### 1) Create an environment
Use Python **3.9+** (recommended). Example:

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -U pip
````

### 2) Install dependencies

This repo does not currently expose a pinned `requirements.txt` in the root, so install what your scripts import (common stacks include `numpy`, `pandas`, `scikit-learn`, `lightgbm`, and a DL framework like `tensorflow` or `torch`).

Recommended approach:

1. Try running a training script once.
2. Install missing imports as they appear.

Example:

```bash
pip install numpy pandas scikit-learn lightgbm matplotlib tqdm
```

> Tip: If you want reproducibility, add a `requirements.txt` after a successful run:
> `pip freeze > requirements.txt`

---

## Data

Bring your own dataset (CSV or similar) containing:

* A URL field (e.g., `url`)
* A label field (e.g., `label`)

Because datasets vary, check the top of each script for:

* Input file path(s)
* Column names
* Label mapping (binary vs multi-class, “3-class” variants, etc.)

A common workflow is:

1. Clean/deduplicate data
2. Extract features for defensive training
3. Train baseline and defensive models
4. Run FGSM/PGD evaluation

---

## Typical workflow

### A) Prepare / clean dataset

```bash
python remove_dups_defensive.py
```

### B) Build defensive feature dataset

```bash
python prep_defensive_dataset_features.py
python prep_defence_dataset_features_for_training.py
```

### C) Train baseline models

```bash
python TrainAllModels_ML.py
python TrainAllModels_DL.py
```

### D) Train defensive models

```bash
python TrainAllFeatureDefensiveModels.py
python train_DefensiveLightGBM.py
```

### E) Evaluate robustness under FGSM/PGD

```bash
python attackModels.py
```

Outputs should appear under one or more `results_*` folders (depending on the script/config you run).

---

## Understanding the “3-class” and “plus_detector” outputs

You’ll see multiple results directories, including:

* **`results_3class/`**: results for a 3-class classification setting
* **`results_3class_plus_detector/`**: results where an additional *detector* component is evaluated (e.g., to flag adversarial/shifted samples)
* **`results_defence_features_3class/`**: results for 3-class with defensive features/defence pipeline
* **`results_evaluation/`**: consolidated evaluation outputs

Exact meaning depends on your label mapping and config inside scripts.

---

## Reproducibility tips

For consistent results:

* Fix random seeds in Python / NumPy / your DL framework
* Log:

  * dataset version/hash
  * feature extraction config
  * model hyperparameters
  * attack parameters (FGSM/PGD strength and iterations)
* Save trained model artifacts and configs alongside outputs

---

## Safety & responsible disclosure

If you discover a weakness or a method that significantly reduces detection accuracy:

* Validate on controlled, synthetic, or permissioned datasets
* Avoid sharing exploit-like details that enable real-world abuse
* Prefer reporting defensively (mitigations, robustness improvements, evaluation evidence)

---

## Contributing

PRs are welcome—especially for:

* Adding a `requirements.txt`
* Adding a single entry-point runner (CLI) and consistent config files
* Documenting dataset format + sample config
* Unit tests for feature extraction and evaluation pipelines
