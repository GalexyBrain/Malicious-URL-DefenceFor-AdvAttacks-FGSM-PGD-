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
source .venv/bin/activate
pip install -U pip
