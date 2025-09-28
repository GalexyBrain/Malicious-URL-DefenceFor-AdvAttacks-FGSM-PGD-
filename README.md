# Malicious URL Detection — Robustness, Attacks (FGSM/PGD), and Defenses

A full pipeline for **malicious URL detection** with **classical ML & deep learning**, **white-box/black-box adversarial attacks (FGSM, PGD)**, and **two defense modes**:
1) **Two-stage gating** (adversarial detector → base classifier), and  
2) **Three-class “all-in-one”** models (benign / malicious / adversarial).

This repository contains training/evaluation scripts and several pre-generated **results folders** for fast inspection of outcomes.

---

## 📦 What’s in this repository

> High-level folders & scripts (alphabetical)

- `results_base3/` – Results for **three-class** (benign/malicious/adversarial) models.
- `results_defence_features_3class/` – End-to-end **3-class defense** evaluations (detector + 3-class).
- `results_defense_feat_stream/` – **Two-stage defense** evaluations on mixed NAT+ADV streams.
- `results_defense_features/Defense-LGBM/` – Metrics/plots for the **adversarial detector** (feature-space; LightGBM, etc.).
- `results_evaluation/` – **Adversarial evaluations** of trained binary models: NAT / FGSM / PGD.
- `.gitignore`
- `TrainAllFeatureDefensiveModels.py` – Train **three-class** models (0=benign, 1=malicious, 2=adversarial).
- `TrainAllModels_DL.py` – Train **deep learning** models (tabular + char-level text).
- `TrainAllModels_ML.py` – Train **classical ML** models on features.
- `attackModels.py` – Run **FGSM/PGD** robustness evaluations (white-box for DL/linear; **black-box transfer via surrogate** for trees).
- `defence_FeatureModels.py` – **Two-stage defense** evaluation (detector → base binary classifier).
- `defence_FeatureModels3Class.py` – **3-class defense** evaluation (detector + 3-class router).
- `prep_defence_dataset_features_for_training.py` – Build **balanced 3-class** training set from NAT/ADV pool.
- `prep_defensive_dataset_features.py` – Generate **adversarial feature** dataset to train detector (NAT vs ADV).
- `remove_dups_defensive.py` – De-duplicate rows in the **defense** dataset.
- `train_DefensiveLightGBM.py` – Train and calibrate the **feature-space adversarial detector** (LightGBM/XGBoost/LR candidates).

> **Note:** Feature extraction / dataset merge scripts are **not included here**. See “Data & prerequisites” for expected inputs.

---

## 🧠 End-to-end pipeline (quick start)

1) **Prepare data** (outside this repo; see “Data & prerequisites”):
   - `features_extracted.csv` (tabular features + label).
   - `merged_url_dataset.csv` (raw `url,label`) if you plan to train char-level models.

2) **Train base models**  
   - Classical ML (tabular):  
     ```bash
     python TrainAllModels_ML.py
     ```
   - Deep learning (tabular + URL text):  
     ```bash
     python TrainAllModels_DL.py
     ```

3) **Craft the defense datasets** (feature-space):
   ```bash
   # NAT vs ADV feature dataset for detector training
   python prep_defensive_dataset_features.py

   # Optional: ensure uniqueness
   python remove_dups_defensive.py

   # Strictly balanced 3-class dataset for all-in-one models (0/1/2)
   python prep_defence_dataset_features_for_training.py
````

4. **Train defenses**

   ```bash
   # Train adversarial detector (feature-space) and pick the best candidate
   python train_DefensiveLightGBM.py

   # Train 3-class models (benign / malicious / adversarial)
   python TrainAllFeatureDefensiveModels.py
   ```

5. **Adversarial evaluation (FGSM/PGD) for binary models**

   ```bash
   python attackModels.py
   ```

6. **Defense evaluations**

   ```bash
   # Two-stage defense (Detector -> Binary Base Model)
   python defence_FeatureModels.py

   # 3-class defense (Detector + 3-class router)
   python defence_FeatureModels3Class.py
   ```

---

## 📁 Outputs & directory contract (exact names)

> This section lists **which script creates which folders/files**.
> Folder names are **exact**; file names are representative and stable.

### 1) `TrainAllModels_ML.py`  → classical ML on tabular features

**Creates:**

* `models/`

  * `_global/`

    * `scaler.joblib` – StandardScaler fit on **train only**.
    * `feature_columns.json` – Ordered list of used columns.
  * `<ModelName>/`

    * `model.joblib`
* `results/`

  * `_global/` – data quality & drift reports, class stats, outliers, class weights.

    * e.g., `data_quality.json`, `label_stats.json`, `drift_report.csv/json`, `outliers.json`, `class_weights.json`
  * `<ModelName>/`

    * `metrics_train.json`, `metrics_val.json`, `metrics_test.json`
    * `metrics_test_at_opt_threshold.json` (e.g., best F1 from validation)
    * `thresholds_val.csv`
    * `confusion_matrix.png` (+ sometimes `.json`)
    * `roc_curve.png`, `pr_curve.png`
    * `calibration.png`, `calibration_stats.txt` (Brier, ECE)
    * `feature_importance.csv`, `feature_importance.png` (for trees/linear)
    * `generalization_gap.json`
  * `_combined/` (cross-model comparisons)

    * `combined_metrics_test.csv`, `combined_metrics_test.json`
    * `combined_metrics_bar_test.png`
    * `combined_roc_test.png`, `combined_pr_test.png`, `combined_calibration_test.png`
    * `combined_feature_importance_raw.csv`, `combined_feature_importance_normalized.csv`
    * `combined_feature_importance_heatmap_all.png`, `combined_feature_importance_topN.png`

### 2) `TrainAllModels_DL.py`  → deep learning (tabular + char-level URL)

**Creates:**

* `models/`

  * `_global_dl/`

    * `scaler.joblib`, `feature_columns.json` (tabular DL)
    * `split_indices.json` (reproducibility)
  * `_global/`

    * `url_tokenizer.json` (char-level models)
  * `<ModelName>/`

    * `model.pt`
* `results/`

  * `<ModelName>/` – same artifact types as ML (metrics/plots), feature importance via permutation.
  * `_combined/` – duplicate of ML combined artifacts but suffixed with `_DL`:

    * `combined_metrics_test_DL.csv`, `combined_roc_test_DL.png`, etc.

### 3) `attackModels.py`  → FGSM/PGD adversarial evaluation (binary models)

**Creates:**

* `results_evaluation/`

  * `<ModelName>/`

    * `adv_NAT/`, `adv_FGSM/`, `adv_PGD/`

      * `metrics.json`, `classification_report.txt`
      * `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`
      * `attack_trace.json` (before/after vectors and flips; PGD step snapshots)
  * `_metrics_all.json` – flat list of all (model, attack) metric rows
  * `_summary_global_metrics.json` – grouped means (NAT/FGSM/PGD)
  * `_combined/`

    * `combined_NAT.png`, `combined_FGSM.png`, `combined_PGD.png`

> **Methodology:**
> • White-box for differentiable models (PyTorch DL-MLP) and linear sklearn (e.g., LR/SVC).
> • **Black-box transfer** for non-differentiable trees (RF, LightGBM, XGBoost, etc.) via a **surrogate MLP** trained to mimic the target; adversarial examples are crafted on the surrogate and transferred to the target.

### 4) `train_DefensiveLightGBM.py`  → feature-space adversarial detector

**Creates:**

* `models/Defense-LGBM/`

  * `model.joblib` (best candidate: LightGBM/XGBoost/LR)
  * `calibrator.joblib` (e.g., IsotonicRegression)
  * `feature_columns.json`
* `results_defense_features/Defense-LGBM/`

  * `candidate_summary.json` (val AUROC per candidate)
  * `metrics_val.json`, `metrics_test.json`
  * `classification_report_*.txt`, `confusion_matrix_*.png`, `roc_curve_*.png`
  * `params_used.json`

### 5) `defence_FeatureModels.py`  → two-stage defense evaluation (detector → base)

**Creates:**

* `results_defense_feat_stream/<ModelName>/`

  * `_common/` – cached detector scores for NAT streams (e.g., `p_adv_nat_full_*.npy`)
  * `NAT_pure/` – NAT-only stream; shows FP/acceptance behavior

    * `metrics.json`, `detector_confusion_tau.png`, `detector_roc_nat_vs_adv.png`, etc.
  * `mixed_FGSM/`, `mixed_PGD/` – NAT+ADV streams

    * `metrics.json` (detector AUROC/AP, acceptance rates, base perf on accepted/rejected)
    * `base_confusion_accepted_nat.png`, `base_roc_accepted_nat.png`
    * `p_adv_*.png` (score histograms)
    * `sample_trace.json`

### 6) `TrainAllFeatureDefensiveModels.py`  → 3-class “all-in-one” models

**Creates:**

* `models_base3/`

  * `_global/` – `scaler.joblib`, `feature_columns.json`, `split_indices.json`, `split_info.json`
  * `<ModelName>/` – `model.joblib` (ML) or `model.pt` (DL) + `params_snapshot.json`
* `results_base3/`

  * `<ModelName>/` – multi-class diagnostics

    * `metrics_*.json` (macro F1, OvR AUC, etc.)
    * `roc_curve.png` (OvR), `pr_curve.png`
    * `confusion_matrix.png`, `classification_report.txt`
    * `thresholds_val_class_0.csv`, `..._1.csv`, `..._2.csv`
  * `_combined/` – files suffixed with `_MC` (micro-avg ROC/PR, comparison tables)

### 7) `defence_FeatureModels3Class.py`  → 3-class defense (detector + router)

**Creates:**

* `results_defence_features_3class/<ModelName>/`

  * `mixed_FGSM/`, `mixed_PGD/`

    * `metrics_before_defense.json` (plain 3-class)
    * `metrics_after_defense.json` (detector-gated + router)
    * `confusion_no_def.png`, `confusion_after_def.png`
    * `roc_ovr_no_def.png`, `roc_ovr_after_def.png`
    * (optionally) `metrics_adaptive_attack.json`

### 8) Data building helpers

* `prep_defensive_dataset_features.py` → **features_adversarial_defense_dataset.csv**
  *NAT vs ADV labels (`is_adv`), original label (`orig_label`), attack type, source model, parent index, etc.)*
* `remove_dups_defensive.py` → **overwrites** `features_adversarial_defense_dataset.csv` with duplicates removed.
* `prep_defence_dataset_features_for_training.py` → **features_base3_strict_cap.csv**
  *Reservoir sampling to balance 0/1/2; caps per class; shuffled output.*

---

## 📚 Data & prerequisites

You will need **two inputs** depending on which models you train:

1. **Tabular features**

   * File: `features_extracted.csv`
   * Columns: `label` (0=benign, 1=malicious) + 30+ lexical/host features (length, dots, hyphens, digits, entropy, token stats, keywords, etc.).
   * Used by: `TrainAllModels_ML.py`, `TrainAllModels_DL.py` (tabular branches), *all* defense scripts.

2. **Raw URL text** *(only for char-level DL)*

   * File: `merged_url_dataset.csv` (columns: `url,label`)
   * Used by: `TrainAllModels_DL.py` (CharCNN / CharTransformer); tokenizer saved as `models/_global/url_tokenizer.json`.

> **Note:** Feature extraction & dataset merging scripts are not in this repo. Ensure consistent preprocessing and **avoid leakage** (fit scalers on train split only).

---

## ⚙️ Environment & execution

* Python ≥ 3.9
* Recommended packages: `numpy`, `pandas`, `scikit-learn`, `lightgbm`, `xgboost`, `matplotlib`, `seaborn`, `scipy`, `joblib`, `tqdm`, `torch` (for DL), etc.
* GPU optional for DL; CPU fallback supported (smaller batch sizes).

General tip:

```bash
python <script>.py --help    # many scripts expose flags (paths, seeds, caps)
```

---

## 🔬 Attacks & defenses — at a glance

* **Attacks:** FGSM (ε-bounded), PGD (iterative; step size & steps configurable)

  * **White-box:** DL-MLP (PyTorch) and linear models.
  * **Black-box transfer:** Tree ensembles via **surrogate MLP**.

* **Defenses (feature-space):**

  1. **Two-stage gate**: Adversarial detector (LightGBM/XGBoost/LR) uses `p_adv(x)`; accept if `p_adv(x) < τ`; pass to base classifier; reject otherwise.
  2. **3-class “all-in-one”**: Single model predicts {benign, malicious, adversarial}; optionally preceded by detector to route/boost reliability.

* **Reports generated** at every stage: dozens of metrics (ACC, F1, AUC, MCC, Brier, ECE, AP), confusion matrices, ROC/PR, calibration, feature importance, acceptance-rate summaries, and **global combined** comparison figures.

---

## 🔁 Reproducibility

* Train/val/test split indices saved (DL: `models/_global_dl/split_indices.json`; 3-class: `models_base3/_global/split_indices.json`).
* Global scalers/feature lists saved for exact re-use.
* Deterministic seeds provided where feasible; GPU non-determinism may remain.

---
