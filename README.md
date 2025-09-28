### 7) `defence_FeatureModels3Class.py`  → 3-class defense (detector + router, Q1 extensions)

**Purpose (drop-in, additive):** Evaluate 3-class models `{0=benign, 1=malicious, 2=ADV}` under mixed NAT/ADV streams, with an adversarial **feature-space detector (LightGBM)** in front. This script preserves legacy outputs/filenames and **adds** Q1-grade analyses:

- **Leakage-safe τ selection**: use a **held-out NAT calibration split** to choose gate threshold τ.
- **Split-conformal τ**: pick τ that caps **accepted-NAT risk** (misclassification on accepted NAT) at a target bound.
- **Risk–coverage** curves + CSVs for the gate and for conformal/legacy gates.
- **Gate baselines**: MSP (max softmax over NAT classes) and **Energy** (−logsumexp of logits) when logits are available.
- **Accepted-set calibration** on NAT: **ECE** and **Brier**.
- **Adaptive end-to-end attacks** on the composition (gate + 3-class): differentiable surrogate of the LightGBM gate (+ **SPSA** fallback).
- **OOM-robust** GPU/CPU fallbacks; optional **fast LightGBM predict** with early-stop margins.
- **New mirrored trees** for clean reporting:
  - `results_3class/...` (pure 3-class “no defense” artifacts, mirrored)
  - `results_3class_plus_detector/...` (conformal-gated 3-class artifacts, mirrored)

> **Attack-crafting restriction (by design):** only uses base (binary) models **DL-MLP**, **LightGBM**, **Random_Forest** to generate adversarials.  
> **Strict pairing** to 3-class model with the same folder name:  
> `DL-MLP → DL-MLP`, `Random_Forest → Random_Forest`, `LightGBM → LightGBM` (override via `--classifier3c`).

**Expects (paths are fixed):**
- Base models root: `models/` with `_global/{scaler.joblib, feature_columns.json}` and `models/<BaseName>/model.(pt|joblib)`.
- 3-class models root: `models_base3/` with `_global/{scaler.joblib, feature_columns.json}` and `models_base3/<Name>/model.(pt|joblib)`.
- Gate (detector): `models/Defense-LGBM/model.joblib` (+ optional `calibrator.joblib`).
- Dataset for feature-space models: `features_extracted.csv` (columns must match saved `feature_columns.json`).

**Outputs (exact filenames, per base and attack kind `mixed_FGSM` / `mixed_PGD`):**

```

results_defence_features_3class/<BaseName>/
_q1_common/
tau_summary.json                          # legacy quantile τ and conformal τ (+ settings)
risk_coverage_gate.csv                    # NAT coverage vs risk for the gate
risk_coverage_gate.png
tau_baselines.json                        # MSP/Energy thresholds (quantiles) used

mixed_FGSM/  (and mixed_PGD/)
metrics_before_defense.json               # 3-class only; NAT-only slice; stream counts; attack cfg
confusion_no_def.png
roc_ovr_no_def.png                        # if OvR AUC computed
classification_report_no_def.txt

```
metrics_after_defense.json                # legacy τ (preserved filename/shape)
classification_report_after_def.txt
confusion_after_def.png
roc_ovr_after_def.png                     # if OvR AUC computed

metrics_after_defense__gate_*.json        # gate_legacy, gate_conformal, msp_quantile, energy_quantile (if logits)
risk_coverage__gate_*.csv                 # for gate_* (legacy & conformal); coverage vs risk

metrics_adaptive_attack.json              # only when 3-class is torch & adaptive enabled
```

```

**Mirrored “clean” trees (no logic change, for reporting only):**
```

results_3class/<BaseName>/mixed_*/              # mirror of “no defense” artifacts
metrics_before_defense.json
classification_report_no_def.txt
confusion_no_def.png
roc_ovr_no_def.png
_q1_common/tau_summary.json
_q1_common/risk_coverage_gate.csv|png

results_3class_plus_detector/<BaseName>/mixed_*/
metrics_after_defense__gate_conformal.json    # conformal τ mirror
risk_coverage__gate_conformal.csv
_q1_common/tau_summary.json
_q1_common/risk_coverage_gate.csv|png
metrics_adaptive_attack.json                  # present when adaptive ran

```

**How it works (condensed):**
1. Load feature scalers and align base ↔ 3-class features safely (throws on mismatch).
2. Score **p_adv(x)** for all NAT with the LightGBM detector (+ optional isotonic calibrator).
3. Split NAT into **calibration** and **evaluation**; compute:
   - **Legacy τ** via quantile on calibration split.
   - **Conformal τ** that keeps accepted-NAT risk ≤ `--target-risk-nat` (with slack `--conformal-delta`).
   - Save **risk–coverage** CSV/PNG.
4. Craft adversarials for the chosen base (FGSM/PGD; white-box for DL-MLP, surrogate for trees).
5. Run 3-class predictions on NAT/ADV; compute **before-defense** metrics.
6. Apply gates (legacy/conformal/MSP/Energy) to compute **after-defense** metrics, calibration (ECE/Brier on accepted NAT), and acceptance/block rates.
7. **Adaptive attack (optional):** train a tiny MLP **gate surrogate** on `(X_raw, p_adv)` and optimize a combined objective to slip through gate **and** avoid class `2` on the 3-class model; **SPSA** fallback if gradients unavailable.

**CLI (only this script):**
```

python defence_FeatureModels3Class.py 
[--model {all|DL-MLP|Random_Forest|LightGBM}] 
[--nat-frac FLOAT] [--stream-size INT] 
[--eps FLOAT] [--alpha FLOAT] [--pgd-steps INT] [--fgsm-steps INT] 
[--eval-bs INT] [--lin-bs INT] [--bb-bs INT] 
[--torch-eval-bs INT] [--torch-attack-bs INT] 
[--no-plots] [--no-progress] [--amp] [--cache-adv] [--cpu] [--fast-predict] [--seed INT] 
[--nat-calib-ratio FLOAT] [--target-risk-nat FLOAT] [--conformal-delta FLOAT] [--tau-grid INT] 
[--disable-adaptive] [--classifier3c NAME]

```

**Practical defaults (from code):**
- Attack budget in z-space: `eps=0.40`, `alpha=0.10`, `FGSM steps=1`, `PGD steps=5`.
- Stream mixing: `NAT_FRAC=0.5`, `STREAM_SIZE=1_000_000` (sampling if needed).
- Conformal knobs: `nat_calib_ratio=0.2`, `target_risk_nat=0.05`, `conformal_delta=0.05`, `tau_grid=50`.
- GPU with OOM-backoff to CPU; LightGBM **fast predict** via `--fast-predict`.
- Reproducibility: `seed=42` (can override).

**Notes:**
- **Energy** gate requires logits/margins; available for many sklearn wrappers (e.g., LightGBM `raw_score=True`).
- **MSP** gate uses max softmax over NAT classes only (`max(p0, p1)`).
- **Mirrors** under `results_3class/` and `results_3class_plus_detector/` are for clean side-by-side reporting and do not affect logic.
```

---

If you want, I can also generate a tiny patch (diff) against your current README so you can apply it in one go, or produce a separately saved `README_addendum.md` that only documents this script’s Q1 extensions.
