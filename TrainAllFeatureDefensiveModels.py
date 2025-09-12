#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multiclass (0=benign, 1=malicious, 2=ADV) trainer — OOM-safe.
Models: DL-MLP, LightGBM (GPU→CPU fallback), Random Forest.

Outputs upgraded to DL-style diagnostics (multiclass-aware):
- Train-only scaler and shared 70/10/20 split indices
- Global diagnostics in results_base3/_global:
  data_quality.json, drift_report.csv/json, outliers.json,
  label_stats.json, class_weights.json
- Per-model diagnostics in results_base3/<Model>/:
  metrics_train.json, metrics_val.json, metrics_test.json,
  thresholds_val_class_<c>.csv (OvR for each class),
  metrics_test_at_opt_threshold_class_<c>.json,
  confusion_matrix.json/.png, roc_curve.png (OvR, all classes),
  pr_curve.png (OvR, all classes), calibration.png (+ stats),
  legacy: metrics.json, classification_report.txt
  feature_importance.csv/.png
  (DL gets permutation FI on ΔAUC_ovr; trees use native FI)
- Generalization gaps (train↔val/test for acc/AUC_ovr)
- Combined overlays for all three models in results_base3/_combined:
  combined_metrics_test_MC.{csv,json}
  combined_roc_test_MC.png (micro-avg),
  combined_pr_test_MC.png (micro-avg),
  combined_calibration_test_class<pos_class>_MC.png (default 2),
  combined_threshold_f1_val_class<pos_class>_MC.png (default 2),
  combined_threshold_youden_val_class<pos_class>_MC.png (default 2),
  combined_feature_importance_{raw,normalized}_MC.csv,
  combined_feature_importance_heatmap_all_MC.png,
  combined_feature_importance_topN_MC.png
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import argparse, json, warnings, os, time, gc, traceback, math, sys

import numpy as np
import pandas as pd
import joblib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, matthews_corrcoef, cohen_kappa_score,
    confusion_matrix, classification_report, log_loss,
    roc_auc_score, average_precision_score,
    RocCurveDisplay, PrecisionRecallDisplay, ConfusionMatrixDisplay,
    roc_curve, precision_recall_curve
)
from sklearn.ensemble import RandomForestClassifier, IsolationForest

# Optional LightGBM
try:
    import lightgbm as lgb
except Exception:
    lgb = None

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="features_base3_strict_cap.csv", help="Path to 3-class dataset CSV")
    p.add_argument("--label-col", default="label", help="Label column name (expects {0,1,2})")

    p.add_argument("--models-dir", default="models_base3")
    p.add_argument("--results-dir", default="results_base3")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.70)
    p.add_argument("--val-ratio", type=float, default=0.10)
    p.add_argument("--test-ratio", type=float, default=0.20)

    # System / parallelism
    p.add_argument("--jobs", type=int, default=max(1, min((os.cpu_count() or 8) - 1, 8)))

    # DataLoader safety for Py3.13
    p.add_argument("--dl-workers", type=int,
                   default=(0 if sys.version_info >= (3, 13) else 4),
                   help="DataLoader workers. Default 0 on Python 3.13 to avoid SymInt worker issues.")
    p.add_argument("--no-pin-memory", action="store_true",
                   help="Disable pin_memory (helps with Py3.13 DataLoader pinning thread issues).")

    # OOM-safe caps (None = full train)
    p.add_argument("--rf-max-samples", type=int, default=200_000,
                   help="Max training rows for RandomForest (stratified).")
    p.add_argument("--fallback-samples", type=int, default=150_000,
                   help="If a model OOMs, retry with this many rows (stratified).")

    # DL knobs
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-gpu", type=int, default=8192)
    p.add_argument("--batch-cpu", type=int, default=8192)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min-delta", type=float, default=1e-4)

    # Combined plots config
    p.add_argument("--fi-top-n", type=int, default=25, help="Top-N features for combined FI bars")
    p.add_argument("--pos-class-for-thresholds", type=int, default=2,
                   help="Which class to use for combined threshold sweeps (default: ADV=2)")
    return p.parse_args()

# ---------------- FS / utils ----------------
def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True); return path

def safe_name(name: str) -> str:
    return name.replace(" ", "_").replace("/", "_")

def save_text(text: str, path: Path):
    path.write_text(text, encoding="utf-8")

def save_json(obj: dict, path: Path):
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()

def mem_gc():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# ---------------- DataLoader helper (Py3.13-safe) ----------------
def make_loader(X_np, y_np, batch_size: int, shuffle: bool, device, workers: int, pin_mem: bool):
    # Force concrete, contiguous arrays and explicit dtypes → prevents SymInt shape issues
    Xc = np.ascontiguousarray(X_np, dtype=np.float32)
    yc = np.ascontiguousarray(y_np, dtype=np.int64)
    ds = TensorDataset(torch.from_numpy(Xc), torch.from_numpy(yc))
    dl_kwargs = dict(
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        num_workers=int(max(0, workers)),
        pin_memory=bool(pin_mem and device.type == "cuda"),
    )
    if dl_kwargs["num_workers"] > 0:
        dl_kwargs["persistent_workers"] = True
    return DataLoader(ds, **dl_kwargs)

# ---------------- Metrics helpers ----------------
def expected_calibration_error_binary(y_true_bin, y_prob, n_bins=15):
    y_true = np.asarray(y_true_bin).astype(int)
    y_prob = np.asarray(y_prob).clip(1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0; total = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (y_prob >= lo) & ((y_prob < hi) if i < n_bins-1 else (y_prob <= hi))
        if not np.any(mask): continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask)/total) * abs(acc - conf)
    return float(ece)

def ece_multiclass_macro(y_true, prob_mat, n_bins=15, classes=(0,1,2)):
    y_bin = label_binarize(y_true, classes=list(classes))
    vals = []
    for i, _ in enumerate(classes):
        vals.append(expected_calibration_error_binary(y_bin[:, i], prob_mat[:, i], n_bins))
    return float(np.mean(vals)) if vals else None

def brier_multiclass(y_true, prob_mat, classes=(0,1,2)):
    y_bin = label_binarize(y_true, classes=list(classes))
    return float(np.mean((prob_mat - y_bin)**2))

def rich_metrics_multiclass(y_true, y_pred, prob_mat=None, classes=(0,1,2), ece_bins=15):
    out = {
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro")),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
    }
    try:
        out["auc_ovr"] = float(roc_auc_score(y_true, prob_mat, multi_class="ovr")) if prob_mat is not None else None
        out["auc_ovo"] = float(roc_auc_score(y_true, prob_mat, multi_class="ovo")) if prob_mat is not None else None
    except Exception:
        out["auc_ovr"] = None; out["auc_ovo"] = None
    if prob_mat is not None:
        out["brier_mc"] = brier_multiclass(y_true, prob_mat, classes)
        try:
            out["log_loss_ce"] = float(log_loss(y_true, prob_mat, labels=list(classes)))
        except Exception:
            out["log_loss_ce"] = None
        try:
            out["ece_macro"] = ece_multiclass_macro(y_true, prob_mat, ece_bins, classes)
        except Exception:
            out["ece_macro"] = None
    else:
        out["brier_mc"] = None; out["log_loss_ce"] = None; out["ece_macro"] = None
    return out

def sweep_thresholds_ovr(y_true, proba, pos_class, thresholds=None):
    """One-vs-rest sweep for a specific class; returns DataFrame + best thresholds."""
    if thresholds is None: thresholds = np.linspace(0.01, 0.99, 99)
    y_pos = (np.asarray(y_true) == pos_class).astype(int)
    rows = []
    for t in thresholds:
        y_pred = (proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_pos, y_pred, labels=[0,1]).ravel()
        prec = precision_score(y_pos, y_pred, zero_division=0)
        rec  = recall_score(y_pos, y_pred)
        f1   = f1_score(y_pos, y_pred)
        spec = tn / (tn + fp) if (tn + fp) else 0.0
        bal  = balanced_accuracy_score(y_pos, y_pred)
        j    = rec + spec - 1.0
        rows.append({"threshold": float(t), "precision": float(prec), "recall": float(rec),
                     "f1": float(f1), "specificity": float(spec), "balanced_accuracy": float(bal),
                     "youden_j": float(j)})
    df = pd.DataFrame(rows)
    t_f1 = float(df.loc[df["f1"].idxmax(), "threshold"]) if len(df) else 0.5
    t_j  = float(df.loc[df["youden_j"].idxmax(), "threshold"]) if len(df) else 0.5
    return df, t_f1, t_j

# ---------------- Diagnostics helpers ----------------
def data_quality_report(df: pd.DataFrame, label_col: str):
    num_rows = int(df.shape[0])
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric = [c for c in df.columns if c not in numeric_cols + [label_col]]
    miss = df.isna().sum().to_dict()
    infs = {c: int(np.isinf(df[c]).sum()) for c in numeric_cols}
    dup_rows = int(df.duplicated().sum())
    const_cols = [c for c in numeric_cols if df[c].nunique(dropna=False) <= 1]
    near_const = [c for c in numeric_cols if df[c].value_counts(normalize=True, dropna=False).iloc[0] >= 0.99]
    return {
        "rows": num_rows,
        "non_numeric_excluded": non_numeric,
        "missing_counts": miss,
        "infinite_counts": infs,
        "duplicate_rows": dup_rows,
        "constant_numeric_columns": const_cols,
        "near_constant_numeric_columns_(>=99%_same_value)": near_const
    }

def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    v = np.unique(np.concatenate([a, b]))
    cdfa = np.searchsorted(np.sort(a), v, side="right") / a.size
    cdfb = np.searchsorted(np.sort(b), v, side="right") / b.size
    return float(np.max(np.abs(cdfa - cdfb)))

def psi(train_col: np.ndarray, test_col: np.ndarray, bins=10) -> float:
    eps = 1e-8
    qs = np.linspace(0, 1, bins+1)
    cuts = np.unique(np.quantile(train_col, qs))
    if cuts.size < 2: return 0.0
    def frac(x):
        idx = np.searchsorted(cuts, x, side="right") - 1
        idx = np.clip(idx, 0, cuts.size-2)
        counts = np.bincount(idx, minlength=cuts.size-1).astype(float)
        return counts / max(1, x.size)
    p = np.clip(frac(train_col), eps, None)
    q = np.clip(frac(test_col),  eps, None)
    return float(np.sum((q - p) * np.log(q / p)))

def drift_report(Xtr_raw, Xte_raw, feat_names, bins=10):
    rows = []
    for j, name in enumerate(feat_names):
        col_tr = Xtr_raw[:, j]; col_te = Xte_raw[:, j]
        rows.append({
            "feature": name,
            "ks": ks_statistic(col_tr, col_te),
            "psi": psi(col_tr, col_te, bins=bins),
            "train_mean": float(np.mean(col_tr)),
            "test_mean": float(np.mean(col_te))
        })
    return pd.DataFrame(rows).sort_values(["psi", "ks"], ascending=False)

# ---------------- Subsampling helper ----------------
def stratified_subsample(X, y, cap: int, seed: int):
    if cap is None or cap <= 0 or cap >= len(y):
        return X, y, None
    rng = np.random.default_rng(seed)
    idx_by_c = [np.where(y == c)[0] for c in np.unique(y)]
    per_c = max(1, cap // len(idx_by_c))
    pick = []
    for arr in idx_by_c:
        take = min(per_c, len(arr))
        pick.append(rng.choice(arr, size=take, replace=False))
    sel = np.concatenate(pick, axis=0)
    rng.shuffle(sel)
    return X[sel], y[sel], int(sel.size)

# ---------------- Models ----------------
def build_models(rs: int, n_classes: int, n_jobs: int):
    models = {}
    # Random Forest
    models["Random_Forest"] = RandomForestClassifier(
        n_estimators=300, random_state=rs, n_jobs=n_jobs,
        bootstrap=True, max_samples=0.2,
        max_features="sqrt", min_samples_leaf=2
    )
    # LightGBM (if available)
    if lgb is not None:
        device = "gpu" if torch.cuda.is_available() else "cpu"
        models["LightGBM"] = lgb.LGBMClassifier(
            objective="multiclass", num_class=n_classes, random_state=rs, n_jobs=n_jobs,
            n_estimators=10000, learning_rate=0.03,
            num_leaves=127, subsample=0.8, colsample_bytree=0.8,
            device_type=device
        )
    return models

# ---------------- DL: Multiclass MLP ----------------
class MLPNetMulti(nn.Module):
    def __init__(self, in_dim: int, out_classes=3, hidden=(256,128,64), p_drop=0.2):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.BatchNorm1d(h), nn.Dropout(p_drop)]
            prev = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, out_classes)
    def forward(self, x):
        return self.head(self.backbone(x))

@torch.no_grad()
def infer_probs_mc(model, loader, device):
    model.eval()
    outs, labs = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True).float()
        logits = model(xb)
        probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
        outs.append(probs); labs.append(yb.numpy())
    return np.concatenate(outs), np.concatenate(labs)

def current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])

def permutation_importance_multiclass_ovr_auc(model, X_test_np, y_test, device, repeats=3):
    """Permutation FI: drop in AUC_ovr averaged across classes."""
    with torch.no_grad():
        base_probs = []
        bs = 65536
        for i in range(0, len(X_test_np), bs):
            xb = torch.from_numpy(X_test_np[i:i+bs]).to(device).float()
            base_probs.append(torch.softmax(model(xb), dim=-1).detach().cpu().numpy())
        base_probs = np.concatenate(base_probs)
    try:
        base_auc = roc_auc_score(y_test, base_probs, multi_class="ovr")
    except Exception:
        base_auc = np.nan

    rng = np.random.default_rng(123)
    F = X_test_np.shape[1]
    importances = np.zeros(F, dtype=float)
    for j in range(F):
        drops = []
        for _ in range(repeats):
            Xp = X_test_np.copy()
            rng.shuffle(Xp[:, j])
            with torch.no_grad():
                probs = []
                for i in range(0, len(Xp), 65536):
                    xb = torch.from_numpy(Xp[i:i+65536]).to(device).float()
                    probs.append(torch.softmax(model(xb), dim=-1).detach().cpu().numpy())
                probs = np.concatenate(probs)
            try:
                auc = roc_auc_score(y_test, probs, multi_class="ovr")
            except Exception:
                auc = np.nan
            drops.append(base_auc - auc if (not np.isnan(base_auc) and not np.isnan(auc)) else 0.0)
        importances[j] = float(np.mean(drops))
    return importances

def write_feature_importance(rdir: Path, feature_names, importances, title: str):
    df_fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
    df_fi.to_csv(rdir / "feature_importance.csv", index=False)
    top = min(40, len(df_fi))
    fig_h = max(4.0, 0.32 * top)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    ax.barh(df_fi["feature"].head(top)[::-1], df_fi["importance"].head(top)[::-1])
    ax.set_title(title)
    ax.set_xlabel("Importance")
    fig.tight_layout()
    fig.savefig(rdir / "feature_importance.png", dpi=200)
    plt.close(fig)

def train_dl_model(name, model, train_loader, val_loader, test_loader, device, out_mdir: Path, out_rdir: Path,
                   feature_names, epochs=20, patience=5, min_delta=1e-4,
                   use_plateau: bool=True, plateau_factor: float=0.5, plateau_patience: int=1, min_lr: float=1e-5):
    start = time.time()
    ensure_dir(out_mdir); ensure_dir(out_rdir)

    # Class weights for CE (multiclass)
    y_tr = train_loader.dataset.tensors[1].cpu().numpy()
    classes = np.unique(y_tr)
    cls_w = compute_class_weight("balanced", classes=classes, y=y_tr)
    weight = torch.tensor(cls_w, dtype=torch.float32, device=device)

    criterion = nn.CrossEntropyLoss(weight=weight)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-5)

    scheduler = None
    if use_plateau:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=plateau_factor, patience=plateau_patience,
            threshold=min_delta, min_lr=min_lr
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type=="cuda"))

    best_score, noimp, best_state = -np.inf, 0, None
    for ep in range(1, epochs+1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True).float()
            yb = yb.to(device, non_blocking=True).long()
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type=="cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            running += loss.item() * xb.size(0)

        # Validate: macro-F1
        val_probs, val_y = infer_probs_mc(model, val_loader, device)
        val_pred = val_probs.argmax(axis=1)
        val_f1 = f1_score(val_y, val_pred, average="macro")
        if scheduler is not None:
            scheduler.step(val_f1)

        improved = (val_f1 > best_score + min_delta)
        if improved:
            best_score = val_f1; noimp = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            noimp += 1
            if noimp >= patience:
                break

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    torch.save(best_state, out_mdir / "model.pt")
    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()

    # Per-split probs/preds
    tr_p, tr_y = infer_probs_mc(model, train_loader, device); tr_hat = tr_p.argmax(axis=1)
    va_p, va_y = infer_probs_mc(model, val_loader,   device); va_hat = va_p.argmax(axis=1)
    te_p, te_y = infer_probs_mc(model, test_loader,  device); te_hat = te_p.argmax(axis=1)

    # Write metrics JSONs
    for split, y, yhat, p in [
        ("train", tr_y, tr_hat, tr_p),
        ("val",   va_y, va_hat, va_p),
        ("test",  te_y, te_hat, te_p),
    ]:
        m = rich_metrics_multiclass(y, yhat, p, classes=(0,1,2), ece_bins=15)
        save_json(m, out_rdir / f"metrics_{split}.json")

    # Legacy + plots
    write_legacy_and_plots_multiclass(name, te_y, te_hat, te_p, out_rdir, ece_bins=15)

    # Threshold sweeps per class on VAL → apply to TEST
    for c in (0,1,2):
        sweep_df, t_f1, t_j = sweep_thresholds_ovr(va_y, va_p[:, c], pos_class=c)
        sweep_df.to_csv(out_rdir / f"thresholds_val_class_{c}.csv", index=False)

        # Apply best F1 threshold (OvR) to TEST and log metrics for that binary task
        y_bin = (te_y == c).astype(int)
        yhat_opt = (te_p[:, c] >= t_f1).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_bin, yhat_opt, labels=[0,1]).ravel()
        try:
            auc = roc_auc_score(y_bin, te_p[:, c])
            ap  = average_precision_score(y_bin, te_p[:, c])
        except Exception:
            auc, ap = None, None
        save_json({
            "class": c,
            "threshold_used": float(t_f1),
            "precision": float(precision_score(y_bin, yhat_opt, zero_division=0)),
            "recall": float(recall_score(y_bin, yhat_opt)),
            "f1": float(f1_score(y_bin, yhat_opt)),
            "specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
            "balanced_accuracy": float(balanced_accuracy_score(y_bin, yhat_opt)),
            "auc_roc_ovr": (float(auc) if auc is not None else None),
            "ap_pr": (float(ap) if ap is not None else None),
            "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        }, out_rdir / f"metrics_test_at_opt_threshold_class_{c}.json")

    # Generalization gaps
    mt = json.loads((out_rdir / "metrics_train.json").read_text())
    mv = json.loads((out_rdir / "metrics_val.json").read_text())
    mte= json.loads((out_rdir / "metrics_test.json").read_text())
    gaps = {
        "acc_gap_train_val": float(mt["accuracy"] - mv["accuracy"]),
        "acc_gap_train_test": float(mt["accuracy"] - mte["accuracy"]),
        "auc_ovr_gap_train_val": (float(mt["auc_ovr"] - mv["auc_ovr"]) if (mt["auc_ovr"] is not None and mv["auc_ovr"] is not None) else None),
        "auc_ovr_gap_train_test": (float(mt["auc_ovr"] - mte["auc_ovr"]) if (mt["auc_ovr"] is not None and mte["auc_ovr"] is not None) else None),
    }
    save_json(gaps, out_rdir / "generalization_gap.json")

    # Permutation FI for DL (ΔAUC_ovr)
    try:
        X_te_np = test_loader.dataset.tensors[0].numpy()
        fi_vec = permutation_importance_multiclass_ovr_auc(model, X_te_np, te_y, device, repeats=3)
        write_feature_importance(out_rdir, feature_names, fi_vec,
                                 title=f"Feature Importances (perm ΔAUC_ovr): {name}")
        fi_map = {fn: float(v) for fn, v in zip(feature_names, fi_vec)}
    except Exception:
        fi_vec = np.zeros(len(feature_names), dtype=float)
        fi_map = {fn: 0.0 for fn in feature_names}
    save_json({"feature_importance_perm_auc_ovr": fi_map}, out_rdir / "other_metrics.json")

    # Return curves & FI for combined overlays
    return dict(
        name=name,
        y_test=te_y,
        proba_test=te_p,
        proba_val=va_p,
        fi_vec=fi_vec
    )

def write_legacy_and_plots_multiclass(name, y_true, y_pred, prob_mat, rdir: Path, ece_bins: int):
    # legacy metrics.json + classification_report.txt
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    rec = recall_score(y_true, y_pred, average="macro")
    f1m = f1_score(y_true, y_pred, average="macro")
    try: auc = roc_auc_score(y_true, prob_mat, multi_class="ovr")
    except Exception: auc = None
    legacy = {
        "model": name,
        "samples_evaluated": int(len(y_true)),
        "accuracy": round(float(acc), 6),
        "precision_macro": round(float(prec), 6),
        "recall_macro": round(float(rec), 6),
        "f1_macro": round(float(f1m), 6),
        "auc_ovr": (round(float(auc), 6) if auc is not None else None),
        "timestamp": now_utc(),
    }
    save_json(legacy, rdir / "metrics.json")
    save_text(classification_report(y_true, y_pred, digits=4), rdir / "classification_report.txt")

    # confusion JSON + PNG
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2])
    save_json({"labels":[0,1,2], "matrix": cm.tolist()}, rdir / "confusion_matrix.json")
    fig, ax = plt.subplots(figsize=(6,5.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0,1,2], ax=ax)
    ax.set_title(f"Confusion Matrix: {name}"); fig.tight_layout()
    fig.savefig(rdir / "confusion_matrix.png", dpi=200); plt.close(fig)

    # ROC OvR (all classes on one plot)
    try:
        y_bin = label_binarize(y_true, classes=[0,1,2])
        fig, ax = plt.subplots(figsize=(7,5.5))
        for i, cls in enumerate([0,1,2]):
            RocCurveDisplay.from_predictions(y_bin[:, i], prob_mat[:, i], name=f"class {cls}", ax=ax)
        ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
        ax.set_title(f"ROC (OvR): {name}"); fig.tight_layout()
        fig.savefig(rdir / "roc_curve.png", dpi=200); plt.close(fig)
    except Exception:
        pass

    # PR OvR
    try:
        y_bin = label_binarize(y_true, classes=[0,1,2])
        fig, ax = plt.subplots(figsize=(7,5.5))
        for i, cls in enumerate([0,1,2]):
            PrecisionRecallDisplay.from_predictions(y_bin[:, i], prob_mat[:, i], name=f"class {cls}", ax=ax)
        ax.set_title(f"Precision-Recall (OvR): {name}"); fig.tight_layout()
        fig.savefig(rdir / "pr_curve.png", dpi=200); plt.close(fig)
    except Exception:
        pass

    # Calibration (per-class on same axes)
    try:
        y_bin = label_binarize(y_true, classes=[0,1,2])
        fig, ax = plt.subplots(figsize=(7,5.5))
        ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
        from sklearn.calibration import CalibrationDisplay
        for i, cls in enumerate([0,1,2]):
            disp = CalibrationDisplay.from_predictions(y_bin[:, i], prob_mat[:, i], n_bins=15, ax=ax)
            if hasattr(disp, "line_"): disp.line_.set_label(f"class {cls}")
        ax.set_title(f"Calibration (OvR): {name}")
        ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
        ax.legend()
        fig.tight_layout(); fig.savefig(rdir / "calibration.png", dpi=200); plt.close(fig)
        save_text(
            f"Brier (multiclass): {brier_multiclass(y_true, prob_mat):.6f}\n"
            f"ECE macro (bins=15): {ece_multiclass_macro(y_true, prob_mat, 15):.6f}\n"
            f"LogLoss (CE): {log_loss(y_true, prob_mat, labels=[0,1,2]):.6f}\n",
            rdir / "calibration_stats.txt"
        )
    except Exception:
        pass

# ---------------- Combined overlays (MC suffix) ----------------
def _normalize_importance(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    vmax = np.nanmax(v) if v.size else 0.0
    vmin = np.nanmin(v) if v.size else 0.0
    if not np.isfinite(vmax) or vmax == vmin:
        return np.zeros_like(v, dtype=float)
    return (v - vmin) / (vmax - vmin)

def finalize_combined_overlays_MC(results_root: Path, combined_curves, combined_rows,
                                  feature_names, model_to_fi, top_n=25, pos_class=2):
    comb_dir = ensure_dir(results_root / "_combined")

    # (1) Combined metrics table
    dfm = pd.DataFrame(combined_rows)
    dfm.to_csv(comb_dir / "combined_metrics_test_MC.csv", index=False)
    save_json(dfm.to_dict(orient="records"), comb_dir / "combined_metrics_test_MC.json")

    # (2) ROC micro-average (multiclass)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for item in combined_curves:
        y = item["y_test"]; P = item["proba_test"]
        if P is None:  # guard
            continue
        y_bin = label_binarize(y, classes=[0,1,2])
        fpr, tpr, _ = roc_curve(y_bin.ravel(), P.ravel())
        auc = roc_auc_score(y_bin, P, average="micro")
        ax.plot(fpr, tpr, label=f"{item['name']} (micro AUC={auc:.3f})")
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Combined ROC (micro-average, Test) — MC")
    ax.legend()
    fig.tight_layout(); fig.savefig(comb_dir / "combined_roc_test_MC.png", dpi=220); plt.close(fig)

    # (3) PR micro-average
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for item in combined_curves:
        y = item["y_test"]; P = item["proba_test"]
        if P is None:
            continue
        y_bin = label_binarize(y, classes=[0,1,2])
        pr, rc, _ = precision_recall_curve(y_bin.ravel(), P.ravel())
        ap = average_precision_score(y_bin, P, average="micro")
        ax.plot(rc, pr, label=f"{item['name']} (micro AP={ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Combined Precision–Recall (micro-average, Test) — MC")
    ax.legend()
    fig.tight_layout(); fig.savefig(comb_dir / "combined_pr_test_MC.png", dpi=220); plt.close(fig)

    # (4) Combined calibration for class=pos_class (OvR)
    fig, ax = plt.subplots(figsize=(7.5, 6))
    ax.plot([0,1],[0,1], linestyle="--", linewidth=1)
    have_any = False
    from sklearn.calibration import CalibrationDisplay
    for item in combined_curves:
        P = item["proba_test"]
        if P is None:
            continue
        y = item["y_test"]; p = P[:, pos_class]
        yb = (y == pos_class).astype(int)
        try:
            disp = CalibrationDisplay.from_predictions(yb, p, n_bins=15, ax=ax)
            if hasattr(disp, "line_"): disp.line_.set_label(item["name"])
            have_any = True
        except Exception:
            pass
    if have_any:
        ax.set_title(f"Combined Calibration (Test) — class {pos_class} vs rest — MC")
        ax.set_xlabel("Mean Predicted Probability"); ax.set_ylabel("Fraction of Positives")
        ax.legend()
        fig.tight_layout(); fig.savefig(comb_dir / f"combined_calibration_test_class{pos_class}_MC.png", dpi=220)
    plt.close(fig)

def finalize_threshold_plots_MC(results_root: Path, combined_curves, y_val, pos_class=2):
    comb_dir = ensure_dir(results_root / "_combined")
    # F1
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for item in combined_curves:
        p_val = item.get("proba_val", None)
        if p_val is None: continue
        sweep_df, _, _ = sweep_thresholds_ovr(y_val, p_val[:, pos_class], pos_class=pos_class)
        ax.plot(sweep_df["threshold"], sweep_df["f1"], label=item["name"])
    ax.set_xlabel("Threshold"); ax.set_ylabel("F1")
    ax.set_title(f"Combined Threshold Sweep (Val) — F1 — class {pos_class} — MC")
    ax.legend()
    fig.tight_layout(); fig.savefig(comb_dir / f"combined_threshold_f1_val_class{pos_class}_MC.png", dpi=220); plt.close(fig)

    # Youden J
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for item in combined_curves:
        p_val = item.get("proba_val", None)
        if p_val is None: continue
        sweep_df, _, _ = sweep_thresholds_ovr(y_val, p_val[:, pos_class], pos_class=pos_class)
        ax.plot(sweep_df["threshold"], sweep_df["youden_j"], label=item["name"])
    ax.set_xlabel("Threshold"); ax.set_ylabel("Youden J (TPR + TNR − 1)")
    ax.set_title(f"Combined Threshold Sweep (Val) — Youden J — class {pos_class} — MC")
    ax.legend()
    fig.tight_layout(); fig.savefig(comb_dir / f"combined_threshold_youden_val_class{pos_class}_MC.png", dpi=220); plt.close(fig)

def build_combined_feature_importance_plots_MC(results_root: Path, feature_names, model_to_fi, top_n=25):
    comb_dir = ensure_dir(results_root / "_combined")
    models = list(model_to_fi.keys())
    F = len(feature_names)

    if not models or F == 0:
        pd.DataFrame(columns=models, index=feature_names).to_csv(comb_dir / "combined_feature_importance_raw_MC.csv")
        pd.DataFrame(columns=models, index=feature_names).to_csv(comb_dir / "combined_feature_importance_normalized_MC.csv")
        return

    raw_mat = np.vstack([model_to_fi[m] for m in models])
    norm_mat = np.vstack([_normalize_importance(model_to_fi[m]) for m in models])

    df_raw = pd.DataFrame(raw_mat.T, index=feature_names, columns=models)
    df_norm = pd.DataFrame(norm_mat.T, index=feature_names, columns=models)
    df_raw.to_csv(comb_dir / "combined_feature_importance_raw_MC.csv")
    df_norm.to_csv(comb_dir / "combined_feature_importance_normalized_MC.csv")

    # Heatmap (normalized)
    fig_w = max(8.0, 0.5 * len(models))
    fig_h = max(6.0, 0.2 * F)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(df_norm.values, aspect="auto", interpolation="nearest")
    ax.set_yticks(np.arange(F)); ax.set_yticklabels(feature_names)
    ax.set_xticks(np.arange(len(models))); ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_title("Feature Importance — All features across models (normalized) — MC")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    fig.tight_layout(); fig.savefig(comb_dir / "combined_feature_importance_heatmap_all_MC.png", dpi=220); plt.close(fig)

    # Top-N grouped bars by mean normalized importance
    mean_norm = np.nanmean(df_norm.values, axis=1)
    top_n = min(max(1, top_n), F)
    idx = np.argsort(mean_norm)[-top_n:]
    top_feats = [feature_names[i] for i in idx]
    x = np.arange(top_n, dtype=float)
    width = max(0.8 / max(1, len(models)), 0.06)
    fig_w2 = max(10.0, 0.5 * top_n)
    fig, ax = plt.subplots(figsize=(fig_w2, 6))
    for i, m in enumerate(models):
        ax.bar(x + i*width, df_norm.values[idx, i], width, label=m)
    ax.set_xticks(x + (len(models)-1)*width/2)
    ax.set_xticklabels(top_feats, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized importance (0–1)")
    ax.set_title(f"Feature Importance — Top-{top_n} across models (normalized) — MC")
    ax.legend(ncols=2)
    fig.tight_layout(); fig.savefig(comb_dir / "combined_feature_importance_topN_MC.png", dpi=220); plt.close(fig)

# ---------------- Main ----------------
def main():
    args = parse_args()

    # Reproducibility hygiene (doesn't force determinism for speed, but seeds PRNGs)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Parallelism hygiene
    os.environ.setdefault("JOBLIB_TEMP_FOLDER", str(Path.cwd() / ".joblib_tmp"))
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, args.jobs)))
    ensure_dir(Path(os.environ["JOBLIB_TEMP_FOLDER"]))

    # Paths
    data_csv = Path(args.data); assert data_csv.exists(), f"Data CSV not found: {data_csv}"
    models_root = ensure_dir(Path(args.models_dir))
    results_root = ensure_dir(Path(args.results_dir))
    global_m = ensure_dir(models_root / "_global")
    global_r = ensure_dir(results_root / "_global")

    rs = int(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===== Load data =====
    df = pd.read_csv(data_csv)
    assert args.label_col in df.columns, f"Missing label column: {args.label_col}"

    # Numeric-only features
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    assert args.label_col in numeric_cols, f"Label column '{args.label_col}' must be numeric."
    feature_cols = [c for c in numeric_cols if c != args.label_col]

    y_all = df[args.label_col].to_numpy(dtype=np.int64)
    uniq = set(np.unique(y_all).tolist())
    assert uniq <= {0,1,2}, f"Expected labels in {{0,1,2}}, got: {sorted(uniq)}"
    n_classes = 3

    # ===== Shared split of indices (then scale TRAIN only) =====
    idx_all = np.arange(len(df))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx_all, y_all, test_size=(1.0 - args.train_ratio), random_state=rs, stratify=y_all
    )
    idx_val, idx_te, y_val, y_te = train_test_split(
        idx_tmp, y_tmp,
        test_size=(args.test_ratio / (args.val_ratio + args.test_ratio)),
        random_state=rs, stratify=y_tmp
    )
    save_json({"train_idx": idx_tr.tolist(), "val_idx": idx_val.tolist(), "test_idx": idx_te.tolist()},
              global_m / "split_indices.json")

    # Raw arrays (pre-scale) for drift
    X_raw_all = df.loc[:, feature_cols].to_numpy(dtype=np.float32)
    X_tr_raw, X_val_raw, X_te_raw = X_raw_all[idx_tr], X_raw_all[idx_val], X_raw_all[idx_te]

    # Train-only scaler (prevents leakage)
    scaler = StandardScaler().fit(X_tr_raw)
    X_tr = scaler.transform(X_tr_raw); X_val = scaler.transform(X_val_raw); X_te = scaler.transform(X_te_raw)
    joblib.dump(scaler, global_m / "scaler.joblib")
    save_json({"feature_columns": feature_cols}, global_m / "feature_columns.json")
    save_json({"shapes": {
        "train": [int(X_tr.shape[0]), int(X_tr.shape[1])],
        "val":   [int(X_val.shape[0]), int(X_val.shape[1])],
        "test":  [int(X_te.shape[0]),  int(X_te.shape[1])] },
        "timestamp": now_utc()}, global_m / "split_info.json")

    # ===== Global diagnostics =====
    save_json(data_quality_report(df[feature_cols + [args.label_col]], args.label_col), global_r / "data_quality.json")
    ddf = drift_report(X_tr_raw, X_te_raw, np.array(feature_cols), bins=10)
    ddf.to_csv(global_r / "drift_report.csv", index=False)
    save_json({
        "top_by_psi": ddf.head(20).to_dict(orient="records"),
        "psi_guidance": "PSI <0.1 stable; 0.1–0.25 moderate; >0.25 major shift",
        "ks_guidance":  "KS <0.1 small; 0.1–0.2 moderate; >0.2 large"
    }, global_r / "drift_report.json")

    iso = IsolationForest(n_estimators=200, random_state=rs, contamination=0.02).fit(X_tr)
    def outlier_frac(Xp): return float(np.mean(iso.predict(Xp) == -1))
    save_json({
        "contamination_cfg": 0.02,
        "train_flagged_fraction": outlier_frac(X_tr),
        "val_flagged_fraction": outlier_frac(X_val),
        "test_flagged_fraction": outlier_frac(X_te)
    }, global_r / "outliers.json")

    cls_w = compute_class_weight("balanced", classes=np.array([0,1,2]), y=y_tr)
    save_json({"class_weights_(0,1,2)": [float(x) for x in cls_w]}, global_r / "class_weights.json")

    # ===== DataLoaders for DL (Py3.13-safe) =====
    bs = int(args.batch_gpu if device.type == "cuda" else args.batch_cpu)
    workers = int(args.dl_workers)
    pin_mem = (not args.no_pin_memory)

    tr_loader = make_loader(X_tr, y_tr, bs, True,  device, workers, pin_mem)
    val_loader= make_loader(X_val, y_val, bs, False, device, workers, pin_mem)
    te_loader = make_loader(X_te,  y_te,  bs, False, device, workers, pin_mem)

    # ===== Train DL-MLP =====
    combined_curves = []
    combined_rows   = []
    model_to_fi     = {}

    name = "DL-MLP"
    mdir = ensure_dir(models_root / safe_name(name))
    rdir = ensure_dir(results_root / safe_name(name))
    dl_model = MLPNetMulti(in_dim=X_tr.shape[1], out_classes=n_classes, hidden=(256,128,64), p_drop=0.2).to(device)
    dl_item = train_dl_model(
        name, dl_model, tr_loader, val_loader, te_loader, device,
        mdir, rdir, feature_cols, epochs=args.epochs, patience=args.patience, min_delta=args.min_delta,
        use_plateau=True
    )
    combined_curves.append({k: dl_item[k] for k in ["name","y_test","proba_test","proba_val"]})
    model_to_fi[name] = dl_item.get("fi_vec", np.zeros(len(feature_cols), dtype=float))

    # Accumulate DL row for combined metrics table
    try:
        mte = json.loads((rdir / "metrics_test.json").read_text())
        combined_rows.append({
            "model": name,
            "samples": mte["samples"],
            "accuracy": mte["accuracy"],
            "precision_macro": mte["precision_macro"],
            "recall_macro": mte["recall_macro"],
            "f1_macro": mte["f1_macro"],
            "balanced_accuracy": mte["balanced_accuracy"],
            "mcc": mte["mcc"],
            "kappa": mte["kappa"],
            "auc_ovr": mte["auc_ovr"],
            "auc_ovo": mte["auc_ovo"],
            "brier_mc": mte["brier_mc"],
            "ece_macro": mte["ece_macro"],
            "log_loss_ce": mte["log_loss_ce"],
        })
    except Exception:
        pass

    # ===== Train Classic ML (RF + LightGBM) =====
    models = build_models(rs, n_classes, args.jobs)

    for name, model in tqdm(models.items(), desc="ML models"):
        mname = safe_name(name)
        mdir = ensure_dir(models_root / mname)
        rdir = ensure_dir(results_root / mname)
        (mdir / "train_log.txt").write_text("", encoding="utf-8")  # fresh log

        start = time.time()
        try:
            params_snapshot = getattr(model, "get_params", lambda: {})()
        except Exception:
            params_snapshot = {}
        # record params snapshot
        save_json({"params": params_snapshot}, mdir / "params_snapshot.json")

        try:
            if (lgb is not None) and isinstance(model, lgb.LGBMClassifier):
                dev = model.get_params().get("device_type", "cpu")
                try:
                    fitted = model.fit(
                        X_tr, y_tr,
                        eval_set=[(X_val, y_val)],
                        eval_metric="multi_logloss",
                        callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
                    )
                except Exception as e:
                    msg = str(e)
                    if ("OpenCL" in msg) or ("No OpenCL device" in msg) or ("GPU" in msg and "not" in msg):
                        # Fallback to CPU
                        params = model.get_params()
                        params["device_type"] = "cpu"
                        model = lgb.LGBMClassifier(**params)
                        fitted = model.fit(
                            X_tr, y_tr,
                            eval_set=[(X_val, y_val)],
                            eval_metric="multi_logloss",
                            callbacks=[lgb.early_stopping(stopping_rounds=200, verbose=False)]
                        )
                    else:
                        raise
            elif isinstance(model, RandomForestClassifier):
                # Optionally subsample ahead of time if dataset is huge
                Xrf, yrf, picked = stratified_subsample(X_tr, y_tr, args.rf_max_samples, seed=rs)
                fitted = model.fit(Xrf, yrf)
            else:
                fitted = model.fit(X_tr, y_tr)
        except MemoryError:
            Xs, ys, picked = stratified_subsample(X_tr, y_tr, args.fallback_samples, seed=rs)
            fitted = model.fit(Xs, ys)
        except Exception as e:
            tb = traceback.format_exc()
            save_text(f"ERROR during fit: {e}\n{tb}", mdir / "train_log.txt")
            continue

        # Save model
        try:
            joblib.dump(fitted, mdir / "model.joblib")
        except Exception:
            pass

        # Per-split predictions
        def score_matrix(estimator, X) -> np.ndarray | None:
            try:
                if hasattr(estimator, "predict_proba"):
                    p = estimator.predict_proba(X)
                    if isinstance(p, list): p = p[0]
                    if getattr(p, "ndim", 1) == 2: return np.asarray(p)
            except Exception:
                pass
            try:
                if hasattr(estimator, "decision_function"):
                    z = np.asarray(estimator.decision_function(X))
                    if z.ndim == 2:
                        e = np.exp(z - z.max(axis=1, keepdims=True))
                        p = e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)
                        return p
            except Exception:
                pass
            return None

        P_tr = score_matrix(fitted, X_tr); yhat_tr = fitted.predict(X_tr)
        P_va = score_matrix(fitted, X_val); yhat_va = fitted.predict(X_val)
        P_te = score_matrix(fitted, X_te);  yhat_te = fitted.predict(X_te)

        # Metrics JSONs
        mt = rich_metrics_multiclass(y_tr, yhat_tr, P_tr, classes=(0,1,2), ece_bins=15)
        mv = rich_metrics_multiclass(y_val, yhat_va, P_va, classes=(0,1,2), ece_bins=15)
        mte= rich_metrics_multiclass(y_te, yhat_te, P_te, classes=(0,1,2), ece_bins=15)
        save_json(mt, rdir / "metrics_train.json")
        save_json(mv, rdir / "metrics_val.json")
        save_json(mte, rdir / "metrics_test.json")

        # Legacy + plots
        write_legacy_and_plots_multiclass(name, y_te, yhat_te, P_te, rdir, ece_bins=15)

        # Threshold sweeps per class on VAL → test application
        for c in (0,1,2):
            if P_va is None or P_te is None: break
            sweep_df, t_f1, t_j = sweep_thresholds_ovr(y_val, P_va[:, c], pos_class=c)
            sweep_df.to_csv(rdir / f"thresholds_val_class_{c}.csv", index=False)

            y_bin = (y_te == c).astype(int)
            yhat_opt = (P_te[:, c] >= t_f1).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_bin, yhat_opt, labels=[0,1]).ravel()
            try:
                auc = roc_auc_score(y_bin, P_te[:, c])
                ap  = average_precision_score(y_bin, P_te[:, c])
            except Exception:
                auc, ap = None, None
            save_json({
                "class": c,
                "threshold_used": float(t_f1),
                "precision": float(precision_score(y_bin, yhat_opt, zero_division=0)),
                "recall": float(recall_score(y_bin, yhat_opt)),
                "f1": float(f1_score(y_bin, yhat_opt)),
                "specificity": float(tn / (tn + fp) if (tn + fp) else 0.0),
                "balanced_accuracy": float(balanced_accuracy_score(y_bin, yhat_opt)),
                "auc_roc_ovr": (float(auc) if auc is not None else None),
                "ap_pr": (float(ap) if ap is not None else None),
                "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
            }, rdir / f"metrics_test_at_opt_threshold_class_{c}.json")

        # Generalization gaps
        gaps = {
            "acc_gap_train_val": float(mt["accuracy"] - mv["accuracy"]),
            "acc_gap_train_test": float(mt["accuracy"] - mte["accuracy"]),
            "auc_ovr_gap_train_val": (float(mt["auc_ovr"] - mv["auc_ovr"]) if (mt["auc_ovr"] is not None and mv["auc_ovr"] is not None) else None),
            "auc_ovr_gap_train_test": (float(mt["auc_ovr"] - mte["auc_ovr"]) if (mt["auc_ovr"] is not None and mte["auc_ovr"] is not None) else None),
        }
        save_json(gaps, rdir / "generalization_gap.json")

        # Feature importances
        fi_vec = None
        try:
            src = fitted
            if hasattr(src, "feature_importances_"):
                vals = np.asarray(src.feature_importances_, dtype=float)
                fi_vec = vals
                write_feature_importance(rdir, feature_cols, vals, title=f"Feature Importances: {name}")
            else:
                fi_vec = np.zeros(len(feature_cols), dtype=float)
        except Exception:
            fi_vec = np.zeros(len(feature_cols), dtype=float)

        model_to_fi[name] = fi_vec

        # Accumulate for combined overlays
        combined_curves.append({
            "name": name,
            "y_test": y_te,
            "proba_test": P_te,
            "proba_val":  P_va
        })
        combined_rows.append({
            "model": name,
            "samples": mte["samples"],
            "accuracy": mte["accuracy"],
            "precision_macro": mte["precision_macro"],
            "recall_macro": mte["recall_macro"],
            "f1_macro": mte["f1_macro"],
            "balanced_accuracy": mte["balanced_accuracy"],
            "mcc": mte["mcc"],
            "kappa": mte["kappa"],
            "auc_ovr": mte["auc_ovr"],
            "auc_ovo": mte["auc_ovo"],
            "brier_mc": mte["brier_mc"],
            "ece_macro": mte["ece_macro"],
            "log_loss_ce": mte["log_loss_ce"],
        })

        mem_gc()

    # ===== Combined overlays (MC suffix) =====
    finalize_combined_overlays_MC(results_root, combined_curves, combined_rows,
                                  feature_cols, model_to_fi, top_n=int(args.fi_top_n),
                                  pos_class=int(args.pos_class_for_thresholds))
    finalize_threshold_plots_MC(results_root, combined_curves, y_val, pos_class=int(args.pos_class_for_thresholds))
    build_combined_feature_importance_plots_MC(results_root, feature_cols, model_to_fi, top_n=int(args.fi_top_n))

    print("\n[DONE] Models trained with upgraded outputs.")
    print(f"  Models : {models_root}")
    print(f"  Results: {results_root}")

if __name__ == "__main__":
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    main()
