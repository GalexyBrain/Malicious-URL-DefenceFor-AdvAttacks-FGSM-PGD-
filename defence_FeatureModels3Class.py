#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Three-class (0=benign, 1=malicious, 2=ADV) evaluation — Q1-augmented, additive only.

Additions (preserving original outputs & file names):
- Leakage-safe τ selection: NAT calibration split (held-out) for the LightGBM gate.
- Split-conformal-style τ: calibrated τ to cap accepted NAT risk.
- Risk–coverage frontier CSV + plots per gate.
- Gate baselines: MSP (max softmax prob over NAT classes) and Energy (logsumexp of logits) when logits are available.
- Accepted-set calibration: ECE/Brier on accepted NAT; ADV-accept & NAT-block rates.
- End-to-end adaptive attacks on the composed system (gate + 3-class):
    * Differentiable surrogate for LightGBM gate (tiny MLP) + combined loss.
    * SPSA fallback if gradients are unavailable.
- Robust OOM backoff + CPU fallbacks retained.
- All original JSON/PNGs still produced (metrics_before_defense.json, metrics_after_defense.json, etc.).

Restrictions (per your request):
- Only these base (binary) models are used for attack crafting: DL-MLP, LightGBM, Random_Forest.

New (requested) outputs (no logic changed):
- results_3class/...                  ← duplicate of “no defense” (pure 3-class)
- results_3class_plus_detector/...    ← duplicate of conformal-gated (3-class + LGBM detector)
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, json, gc, warnings, argparse, hashlib, math, traceback
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Threads / CUDA alloc
os.environ.setdefault("OMP_NUM_THREADS", str(os.cpu_count() or 1))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay, RocCurveDisplay, roc_auc_score,
    classification_report, confusion_matrix
)
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------- Config --------------------
CFG = {
    # DATA / MODELS
    "models_dir": "models",                         # base (binary) models live here
    "defense_dir": "models/Defense-LGBM",          # gate
    "dataset_csv": "features_extracted.csv",       # dataset used by base models (label in {0,1})
    "label_col": "label",

    # RESULTS (original tree + new clean trees)
    "results_dir": "results_defence_features_3class",
    "results_dir_3class": "results_3class",
    "results_dir_3class_plus_detector": "results_3class_plus_detector",

    # ATTACK budget in z-space of base scaler
    "eps_z": 0.40,
    "alpha_z": 0.10,
    "steps_fgsm": 1,
    "steps_pgd": 5,

    # surrogate for non-linear sklearn base attacks
    "surrogate_hidden": 256,
    "surrogate_epochs_global": 5,
    "surrogate_epochs_finetune": 6,
    "surrogate_batch": 4096,
    "surrogate_sample_limit": 150_000,

    # batching
    "eval_bs": 100_000,     # sklearn/defense scoring
    "lin_attack_bs": 32_768,
    "bb_attack_bs": 8_192,
    "torch_eval_bs": 8192,   # torch eval (auto-backoff + CPU fallback on OOM)
    "torch_attack_bs": 4096, # torch attack (auto-backoff + CPU fallback on OOM)

    # device
    "use_gpu": True,

    # stream composition
    "NAT_FRAC": 0.5,
    "STREAM_SIZE": 1_000_000,

    # 3-class pipeline (ADV-aware)
    "models_base3_dir": "models_base3",
    "tau_quantile_3c": 0.995,    # legacy: aim ~99.5% NAT pass on NAT dist
    "tau_abs_3c": None,          # if set, overrides quantile
    "tau_target_accept_nat": None,# legacy target NAT accept (quantile-based)
    "classifier3c": "auto",      # override; otherwise pair with base
    "threeclass_eval": True,

    # verbosity / speed
    "use_progress": True,
    "use_plots": True,
    "use_amp": False,
    "cache_adv": False,

    # LightGBM booster fast predict
    "FAST_PREDICT": False,
    "EARLY_STOP_MARGIN": 2.0,

    # Calibration / conformal-style selection
    "nat_calib_ratio": 0.2,      # NAT held-out fraction for τ selection / risk calibration
    "target_risk_nat": 0.05,     # desired max misclassification rate on accepted NAT (empirical split conformal)
    "conformal_delta": 0.05,     # slack for empirical bound

    # Risk–coverage plotting grid
    "tau_grid_size": 50,

    # Adaptive attack (end-to-end)
    "adaptive_attack": True,
    "adaptive_steps": 20,
    "adaptive_alpha": 0.02,      # step size in z-space for combined loss
    "adaptive_lambda_gate": 3.0, # strength for pushing below τ on surrogate gate
    "adaptive_margin": 0.02,     # margin below τ for gate loss
    "spsa_iters": 100,           # fallback SPSA iters
    "spsa_delta": 0.01,
    "spsa_lr": 0.05,

    # RNG
    "seed": 42,
}

LABELS3 = [0, 1, 2]
LABELS_NAT = [0, 1]        # for per-class metrics averaging
NAT_CM_LABELS = [0, 1, 2]  # for confusion so preds==2 are counted

# -------------------- Helpers --------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_text(txt, p: Path):
    p.write_text(txt, encoding="utf-8")

def _file_md5(path: Path) -> str:
    if not path.exists(): return "missing"
    import hashlib as _hashlib
    md5 = _hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1<<20), b""):
            md5.update(chunk)
    return md5.hexdigest()

def plot_confusion_any(y_true, y_pred, out_path: Path, title: str, labels):
    if len(y_true) == 0: return
    fig, ax = plt.subplots(figsize=(6, 5.5))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=labels, ax=ax)
    ax.set_title(title); fig.tight_layout(); fig.savefig(out_path, dpi=200); plt.close(fig)

def plot_confusion_3c(y_true, y_pred, out_path: Path, title: str):
    plot_confusion_any(y_true, y_pred, out_path, title, LABELS3)

def plot_confusion_nat(y_true, y_pred, out_path: Path, title: str):
    plot_confusion_any(y_true, y_pred, out_path, title, NAT_CM_LABELS)

def plot_roc_multiclass(y_true, prob_mat, out_path: Path, title: str):
    try:
        y_true = np.asarray(y_true)
        if prob_mat is None or len(y_true) == 0: return
        fig, ax = plt.subplots(figsize=(7, 5.5))
        ok = False
        for cls in LABELS3:
            y_bin = (y_true == cls).astype(int)
            try:
                RocCurveDisplay.from_predictions(y_bin, prob_mat[:, cls], name=f"class {cls}", ax=ax)
                ok = True
            except Exception:
                pass
        if ok:
            ax.set_title(title); fig.tight_layout(); fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass

def cm_as_dict(y_true, y_pred, labels):
    """Return confusion + accuracy_from_confusion using len(y_true) as denominator."""
    y_true = np.asarray(y_true); y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    total = int(y_true.shape[0])
    acc_from_cm = float(np.trace(cm) / max(1, total))
    return {"labels": list(map(int, labels)), "matrix": cm.tolist(), "total": total, "accuracy_from_confusion": acc_from_cm}

def ece_score(y_true, prob_vec, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
    prob_vec = np.asarray(prob_vec).clip(1e-7, 1-1e-7)
    bins = np.linspace(0.0, 1.0, n_bins+1)
    ece = 0.0; total = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (prob_vec >= lo) & ((prob_vec < hi) if i < n_bins-1 else (prob_vec <= hi))
        if not np.any(mask): continue
        acc = np.mean(y_true[mask])
        conf = np.mean(prob_vec[mask])
        ece += (np.sum(mask)/total) * abs(acc - conf)
    return float(ece)

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def _logsumexp(a: np.ndarray, axis=1):
    m = np.max(a, axis=axis, keepdims=True)
    s = np.exp(a - m).sum(axis=axis, keepdims=True)
    return (np.log(s) + m).squeeze(axis)

# -------------------- Load globals --------------------
def load_globals(models_root: Path):
    gdir = models_root / "_global"
    scaler: StandardScaler = joblib.load(gdir / "scaler.joblib")
    feat_info = json.loads((gdir / "feature_columns.json").read_text(encoding="utf-8"))
    feature_cols = feat_info["feature_columns"] if isinstance(feat_info, dict) else feat_info
    return scaler, list(feature_cols)

def load_dataset(csv_path: str, feature_cols: list[str], label_col: str):
    df = pd.read_csv(csv_path)
    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Feature column '{c}' missing in {csv_path}")
    X_raw = df[feature_cols].to_numpy(dtype=np.float32)
    y_base = pd.to_numeric(df[label_col], errors="coerce").fillna(0).astype(int).to_numpy()
    return X_raw, y_base

# -------------------- Base (binary) models --------------------
def unwrap_calibrated(estimator):
    if isinstance(estimator, CalibratedClassifierCV):
        if hasattr(estimator, "calibrated_classifiers_") and estimator.calibrated_classifiers_:
            inner = estimator.calibrated_classifiers_[0]
            base = getattr(inner, "estimator", None) or getattr(inner, "base_estimator", None)
            if base is not None: return base
    return estimator

def linear_params(estimator):
    est = unwrap_calibrated(estimator)
    if hasattr(est, "coef_") and hasattr(est, "intercept_"):
        w = np.array(est.coef_, dtype=np.float32).reshape(1, -1)
        b = np.array(getattr(est, "intercept_", np.zeros(w.shape[1], dtype=np.float32)), dtype=np.float32).reshape(-1)
        w = (w[0] if w.shape[0] == 1 else np.mean(w, axis=0)).astype(np.float32, copy=False)
        b = np.float32(b[0] if b.size else 0.0)
        return w, b
    raise RuntimeError("Linear parameters not found for white-box attack.")

def attack_linear_fgsm_pgd(estimator, Xz, y, steps, eps, alpha, loss_kind="logistic", batch=32_768,
                           desc="lin/attack", use_progress=True):
    w, b = linear_params(estimator)
    sign_w = np.sign(w).astype(np.float32)
    X_adv = Xz.copy().astype(np.float32, copy=False); X0 = Xz.astype(np.float32, copy=False)
    y = y.astype(np.float32, copy=False)
    if loss_kind == "hinge": ypm = np.where(y == 1, 1.0, -1.0).astype(np.float32)
    N = Xz.shape[0]; steps = max(1, int(steps)); eps = np.float32(eps); alpha = np.float32(alpha)
    s = 0; bs = int(batch); total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while s < N:
        e = min(N, s + bs)
        Xa = X_adv[s:e]; Xb = X0[s:e]
        for _ in range(steps):
            z = Xa @ w + b
            if loss_kind == "logistic":
                p = 1.0 / (1.0 + np.exp(-z))
                residual = p - y[s:e]
                step = (np.sign(residual)[:, None] * sign_w[None, :])
            else:
                ypm_se = ypm[s:e]
                m = ypm_se * z
                mask = (m < 1.0).astype(np.float32)
                step = mask[:, None] * ((-ypm_se)[:, None] * sign_w[None, :])
            Xa = Xa + alpha * step
            Xa = np.clip(Xa, Xb - eps, Xb + eps)
        X_adv[s:e] = Xa; s = e; pbar.update(1)
    pbar.close()
    return X_adv

class MLPNetBin(nn.Module):
    def __init__(self, in_dim: int, hidden=(256,128,64), p_drop=0.2):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(inplace=True), nn.BatchNorm1d(h), nn.Dropout(p_drop)]; prev=h
        self.backbone = nn.Sequential(*layers); self.head = nn.Linear(prev, 1)
    def forward(self, x): return self.head(self.backbone(x)).squeeze(-1)

@torch.no_grad()
def eval_torch_bin(model, Xz, device, batch_size, desc, use_progress, use_amp):
    N = Xz.shape[0]; out=[]; i=0
    bs = max(1, int(batch_size))
    local_device = device
    model.eval()
    while True:
        try:
            pbar = tqdm(total=math.ceil(N/bs), desc=desc, disable=not use_progress)
            while i < N:
                j = min(N, i+bs)
                xb = torch.from_numpy(Xz[i:j]).to(local_device).float()
                with torch.cuda.amp.autocast(enabled=(use_amp and local_device.type=="cuda")):
                    logits = model(xb).float()
                probs = torch.sigmoid(logits).detach().cpu().numpy()
                del xb, logits
                out.append(probs); i=j; pbar.update(1)
            pbar.close()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and local_device.type == "cuda":
                cleanup_cuda()
                if bs > 1:
                    bs = max(1, bs // 2)
                    tqdm.write(f"[WARN] OOM in {desc}; retry with batch_size={bs}")
                    continue
                else:
                    tqdm.write(f"[WARN] OOM in {desc} at batch=1 → falling back to CPU.")
                    model.to(torch.device("cpu")); local_device = torch.device("cpu")
                    bs = max(256, bs)
                    continue
            raise
    return np.concatenate(out, axis=0)

def attack_tabular_torch_batched(model, Xz, y, eps, alpha, steps, device, z_min, z_max, batch_size,
                                 desc="torch/attack", use_progress=True, use_amp=False):
    N, F = Xz.shape
    adv_list = []; bs = max(1, int(batch_size))
    steps_eff = 1 if steps == 0 else steps
    alpha_eff = eps if steps == 0 else alpha
    local_device = device
    zmin_t = torch.from_numpy(z_min).to(local_device)
    zmax_t = torch.from_numpy(z_max).to(local_device)
    loss_fn = nn.BCEWithLogitsLoss()
    ptr = 0
    while ptr < N:
        end = min(N, ptr + bs)
        try:
            X = torch.from_numpy(Xz[ptr:end]).to(local_device).float()
            Y = torch.from_numpy(y[ptr:end]).to(local_device).float()
            X0 = X.clone().detach()
            X_adv = X.clone().detach()
            for _ in range(steps_eff):
                X_adv.requires_grad_(True)
                with torch.cuda.amp.autocast(enabled=(use_amp and local_device.type=="cuda")):
                    logits = model(X_adv).float()
                    loss = loss_fn(logits, Y)
                model.zero_grad(set_to_none=True)
                if X_adv.grad is not None: X_adv.grad.zero_()
                loss.backward()
                with torch.no_grad():
                    X_adv = X_adv + alpha_eff * X_adv.grad.sign()
                    X_adv = torch.clamp(X_adv, X0 - eps, X0 + eps)
                    X_adv = torch.max(torch.min(X_adv, zmax_t), zmin_t).detach()
            adv_list.append(X_adv.detach().cpu().numpy()); ptr = end
        except RuntimeError as e:
            if "CUDA out of memory" in str(e) or "out of memory" in str(e).lower():
                cleanup_cuda()
                if local_device.type == "cuda":
                    if bs > 1:
                        bs = max(1, bs // 2)
                        tqdm.write(f"[WARN] OOM during {desc}; retry with batch_size={bs}")
                        continue
                    else:
                        tqdm.write(f"[WARN] OOM during {desc} at batch=1 → falling back to CPU.")
                        model.to(torch.device("cpu"))
                        local_device = torch.device("cpu")
                        zmin_t = torch.from_numpy(z_min)
                        zmax_t = torch.from_numpy(z_max)
                        bs = max(128, bs)
                        continue
                else:
                    raise
            else:
                raise
    return np.concatenate(adv_list, axis=0)

# -------------------- Defense gate --------------------
def _booster_from_defense(defense):
    try:
        import lightgbm as lgb  # noqa
    except Exception:
        return None
    if hasattr(defense, "booster_"):
        b = getattr(defense, "booster_", None)
        if b is not None:
            return b
    try:
        import lightgbm as lgb  # noqa
        if isinstance(defense, lgb.Booster):
            return defense
    except Exception:
        pass
    return None

def _predict_with_defense(def_model, X, fast_predict=False, early_stop_margin=2.0):
    booster = _booster_from_defense(def_model)
    if booster is not None:
        kw = dict(raw_score=False)
        if fast_predict:
            kw.update(dict(pred_early_stop=True, pred_early_stop_margin=early_stop_margin))
        p = booster.predict(X, **kw)  # (N,) prob of class-1
        return np.asarray(p)
    if hasattr(def_model, "predict_proba"):
        p = def_model.predict_proba(X)
        if isinstance(p, list):
            p = p[0]
        p = np.asarray(p)
        if p.ndim == 2 and p.shape[1] == 2:
            return p[:, 1]
        if p.ndim == 2 and p.shape[1] > 1:
            return p[:, -1]
        return p.reshape(-1)
    if hasattr(def_model, "predict"):
        try:
            p = def_model.predict(X, raw_score=False)
            return np.asarray(p).reshape(-1)
        except Exception:
            pass
        return np.asarray(def_model.predict(X)).reshape(-1)
    if hasattr(def_model, "decision_function"):
        z = np.asarray(def_model.decision_function(X)).reshape(-1)
        return 1.0 / (1.0 + np.exp(-z))
    raise RuntimeError("Defense model does not support predict/predict_proba/decision_function.")

def defense_p_adv(def_model, X_raw, batch=100_000, calibrator=None, desc="defense/p_adv",
                  use_progress=True, fast_predict=False, early_stop_margin=2.0):
    N = X_raw.shape[0]; probs = np.empty(N, dtype=np.float32)
    bs = max(10_000, int(batch)); i = 0; total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while i < N:
        j = min(N, i + bs)
        xb = np.ascontiguousarray(X_raw[i:j], dtype=np.float32)
        try:
            p1 = _predict_with_defense(def_model, xb, fast_predict=fast_predict, early_stop_margin=early_stop_margin)
        except Exception as ex:
            if "MemoryError" in str(ex) or bs > 10_000:
                bs = max(10_000, bs // 2); cleanup_cuda()
                total = math.ceil((N - i) / bs); pbar.total = pbar.n + total; pbar.refresh(); continue
            pbar.close(); raise
        p1 = np.asarray(p1)
        if p1.ndim == 2 and p1.shape[1] == 2: p1 = p1[:,1]
        if p1.min() < 0 or p1.max() > 1: p1 = 1.0 / (1.0 + np.exp(-p1))
        if calibrator is not None:
            try: p1 = calibrator.predict(p1)
            except Exception: pass
        probs[i:j] = p1.astype(np.float32, copy=False); i = j; pbar.update(1)
    pbar.close(); return probs

def defense_p_adv_from_z(def_model, scaler: StandardScaler, Xz, batch=100_000, calibrator=None,
                         desc="defense/p_adv(z->raw)", use_progress=True,
                         fast_predict=False, early_stop_margin=2.0):
    N = Xz.shape[0]; out = np.empty(N, dtype=np.float32)
    bs = max(10_000, int(batch)); i = 0; total = math.ceil(N / bs)
    pbar = tqdm(total=total, desc=desc, disable=not use_progress)
    while i < N:
        j = min(N, i + bs)
        xb_raw = scaler.inverse_transform(Xz[i:j]).astype(np.float32, copy=False)
        xb_raw = np.ascontiguousarray(xb_raw)
        p1 = _predict_with_defense(def_model, xb_raw, fast_predict=fast_predict, early_stop_margin=early_stop_margin)
        p1 = np.asarray(p1)
        if p1.ndim == 2 and p1.shape[1] == 2: p1 = p1[:,1]
        if p1.min() < 0 or p1.max() > 1: p1 = 1.0 / (1.0 + np.exp(-p1))
        if calibrator is not None:
            try: p1 = calibrator.predict(p1)
            except Exception: pass
        out[i:j] = p1.astype(np.float32, copy=False); i = j; pbar.update(1)
    pbar.close(); return out

# τ helpers (legacy + split-calibration)
def choose_tau_from_nat_full(p_adv_nat_full: np.ndarray, q: float, tau_abs=None, target_accept=None) -> float:
    if tau_abs is not None:
        return float(np.clip(tau_abs, 0.0, 1.0))
    q = float(np.clip(q, 0.50, 0.9999))
    tau = float(np.quantile(p_adv_nat_full, q)) if len(p_adv_nat_full) else 0.5
    if target_accept is not None and len(p_adv_nat_full):
        ta = float(np.clip(target_accept, 0.5, 0.9999))
        tau = float(np.quantile(p_adv_nat_full, ta))
    return tau

def split_nat_calibration(p_adv_nat: np.ndarray, y_nat: np.ndarray, yhat_nat_probs_3c: np.ndarray,
                          target_risk=0.05, delta=0.05, grid=50):
    """
    Split-based selection of τ that aims to keep accepted NAT risk <= target_risk.
    Risk measured as 1 - accuracy on {0,1} among accepted NAT.
    Returns: tau_star, table (DataFrame) with coverage, risk columns over τ grid.
    """
    assert p_adv_nat.shape[0] == y_nat.shape[0] == yhat_nat_probs_3c.shape[0]
    qs = np.linspace(0.80, 0.9999, grid)  # conservative tail
    taus = np.quantile(p_adv_nat, qs)
    recs = []
    for tau in taus:
        acc_mask = (p_adv_nat < tau)
        n = int(acc_mask.sum())
        if n == 0:
            recs.append({"tau": float(tau), "coverage_nat": 0.0, "risk_nat": 1.0})
            continue
        y_acc = y_nat[acc_mask]
        p_acc = yhat_nat_probs_3c[acc_mask]
        yhat_acc = p_acc.argmax(axis=1)
        # risk on NAT is 1 - accuracy (counting pred==2 as error)
        nat_mask_ok = np.isin(yhat_acc, [0,1])
        acc = float(np.mean((yhat_acc == y_acc) & nat_mask_ok))
        risk = 1.0 - acc
        risk = min(1.0, risk + float(delta))  # conservative slack
        cov = float(n / max(1, y_nat.size))
        recs.append({"tau": float(tau), "coverage_nat": cov, "risk_nat": risk})
    df = pd.DataFrame(recs)
    ok = df[df["risk_nat"] <= float(target_risk)]
    tau_star = float(ok["tau"].iloc[0]) if len(ok) else float(df["tau"].iloc[-1])
    return tau_star, df

# -------------------- 3-class model helpers --------------------
def _load_base3_globals(root: Path):
    gdir = root / "_global"
    scaler3c: StandardScaler = joblib.load(gdir / "scaler.joblib")
    info = json.loads((gdir / "feature_columns.json").read_text(encoding="utf-8"))
    feat3c = info["feature_columns"] if isinstance(info, dict) else info
    return scaler3c, list(feat3c)

def _align_X_for_3c(X_raw_base: np.ndarray, feat_base: list[str], feat3c: list[str]) -> np.ndarray:
    set_b, set_c = set(feat_base), set(feat3c)
    if set_b != set_c:
        missing_b = sorted(set_c - set_b)
        missing_c = sorted(set_b - set_c)
        raise ValueError(f"Feature mismatch base vs 3-class. Missing in base: {missing_b[:6]}..., missing in 3c: {missing_c[:6]}...")
    pos = [feat_base.index(c) for c in feat3c]  # reorder to 3c order
    return X_raw_base[:, pos]

class MLPNetMulti(nn.Module):
    def __init__(self, in_dim: int, out_classes=3, hidden=(256,128,64), p_drop=0.2):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(inplace=True), nn.BatchNorm1d(h), nn.Dropout(p_drop)]; prev=h
        self.backbone = nn.Sequential(*layers); self.head = nn.Linear(prev, out_classes)
    def forward(self, x): return self.head(self.backbone(x))

def _torch_logits_and_probs(model, Xz, device, batch_size, desc, use_progress, use_amp):
    """Inference with logits + softmax probs, OOM-safe."""
    N = Xz.shape[0]; probs=[]; logits_all=[]; i=0
    bs = max(1, int(batch_size))
    local_device = device
    model.eval()
    while True:
        try:
            pbar = tqdm(total=math.ceil(N/bs), desc=desc, disable=not use_progress)
            with torch.no_grad():
                while i < N:
                    j = min(N, i+bs)
                    xb = torch.from_numpy(Xz[i:j]).to(local_device).float()
                    with torch.cuda.amp.autocast(enabled=(use_amp and local_device.type=="cuda")):
                        logits = model(xb).float()
                    p = torch.softmax(logits, dim=-1).detach().cpu().numpy()
                    logits_all.append(logits.detach().cpu().numpy()); probs.append(p)
                    del xb, logits
                    i=j; pbar.update(1)
            pbar.close()
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and local_device.type == "cuda":
                cleanup_cuda()
                if bs > 1:
                    bs = max(1, bs // 2)
                    tqdm.write(f"[WARN] OOM in {desc}; retry with batch_size={bs}")
                    continue
                else:
                    tqdm.write(f"[WARN] OOM in {desc} at batch=1 → falling back to CPU.")
                    model.to(torch.device("cpu")); local_device = torch.device("cpu")
                    bs = max(256, bs)
                    continue
            raise
    return np.concatenate(logits_all, axis=0), np.concatenate(probs, axis=0)

def _torch_softmax_probs(model, Xz, device, batch_size, desc, use_progress, use_amp):
    _, P = _torch_logits_and_probs(model, Xz, device, batch_size, desc, use_progress, use_amp)
    return P

def _reorder_probs_to_labels(p: np.ndarray, estimator, labels=(0,1,2)):
    try:
        classes = getattr(estimator, "classes_", None)
        if classes is None: return p
        classes = list(map(int, list(classes)))
        want = list(labels)
        colmap = {c:i for i,c in enumerate(classes)}
        idx = [colmap.get(c, None) for c in want]
        out = np.zeros((p.shape[0], len(want)), dtype=np.float32)
        for j, src in enumerate(idx):
            if src is not None:
                out[:, j] = p[:, src]
            else:
                out[:, j] = 0.0
        return out
    except Exception:
        return p

def _chunked_probs_multi(est, X, chunk, desc, use_progress):
    N = X.shape[0]; bs = max(4096, int(chunk)); i=0; outs=[]
    pbar = tqdm(total=math.ceil(N/bs), desc=desc, disable=not use_progress)
    while i < N:
        j = min(N, i+bs); xb = X[i:j]
        if hasattr(est, "predict_proba"):
            p = np.asarray(est.predict_proba(xb))
            if isinstance(p, list): p = p[0]
        else:
            z = np.asarray(est.decision_function(xb))
            e = np.exp(z - z.max(axis=1, keepdims=True))
            p = e / np.clip(e.sum(axis=1, keepdims=True), 1e-12, None)
        p = p.astype(np.float32, copy=False)
        p = _reorder_probs_to_labels(p, est, labels=LABELS3)
        outs.append(p); i=j; pbar.update(1)
    pbar.close(); return np.concatenate(outs, axis=0)

def _sk_multiclass_logits(est, X) -> np.ndarray | None:
    """Try to extract raw logits/margins for Energy gate from sklearn-like estimators."""
    try:
        # LightGBM scikit wrapper supports raw_score=True for margins
        z = est.predict(X, raw_score=True)
        if isinstance(z, list): z = z[0]
        z = np.asarray(z, dtype=np.float32)
        if z.ndim == 1:  # binary margin; expand to 2-class then pad to 3? Not applicable here.
            return None
        return z
    except TypeError:
        pass
    if hasattr(est, "decision_function"):
        try:
            z = est.decision_function(X)
            z = np.asarray(z, dtype=np.float32)
            return z
        except Exception:
            return None
    return None

# MSP / Energy gate scores from a 3-class model
def msp_scores_nat(p3_probs: np.ndarray):
    return p3_probs[:, :2].max(axis=1).astype(np.float32)

def energy_scores_from_logits(logits: np.ndarray | None):
    if logits is None: return None
    return (-_logsumexp(logits, axis=1)).astype(np.float32)

# -------------------- Utility --------------------
def pick_stream_indices(N: int, stream_size, nat_frac: float, rng: np.random.RandomState):
    if stream_size is None:
        adv_idx = np.arange(N)
        nat_count = int(np.floor(N * nat_frac))
        nat_idx = rng.choice(N, size=nat_count, replace=False)
    else:
        total = int(max(1, stream_size))
        nat_count = int(np.clip(int(total * nat_frac), 1, total-1))
        adv_count = total - nat_count
        adv_idx = rng.choice(N, size=adv_count, replace=False)
        nat_idx = rng.choice(N, size=nat_count, replace=False)
    return adv_idx, nat_idx

def adv_cache_path(root: Path, base_name: str, which: str, adv_idx: np.ndarray, steps: int, eps: float, alpha: float):
    h = hashlib.md5(adv_idx.tobytes()).hexdigest()[:12]
    fname = f"adv_{base_name}_{which}_N{adv_idx.size}_steps{steps}_eps{eps:.3f}_alpha{alpha:.3f}_{h}.npz"
    return root / "_cache" / fname

# STRICT one-to-one pairing: every base uses its 3-class counterpart with the SAME folder name
BASE_TO_3C = {
    "DL-MLP": "DL-MLP",
    "Random_Forest": "Random_Forest",
    "LightGBM": "LightGBM",
}

def _find_model_file(dirpath: Path):
    pt = dirpath / "model.pt"
    if pt.exists(): return "torch", pt
    for nm in ["model.joblib", "model.pkl"]:
        p = dirpath / nm
        if p.exists(): return "sk", p
    return None, None

def load_threeclass_for_name(base3_root: Path, name3: str, feat3c_len: int, device):
    kind, path = _find_model_file(base3_root / name3)
    assert kind is not None, f"3-class model not found for '{name3}' under {base3_root}"
    if kind == "torch":
        if name3 == "DL-MLP":
            model3 = MLPNetMulti(in_dim=feat3c_len, out_classes=3).to(device)
        else:
            raise RuntimeError(f"Unexpected torch 3-class model folder '{name3}'")
        state = torch.load(path, map_location=device)
        model3.load_state_dict(state); model3.eval()
        return ("torch", name3, model3)
    else:
        return ("sk", name3, joblib.load(path))

# -------------------- Adaptive composed attack --------------------
class GateSurrogate(nn.Module):
    """Tiny MLP to mimic p_adv (LightGBM gate) for differentiable attacks."""
    def __init__(self, D, H=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D, H), nn.ReLU(),
            nn.Linear(H, H), nn.ReLU(),
            nn.Linear(H, 1),
        )
    def forward(self, x): return self.net(x).squeeze(-1)  # logits; pass through sigmoid for prob

def train_gate_surrogate(X_raw, p_adv, device, epochs=5, batch=8192, lr=3e-4):
    model = GateSurrogate(D=X_raw.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X_raw).float(), torch.from_numpy(p_adv.astype(np.float32)))
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True, drop_last=False)
    best=None; best_loss=float("inf")
    for ep in range(1, epochs+1):
        run=0.0; n=0
        for xb, yb in dl:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb)
            loss.backward(); opt.step()
            run += loss.item()*xb.size(0); n+=xb.size(0)
        avg = run/max(1,n)
        if avg < best_loss - 1e-6:
            best_loss=avg; best={k:v.detach().cpu() for k,v in model.state_dict().items()}
    if best is not None: model.load_state_dict({k:v.to(device) for k,v in best.items()})
    model.eval(); return model

def composed_attack_dl(
    Xz_in, scaler_base: StandardScaler, model3_torch, gate_surrogate, y_true3, tau, eps, alpha, steps,
    device, z_min, z_max, lambda_gate=3.0, margin=0.02, use_amp=False
):
    """
    End-to-end attack that tries to:
      (1) fool the gate (surrogate): sigmoid(g(x_raw)) < tau - margin, and
      (2) cause the 3-class DL model to predict {0,1} if y=2, or the wrong NAT class if y in {0,1}.
    Optimizes in z-space; converts to raw for gate surrogate each step.

    [FIX] Make inverse-transform differentiable (no numpy): x_raw = z * scale + mean
    """
    Xz = torch.from_numpy(Xz_in.copy()).to(device).float()
    X0 = Xz.clone().detach()
    X_adv = Xz.clone().detach()
    y = torch.from_numpy(y_true3.astype(np.int64)).to(device)
    zmin_t = torch.from_numpy(z_min).to(device); zmax_t=torch.from_numpy(z_max).to(device)

    # Precompute scaler params as torch tensors for differentiability
    scale_t = torch.from_numpy(np.asarray(scaler_base.scale_, dtype=np.float32)).to(device)
    mean_t  = torch.from_numpy(np.asarray(scaler_base.mean_, dtype=np.float32)).to(device)

    opt_steps = max(1, int(steps))
    for _ in range(opt_steps):
        X_adv.requires_grad_(True)
        with torch.cuda.amp.autocast(enabled=use_amp and device.type=="cuda"):
            # 3-class loss
            logits3 = model3_torch(X_adv)  # (N,3)
            probs3 = torch.softmax(logits3, dim=-1)
            q = torch.zeros_like(probs3)
            mask_adv = (y == 2)
            q[mask_adv, 0] = 0.5; q[mask_adv, 1] = 0.5
            q[~mask_adv, 1 - y[~mask_adv]] = 1.0
            loss_f = F.kl_div((probs3+1e-12).log(), q, reduction="batchmean")

            # Gate surrogate loss (raw space) — differentiable inverse transform
            X_raw = X_adv * scale_t + mean_t
            logits_gate = gate_surrogate(X_raw)
            p_gate = torch.sigmoid(logits_gate)
            gate_hinge = torch.clamp(p_gate - float(tau) + float(margin), min=0.0)
            loss_g = gate_hinge.mean()

            loss = loss_f + float(lambda_gate) * loss_g

        model3_torch.zero_grad(set_to_none=True)
        if X_adv.grad is not None: X_adv.grad.zero_()
        loss.backward()

        with torch.no_grad():
            X_adv = X_adv + float(alpha) * torch.sign(X_adv.grad)
            X_adv = torch.clamp(X_adv, X0 - float(eps), X0 + float(eps))
            X_adv = torch.max(torch.min(X_adv, zmax_t), zmin_t).detach()
    return X_adv.detach().cpu().numpy()

def spsa_attack(Xz_in, loss_fn, steps=100, lr=0.05, delta=0.01, eps=0.40, z_min=None, z_max=None, rng=None):
    """Model-agnostic SPSA in z-space; keeps within L_inf eps and feature bounds."""
    if rng is None:
        rng = np.random.RandomState(123)
    X0 = Xz_in.copy()
    X = X0.copy()
    N, F = X.shape
    for t in range(int(steps)):
        v = rng.choice([-1.0, 1.0], size=X.shape).astype(np.float32)
        loss_plus  = loss_fn(np.clip(X + delta*v, X0 - eps, X0 + eps))
        loss_minus = loss_fn(np.clip(X - delta*v, X0 - eps, X0 + eps))
        ghat = (loss_plus - loss_minus)[:, None] * v / (2.0 * delta + 1e-12)
        X = X + lr * np.sign(ghat)
        X = np.clip(X, X0 - eps, X0 + eps)
        if z_min is not None and z_max is not None:
            X = np.minimum(np.maximum(X, z_min), z_max)
    return X

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="3-class evaluation (additive Q1 extensions)")
    p.add_argument("--model", default="all", choices=["all", "DL-MLP", "Random_Forest","LightGBM"])
    p.add_argument("--nat-frac", type=float, default=None)
    p.add_argument("--stream-size", type=int, default=None)
    p.add_argument("--eps", type=float, default=None)
    p.add_argument("--alpha", type=float, default=None)
    p.add_argument("--pgd-steps", type=int, default=None)
    p.add_argument("--fgsm-steps", type=int, default=None)
    p.add_argument("--eval-bs", type=int, default=None)
    p.add_argument("--lin-bs", type=int, default=None)
    p.add_argument("--bb-bs", type=int, default=None)
    p.add_argument("--torch-eval-bs", type=int, default=None)
    p.add_argument("--torch-attack-bs", type=int, default=None)
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-progress", action="store_true")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--cache-adv", action="store_true")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--fast-predict", action="store_true")
    p.add_argument("--seed", type=int, default=None)

    # Q1 knobs
    p.add_argument("--nat-calib-ratio", type=float, default=None)
    p.add_argument("--target-risk-nat", type=float, default=None)
    p.add_argument("--conformal-delta", type=float, default=None)
    p.add_argument("--tau-grid", type=int, default=None)
    p.add_argument("--disable-adaptive", action="store_true")
    p.add_argument("--classifier3c", type=str, default=None)
    return p.parse_args()

# -------- helpers defined late to keep code together --------
def _per_class_metrics(y_true, y_pred, cls_labels):
    out = {}
    prc = precision_score(y_true, y_pred, labels=cls_labels, average=None, zero_division=0)
    rec = recall_score(y_true, y_pred, labels=cls_labels, average=None, zero_division=0)
    f1  = f1_score(y_true, y_pred, labels=cls_labels, average=None, zero_division=0)
    for i, c in enumerate(cls_labels):
        out[str(c)] = {
            "precision": float(prc[i]),
            "recall": float(rec[i]),
            "f1": float(f1[i]),
            "support": int((y_true == c).sum()),
        }
    return out

# -------------------- Main --------------------
def main():
    args = parse_args()
    if args.nat_frac is not None:  CFG["NAT_FRAC"] = float(args.nat_frac)
    if args.stream_size is not None: CFG["STREAM_SIZE"] = int(args.stream_size)
    if args.eps is not None:       CFG["eps_z"] = float(args.eps)
    if args.alpha is not None:     CFG["alpha_z"] = float(args.alpha)
    if args.pgd_steps is not None: CFG["steps_pgd"] = int(args.pgd_steps)
    if args.fgsm_steps is not None: CFG["steps_fgsm"] = int(args.fgsm_steps)
    if args.eval_bs is not None:   CFG["eval_bs"] = int(args.eval_bs)
    if args.lin_bs is not None:    CFG["lin_attack_bs"] = int(args.lin_bs)
    if args.bb_bs is not None:     CFG["bb_attack_bs"] = int(args.bb_bs)
    if args.torch_eval_bs is not None:   CFG["torch_eval_bs"] = int(args.torch_eval_bs)
    if args.torch_attack_bs is not None: CFG["torch_attack_bs"] = int(args.torch_attack_bs)
    if args.no_plots:              CFG["use_plots"] = False
    if args.no_progress:           CFG["use_progress"] = False
    if args.amp:                   CFG["use_amp"] = True
    if args.cache_adv:             CFG["cache_adv"] = True
    if args.cpu:                   CFG["use_gpu"] = False
    if args.fast_predict:          CFG["FAST_PREDICT"] = True
    if args.seed is not None:      CFG["seed"] = int(args.seed)
    if args.nat_calib_ratio is not None: CFG["nat_calib_ratio"] = float(args.nat_calib_ratio)
    if args.target_risk_nat is not None: CFG["target_risk_nat"] = float(args.target_risk_nat)
    if args.conformal_delta is not None: CFG["conformal_delta"] = float(args.conformal_delta)
    if args.tau_grid is not None:  CFG["tau_grid_size"] = int(args.tau_grid)
    if args.disable_adaptive:      CFG["adaptive_attack"] = False
    if args.classifier3c is not None: CFG["classifier3c"] = args.classifier3c

    rng = np.random.RandomState(CFG["seed"])
    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    print(f"[INFO] Using device: {device}")

    # Load base globals & dataset
    models_root = Path(CFG["models_dir"])
    results_root = ensure_dir(Path(CFG["results_dir"]))
    results_root_3c  = ensure_dir(Path(CFG["results_dir_3class"]))
    results_root_3cd = ensure_dir(Path(CFG["results_dir_3class_plus_detector"]))
    defense_root = Path(CFG["defense_dir"])

    scaler_base, feat_base = load_globals(models_root)
    X_raw, y_base = load_dataset(CFG["dataset_csv"], feat_base, CFG["label_col"])
    Xz = scaler_base.transform(X_raw).astype(np.float32, copy=False)
    N = Xz.shape[0]; z_min = Xz.min(axis=0); z_max = Xz.max(axis=0)
    print(f"[INFO] Loaded dataset: N={N:,}, D={Xz.shape[1]}")

    # Load defense & calibrator
    def_path = defense_root / "model.joblib"
    assert def_path.exists(), "Missing Defense-LGBM at models/Defense-LGBM/model.joblib"
    defense = joblib.load(def_path)
    calibrator = None
    cal_path = defense_root / "calibrator.joblib"
    if cal_path.exists():
        try:
            calibrator = joblib.load(cal_path); print("[INFO] Using isotonic calibrator for defense probabilities.")
        except Exception:
            calibrator = None

    # p_adv over full NAT pool
    p_adv_full = defense_p_adv(
        defense, X_raw, batch=CFG["eval_bs"], calibrator=calibrator,
        desc="defense p_adv(all)", use_progress=CFG["use_progress"],
        fast_predict=CFG["FAST_PREDICT"], early_stop_margin=CFG["EARLY_STOP_MARGIN"]
    )

    # Split NAT for τ selection (held-out calibration)
    nat_idx_all = np.where(np.isin(y_base, [0,1]))[0]
    rng.shuffle(nat_idx_all)
    split = int(len(nat_idx_all) * (1.0 - CFG["nat_calib_ratio"]))
    nat_eval_idx = nat_idx_all[:split]
    nat_calib_idx = nat_idx_all[split:]
    print(f"[INFO] NAT split -> calib={len(nat_calib_idx)}, eval={len(nat_eval_idx)}")

    # Legacy quantile τ (kept, on calib split)
    tau_legacy = choose_tau_from_nat_full(
        p_adv_full[nat_calib_idx], q=CFG["tau_quantile_3c"],
        tau_abs=CFG["tau_abs_3c"], target_accept=CFG["tau_target_accept_nat"]
    )

    # Load 3-class globals + reorder matrix
    base3_root = Path(CFG["models_base3_dir"])
    scaler3c, feat3c = _load_base3_globals(base3_root)
    X_raw_reordered_for_3c = _align_X_for_3c(X_raw, feat_base, feat3c)
    Xz3 = scaler3c.transform(X_raw_reordered_for_3c).astype(np.float32, copy=False)

    # Candidate base models (restricted to your 3)
    candidates = ["DL-MLP", "Random_Forest", "LightGBM"]
    if args.model != "all":
        candidates = [args.model]

    # Load available base (binary) models
    bases = []
    for name in candidates:
        mdir = models_root / name
        if not mdir.exists():
            print(f"[WARN] Skipping {name}: {mdir} not found.")
            continue
        if name == "DL-MLP":
            mdl = MLPNetBin(in_dim=Xz.shape[1]).to(device)
            state = torch.load(mdir / "model.pt", map_location=device)
            mdl.load_state_dict(state); mdl.eval()
            bases.append((name, mdl))
        else:
            mpath = mdir / "model.joblib"
            if mpath.exists():
                bases.append((name, joblib.load(mpath)))
            else:
                print(f"[WARN] Skipping {name}: {mpath} not found.")
    assert bases, "No base (binary) models found for attack crafting."

    # Pre-pick one stream (shared across bases)
    adv_idx_fixed, nat_idx_fixed = pick_stream_indices(N, CFG["STREAM_SIZE"], CFG["NAT_FRAC"], rng)

    # 3c loader cache
    threeclass_cache: dict[str, tuple[str,str,object]] = {}
    def get_threeclass_for_base(base_name: str):
        if CFG["classifier3c"] and CFG["classifier3c"] != "auto":
            name3 = CFG["classifier3c"]
        else:
            name3 = BASE_TO_3C.get(base_name, None)
            assert name3 is not None, f"No mapping from base '{base_name}' to 3-class model."
        if name3 in threeclass_cache:
            return threeclass_cache[name3]
        kind, name_loaded, est = load_threeclass_for_name(base3_root, name3, len(feat3c), device)
        threeclass_cache[name3] = (kind, name_loaded, est)
        print(f"[INFO] 3-class model for base '{base_name}': {name_loaded} ({kind})")
        return threeclass_cache[name3]

    # Convenience: NAT calib/eval in 3c z-space
    Xz3_nat_cal = Xz3[nat_calib_idx]
    Xz3_nat_eval = Xz3[nat_eval_idx]
    y_nat_cal = y_base[nat_calib_idx]
    y_nat_eval = y_base[nat_eval_idx]

    # --------- Gate surrogate (trained once over ALL data) for adaptive attack ---------
    gate_sur = None
    if CFG["adaptive_attack"]:
        print("[INFO] Training gate surrogate (global) to mimic p_adv...")
        gate_sur = train_gate_surrogate(X_raw, p_adv_full, device=device, epochs=5, batch=8192, lr=3e-4)

    # ---------- Core per-base loop ----------
    for base_name, base_est in bases:
        print(f"\n==== Using base (for ATTACK crafting only): {base_name} ====")
        out_root = ensure_dir(results_root / base_name)
        out_root_3c  = ensure_dir(results_root_3c  / base_name)
        out_root_3cd = ensure_dir(results_root_3cd / base_name)

        # Load paired 3c
        kind3, name3, est3 = get_threeclass_for_base(base_name)

        # NAT probs/logits on calibration/eval for conformal τ + baselines
        if kind3 == "torch":
            log_nat_cal, P3_nat_cal = _torch_logits_and_probs(est3, Xz3_nat_cal, device, CFG["torch_eval_bs"], f"3c/{name3} NAT calib", CFG["use_progress"], CFG["use_amp"])
            log_nat_eval, P3_nat_eval = _torch_logits_and_probs(est3, Xz3_nat_eval, device, CFG["torch_eval_bs"], f"3c/{name3} NAT eval",  CFG["use_progress"], CFG["use_amp"])
        else:
            P3_nat_cal = _chunked_probs_multi(est3, Xz3_nat_cal, CFG["eval_bs"], f"3c/{name3} NAT calib", CFG["use_progress"])
            P3_nat_eval = _chunked_probs_multi(est3, Xz3_nat_eval, CFG["eval_bs"], f"3c/{name3} NAT eval",  CFG["use_progress"])
            # try logits for Energy baseline (LightGBM supports raw_score)
            log_nat_cal = _sk_multiclass_logits(est3, Xz3_nat_cal)
            log_nat_eval = _sk_multiclass_logits(est3, Xz3_nat_eval)

        # Conformal-style τ (split, empirical)
        tau_conf, df_rc = split_nat_calibration(
            p_adv_full[nat_calib_idx], y_nat_cal, P3_nat_cal,
            target_risk=CFG["target_risk_nat"], delta=CFG["conformal_delta"], grid=CFG["tau_grid_size"]
        )

        # Save τs (original)
        common_dir = ensure_dir(out_root / "_q1_common")
        save_json({"tau_legacy_quantile": float(tau_legacy),
                   "tau_conformal_nat_risk": float(tau_conf),
                   "settings": {"target_risk_nat": CFG["target_risk_nat"], "delta": CFG["conformal_delta"],
                                "nat_calib_ratio": CFG["nat_calib_ratio"]}},
                  common_dir / "tau_summary.json")

        # Also mirror τ summary & gate risk-coverage into detector tree
        common_dir_3cd = ensure_dir(out_root_3cd / "_q1_common")
        save_json({"tau_legacy_quantile": float(tau_legacy),
                   "tau_conformal_nat_risk": float(tau_conf),
                   "settings": {"target_risk_nat": CFG["target_risk_nat"], "delta": CFG["conformal_delta"],
                                "nat_calib_ratio": CFG["nat_calib_ratio"]}},
                  common_dir_3cd / "tau_summary.json")

        # Risk–coverage curve CSV for the gate (original + mirror)
        df_rc.to_csv(common_dir / "risk_coverage_gate.csv", index=False)
        df_rc.to_csv(common_dir_3cd / "risk_coverage_gate.csv", index=False)
        if CFG["use_plots"]:
            fig, ax = plt.subplots(figsize=(7,5))
            ax.plot(df_rc["coverage_nat"], df_rc["risk_nat"], marker="o", linewidth=1)
            ax.set_xlabel("Coverage (NAT accepted)"); ax.set_ylabel("Risk on accepted NAT")
            ax.set_title(f"Risk–Coverage (Gate) — {base_name}/{name3}")
            fig.tight_layout(); fig.savefig(common_dir / "risk_coverage_gate.png", dpi=200); plt.close(fig)

            # mirror image to detector tree
            fig, ax = plt.subplots(figsize=(7,5))
            ax.plot(df_rc["coverage_nat"], df_rc["risk_nat"], marker="o", linewidth=1)
            ax.set_xlabel("Coverage (NAT accepted)"); ax.set_ylabel("Risk on accepted NAT")
            ax.set_title(f"Risk–Coverage (Gate) — {base_name}/{name3}")
            fig.tight_layout(); fig.savefig(common_dir_3cd / "risk_coverage_gate.png", dpi=200); plt.close(fig)

        # Baseline gates: MSP & Energy (Energy only if logits available) — original tree only
        msp_nat = msp_scores_nat(P3_nat_eval)
        tau_msp = float(np.quantile(msp_nat, CFG["tau_quantile_3c"])) if msp_nat.size else 1.0

        energy_nat = energy_scores_from_logits(log_nat_eval)
        if energy_nat is not None and np.std(energy_nat) > 1e-6:
            # higher confidence → lower energy; accept if energy <= τ_energy
            tau_energy = float(np.quantile(energy_nat, 1.0 - (1.0 - CFG["tau_quantile_3c"])))
            have_energy = True
        else:
            tau_energy = None; have_energy = False

        save_json({"tau_msp": tau_msp, "tau_energy": tau_energy, "quantile": CFG["tau_quantile_3c"]}, common_dir / "tau_baselines.json")

        # Per attack kind (FGSM/PGD)
        for which in ["FGSM", "PGD"]:
            out_dir = ensure_dir(out_root / f"mixed_{which}")
            out_dir_3c  = ensure_dir(out_root_3c  / f"mixed_{which}")
            out_dir_3cd = ensure_dir(out_root_3cd / f"mixed_{which}")
            cache_dir = ensure_dir(out_root / "_cache")
            steps_used = int(CFG["steps_fgsm"] if which == "FGSM" else CFG["steps_pgd"])

            # 1) Craft ADV subset (cacheable)
            cache_path = adv_cache_path(out_root, base_name, which, adv_idx_fixed, steps_used, CFG["eps_z"], CFG["alpha_z"])
            use_cached = False
            if CFG["cache_adv"] and cache_path.exists():
                try:
                    data = np.load(cache_path)
                    if np.array_equal(data["adv_idx"], adv_idx_fixed):
                        Xz_adv = data["Xz_adv"].astype(np.float32, copy=False)
                        use_cached = True; print(f"[CACHE] Loaded ADV subset from {cache_path.name}")
                    del data
                except Exception:
                    use_cached = False
            if not use_cached:
                def craft(Xz_sub, y_sub):
                    if base_name == "DL-MLP":
                        return attack_tabular_torch_batched(
                            base_est, Xz_sub, y_sub, eps=CFG["eps_z"], alpha=CFG["alpha_z"], steps=steps_used,
                            device=device, z_min=z_min, z_max=z_max, batch_size=CFG["torch_attack_bs"],
                            desc=f"{base_name}/{which} torch attack", use_progress=CFG["use_progress"], use_amp=CFG["use_amp"]
                        )
                    # non-linear sklearn bases → distilled surrogate
                    class SurrogateMLP(nn.Module):
                        def __init__(self, D, H=CFG["surrogate_hidden"]):
                            super().__init__()
                            self.net = nn.Sequential(
                                nn.Linear(D, H), nn.ReLU(),
                                nn.Linear(H, H), nn.ReLU(),
                                nn.Linear(H, 1)
                            )
                        def forward(self, x): return self.net(x).squeeze(-1)
                    idx = np.random.RandomState(42).choice(Xz.shape[0], size=min(CFG["surrogate_sample_limit"], Xz.shape[0]), replace=False)
                    if hasattr(base_est, "predict_proba"):
                        soft = base_est.predict_proba(Xz[idx])[:,1].astype(np.float32)
                    elif hasattr(base_est, "decision_function"):
                        df = base_est.decision_function(Xz[idx]).astype(np.float32); soft = 1./(1.+np.exp(-df))
                    else:
                        soft = base_est.predict(Xz[idx]).astype(np.float32)
                    mdl = SurrogateMLP(D=Xz.shape[1]).to(device)
                    opt = torch.optim.Adam(mdl.parameters(), lr=3e-4, weight_decay=1e-5)
                    ds = torch.utils.data.TensorDataset(torch.from_numpy(Xz[idx]).float(), torch.from_numpy(soft))
                    dl = torch.utils.data.DataLoader(ds, batch_size=CFG["surrogate_batch"], shuffle=True)
                    mdl.train()
                    for _ in range(CFG["surrogate_epochs_finetune"]):
                        for xb, sb in dl:
                            xb=xb.to(device); sb=sb.to(device)
                            opt.zero_grad(set_to_none=True)
                            loss = F.binary_cross_entropy_with_logits(mdl(xb), sb)
                            loss.backward(); opt.step()
                    mdl.eval()
                    return attack_tabular_torch_batched(
                        mdl, Xz_sub, y_sub, eps=CFG["eps_z"], alpha=CFG["alpha_z"], steps=steps_used,
                        device=device, z_min=z_min, z_max=z_max, batch_size=CFG["bb_attack_bs"],
                        desc=f"{base_name}/{which} bb attack", use_progress=CFG["use_progress"], use_amp=CFG["use_amp"]
                    )

                Xz_adv = craft(Xz[adv_idx_fixed], y_base[adv_idx_fixed])
                if CFG["cache_adv"]:
                    ensure_dir(cache_dir); np.savez_compressed(cache_path, adv_idx=adv_idx_fixed, Xz_adv=Xz_adv)
                    print(f"[CACHE] Saved ADV subset to {cache_path.name}")

            # 2) Defense scores for acceptance masks (τ legacy & conformal)
            p_adv_nat = p_adv_full[nat_idx_fixed]
            p_adv_adv = defense_p_adv_from_z(
                defense, scaler_base, Xz_adv, batch=CFG["eval_bs"], calibrator=calibrator,
                desc=f"defense p_adv ADV/{base_name}/{which}", use_progress=CFG["use_progress"],
                fast_predict=CFG["FAST_PREDICT"], early_stop_margin=CFG["EARLY_STOP_MARGIN"]
            )

            def accept_mask(pvec, tau): return (pvec < float(tau))

            # 3) Build 3-class GT
            y3_nat = y_base[nat_idx_fixed].copy()
            y3_adv = np.full_like(y_base[adv_idx_fixed], 2)

            # 4) Prepare 3-class features (reorder+scale)
            Xz3_nat = Xz3[nat_idx_fixed]
            X_raw_adv = scaler_base.inverse_transform(Xz_adv).astype(np.float32, copy=False)
            X_raw_adv_3c = _align_X_for_3c(X_raw_adv, feat_base, feat3c)
            Xz3_adv = scaler3c.transform(X_raw_adv_3c).astype(np.float32, copy=False)

            # 5) 3-class probabilities/preds
            if kind3 == "torch":
                log_nat, P3_nat = _torch_logits_and_probs(est3, Xz3_nat, device, CFG["torch_eval_bs"], f"3c/{name3} NAT (no-def)", CFG["use_progress"], CFG["use_amp"])
                log_adv, P3_adv = _torch_logits_and_probs(est3, Xz3_adv, device, CFG["torch_eval_bs"], f"3c/{name3} ADV (no-def)", CFG["use_progress"], CFG["use_amp"])
            else:
                P3_nat = _chunked_probs_multi(est3, Xz3_nat, CFG["eval_bs"], f"3c/{name3} NAT (no-def)", CFG["use_progress"])
                P3_adv = _chunked_probs_multi(est3, Xz3_adv, CFG["eval_bs"], f"3c/{name3} ADV (no-def)", CFG["use_progress"])
                log_nat = _sk_multiclass_logits(est3, Xz3_nat)
                log_adv = _sk_multiclass_logits(est3, Xz3_adv)

            # ---------- (A) NO DEFENSE ----------
            y3_mixed = np.concatenate([y3_nat, y3_adv])
            P3_mixed = np.concatenate([P3_nat, P3_adv])
            yhat3_mixed = P3_mixed.argmax(axis=1)

            acc_nd = float(accuracy_score(y3_mixed, yhat3_mixed))
            f1_nd  = float(f1_score(y3_mixed, yhat3_mixed, average="macro"))
            try:
                auc_nd = float(roc_auc_score(y3_mixed, P3_mixed, multi_class="ovr"))
            except Exception:
                auc_nd = None
            per_class_nd = _per_class_metrics(y3_mixed, yhat3_mixed, LABELS3)
            cm_nd = cm_as_dict(y3_mixed, yhat3_mixed, LABELS3)

            # plots (original)
            if CFG["use_plots"]:
                plot_confusion_3c(y3_mixed, yhat3_mixed, out_path=out_dir / "confusion_no_def.png", title=f"3-class Confusion (no defense) [{name3}]")
                if auc_nd is not None:
                    plot_roc_multiclass(y3_mixed, P3_mixed, out_path=out_dir / "roc_ovr_no_def.png", title=f"ROC OvR (no defense) [{name3}]")
                # mirrored plots to 3-class clean tree
                plot_confusion_3c(y3_mixed, yhat3_mixed, out_path=out_dir_3c / "confusion_no_def.png", title=f"3-class Confusion (no defense) [{name3}]")
                if auc_nd is not None:
                    plot_roc_multiclass(y3_mixed, P3_mixed, out_path=out_dir_3c / "roc_ovr_no_def.png", title=f"ROC OvR (no defense) [{name3}]")

            # classification report (original + mirror)
            try:
                rep_txt = classification_report(y3_mixed, yhat3_mixed, digits=4)
                save_text(rep_txt, out_dir / "classification_report_no_def.txt")
                save_text(rep_txt, out_dir_3c / "classification_report_no_def.txt")
            except Exception:
                pass

            # NAT-only NO DEFENSE
            nat_mask_all = (y3_mixed != 2)
            y_nat_only = y3_mixed[nat_mask_all]
            yhat_nat_only = yhat3_mixed[nat_mask_all]
            acc_nat_nd = float(accuracy_score(y_nat_only, yhat_nat_only))
            f1_nat_nd = float(f1_score(y_nat_only, yhat_nat_only, labels=LABELS_NAT, average="macro", zero_division=0))
            per_class_nat_nd = _per_class_metrics(y_nat_only, yhat_nat_only, LABELS_NAT)
            cm_nat_nd = cm_as_dict(y_nat_only, yhat_nat_only, NAT_CM_LABELS)
            if CFG["use_plots"]:
                plot_confusion_nat(y_nat_only, yhat_nat_only, out_path=out_dir / "confusion_nat_no_def.png", title=f"NAT-only Confusion (no defense) [{name3}]")
                plot_confusion_nat(y_nat_only, yhat_nat_only, out_path=out_dir_3c / "confusion_nat_no_def.png", title=f"NAT-only Confusion (no defense) [{name3}]")

            # BEFORE (original + mirror)
            before_payload = {
                "scenario": f"mixed_{which}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attack_base_model": base_name,
                "threeclass_model": name3,
                "lenient_tau_3c": float(tau_legacy),
                "stream_counts": {"nat": int(nat_idx_fixed.size), "adv": int(adv_idx_fixed.size), "total": int(nat_idx_fixed.size + adv_idx_fixed.size)},
                "three_class": {
                    "accuracy": acc_nd, "f1_macro": f1_nd, "auc_ovr": auc_nd,
                    "per_class": per_class_nd, "counts": {"total": int(y3_mixed.size), "nat": int(y3_nat.size), "adv": int(y3_adv.size)},
                    "adv_pred2_rate": float((P3_adv.argmax(axis=1) == 2).mean()) if P3_adv.shape[0] else float("nan"),
                    "confusion": cm_nd,
                },
                "nat_only_slice": {
                    "accuracy": acc_nat_nd, "f1_macro": f1_nat_nd, "per_class": per_class_nat_nd,
                    "counts": {"nat_total": int(y_nat_only.size)}, "confusion": cm_nat_nd,
                },
                "attack": {"kind": which, "steps": steps_used, "eps_z": float(CFG["eps_z"]), "alpha_z": float(CFG["alpha_z"])},
            }
            save_json(before_payload, out_dir / "metrics_before_defense.json")
            save_json(before_payload, out_dir_3c / "metrics_before_defense.json")

            # ---------- (B) AFTER DEFENSE for multiple gates (original tree only) ----------
            gates = {
                "gate_legacy": lambda pnat, padv: (accept_mask(pnat, tau_legacy), accept_mask(padv, tau_legacy)),
                "gate_conformal": lambda pnat, padv: (accept_mask(pnat, tau_conf), accept_mask(padv, tau_conf)),
                "msp_quantile": None,   # fill below
                "energy_quantile": None # fill below if available
            }

            # Baseline gates using MSP/Energy
            msp_nat_all = msp_scores_nat(P3_nat)
            msp_adv_all = msp_scores_nat(P3_adv)
            gates["msp_quantile"] = lambda _pn, _pa: ((msp_nat_all >= tau_msp), (msp_adv_all >= tau_msp))

            if have_energy:
                energy_nat_all = energy_scores_from_logits(log_nat)
                energy_adv_all = energy_scores_from_logits(log_adv)
                gates["energy_quantile"] = lambda _pn, _pa: ((energy_nat_all <= tau_energy), (energy_adv_all <= tau_energy))
            else:
                gates.pop("energy_quantile", None)  # skip if not reliable

            # Evaluate each gate and save (original). If conformal, also mirror to 3cd.
            for gname, mask_fn in list(gates.items()):
                if mask_fn is None: continue
                acc_nat_mask, acc_adv_mask = mask_fn(p_adv_nat, p_adv_adv)
                Yacc_nat = y3_nat[acc_nat_mask]
                Pacc_nat = P3_nat[acc_nat_mask]
                Yacc_adv = y3_adv[acc_adv_mask]
                Pacc_adv = P3_adv[acc_adv_mask]

                if (Yacc_nat.size + Yacc_adv.size) > 0:
                    P3_acc = np.concatenate([Pacc_nat, Pacc_adv], axis=0)
                    y3_acc = np.concatenate([Yacc_nat, Yacc_adv], axis=0)
                    yhat3_acc = P3_acc.argmax(axis=1)

                    acc_ad = float(accuracy_score(y3_acc, yhat3_acc))
                    f1_ad  = float(f1_score(y3_acc, yhat3_acc, average="macro"))
                    try:
                        auc_ad = float(roc_auc_score(y3_acc, P3_acc, multi_class="ovr"))
                    except Exception:
                        auc_ad = None
                    per_class_ad = _per_class_metrics(y3_acc, yhat3_acc, LABELS3)
                    cm_ad = cm_as_dict(y3_acc, yhat3_acc, LABELS3)

                    # Accepted-set calibration on NAT portion
                    if Yacc_nat.size > 0:
                        yhat_nat_acc = Pacc_nat.argmax(axis=1)
                        p_nat_pos = Pacc_nat[np.arange(Pacc_nat.shape[0]), yhat_nat_acc]
                        ece_nat = ece_score((yhat_nat_acc == Yacc_nat).astype(int), p_nat_pos, n_bins=15)
                        brier_nat = float(np.mean((p_nat_pos - (yhat_nat_acc == Yacc_nat).astype(float))**2))
                    else:
                        ece_nat, brier_nat = float("nan"), float("nan")

                    adv_acc_rate = float(acc_adv_mask.mean()) if p_adv_adv.size else float("nan")
                    nat_block_rate = float((~acc_nat_mask).mean()) if p_adv_nat.size else float("nan")

                    after_payload = {
                        "scenario": f"mixed_{which}",
                        "gate": gname,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "threeclass_model": name3,
                        "stream_counts": {"nat": int(len(y3_nat)), "adv": int(len(y3_adv))},
                        "gate_stats": {
                            "accept_rate_nat": float(acc_nat_mask.mean()) if p_adv_nat.size else float("nan"),
                            "accept_rate_adv": adv_acc_rate,
                            "nat_block_rate": nat_block_rate
                        },
                        "three_class": {
                            "accuracy": acc_ad, "f1_macro": f1_ad, "auc_ovr": auc_ad,
                            "per_class": per_class_ad,
                            "counts": {"total_accepted": int((Yacc_nat.size + Yacc_adv.size)),
                                       "nat_accepted": int(Yacc_nat.size), "adv_accepted": int(Yacc_adv.size)},
                            "accepted_nat_calibration": {"ece": ece_nat, "brier": brier_nat},
                            "adv_pred2_rate_accepted": float((Pacc_adv.argmax(axis=1) == 2).mean()) if Pacc_adv.shape[0] else float("nan")
                        }
                    }
                    save_json(after_payload, out_dir / f"metrics_after_defense__{gname}.json")

                    # Risk–coverage CSV for gate_* (not MSP/Energy)
                    if gname.startswith("gate_"):
                        qs = np.linspace(0.80, 0.9999, CFG["tau_grid_size"])
                        taus = np.quantile(p_adv_nat, qs)
                        rows=[]
                        for tau in taus:
                            m_nat = (p_adv_nat < tau); m_adv=(p_adv_adv < tau)
                            Yn = y3_nat[m_nat]; Pn=P3_nat[m_nat]
                            Ya = y3_adv[m_adv]; Pa=P3_adv[m_adv]
                            if (Yn.size + Ya.size)==0:
                                rows.append({"tau": float(tau), "coverage": 0.0, "risk": 1.0}); continue
                            Pacc = np.concatenate([Pn,Pa]); Yacc=np.concatenate([Yn,Ya])
                            yhat = Pacc.argmax(axis=1)
                            risk = 1.0 - float(accuracy_score(Yacc, yhat))
                            cov = float((Yn.size + Ya.size)/ (len(y3_nat)+len(y3_adv)))
                            rows.append({"tau": float(tau), "coverage": cov, "risk": risk})
                        rc_df = pd.DataFrame(rows)
                        rc_df.to_csv(out_dir / f"risk_coverage__{gname}.csv", index=False)

                        # If conformal, mirror to 3cd tree
                        if gname == "gate_conformal":
                            save_json(after_payload, out_dir_3cd / f"metrics_after_defense__{gname}.json")
                            rc_df.to_csv(out_dir_3cd / f"risk_coverage__{gname}.csv", index=False)

                else:
                    # No accepted items: still log the gate stats
                    na_payload = {
                        "scenario": f"mixed_{which}",
                        "gate": gname,
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "threeclass_model": name3,
                        "stream_counts": {"nat": int(len(y3_nat)), "adv": int(len(y3_adv))},
                        "three_class": {"note": "no samples accepted at this threshold"}
                    }
                    save_json(na_payload, out_dir / f"metrics_after_defense__{gname}.json")
                    if gname == "gate_conformal":
                        save_json(na_payload, out_dir_3cd / f"metrics_after_defense__{gname}.json")

            # ---------- Preserve original AFTER-DEFENSE (legacy τ) outputs exactly ----------
            acc_nat_mask_3c = (p_adv_nat < tau_legacy)
            acc_adv_mask_3c = (p_adv_adv < tau_legacy)
            Pacc_nat = P3_nat[acc_nat_mask_3c]; Yacc_nat = y3_nat[acc_nat_mask_3c]
            Pacc_adv = P3_adv[acc_adv_mask_3c]; Yacc_adv = y3_adv[acc_adv_mask_3c]
            if (Yacc_nat.size + Yacc_adv.size) > 0:
                P3_acc = np.concatenate([Pacc_nat, Pacc_adv], axis=0)
                y3_acc = np.concatenate([Yacc_nat, Yacc_adv], axis=0)
                yhat3_acc = P3_acc.argmax(axis=1)
                acc_ad = float(accuracy_score(y3_acc, yhat3_acc))
                f1_ad  = float(f1_score(y3_acc, yhat3_acc, average="macro"))
                try:
                    auc_ad = float(roc_auc_score(y3_acc, P3_acc, multi_class="ovr"))
                except Exception:
                    auc_ad = None
                per_class_ad = _per_class_metrics(y3_acc, yhat3_acc, LABELS3)
                cm_ad = cm_as_dict(y3_acc, yhat3_acc, LABELS3)
                if CFG["use_plots"]:
                    plot_confusion_3c(y3_acc, yhat3_acc, out_path=out_dir / "confusion_after_def.png", title=f"3-class Confusion (after defense τ3c) [{name3}]")
                    if auc_ad is not None:
                        plot_roc_multiclass(y3_acc, P3_acc, out_path=out_dir / "roc_ovr_after_def.png", title=f"ROC OvR (after defense τ3c) [{name3}]")
                try:
                    save_text(classification_report(y3_acc, yhat3_acc, digits=4), out_dir / "classification_report_after_def.txt")
                except Exception:
                    pass
            else:
                acc_ad = float("nan"); f1_ad = float("nan"); auc_ad = None; per_class_ad = {str(k): {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "support": 0} for k in LABELS3}
                cm_ad = {"labels": LABELS3, "matrix": [[0,0,0],[0,0,0],[0,0,0]], "total": 0, "accuracy_from_confusion": float("nan")}

            # NAT-only AFTER DEFENSE (accepted NAT only)
            if Yacc_nat.size > 0:
                yhat_nat_acc = Pacc_nat.argmax(axis=1)
                acc_nat_ad = float(accuracy_score(Yacc_nat, yhat_nat_acc))
                f1_nat_ad = float(f1_score(Yacc_nat, yhat_nat_acc, labels=LABELS_NAT, average="macro", zero_division=0))
                per_class_nat_ad = _per_class_metrics(Yacc_nat, yhat_nat_acc, LABELS_NAT)
                cm_nat_ad = cm_as_dict(Yacc_nat, yhat_nat_acc, NAT_CM_LABELS)
                if CFG["use_plots"]:
                    plot_confusion_nat(Yacc_nat, yhat_nat_acc, out_path=out_dir / "confusion_nat_after_def.png", title=f"NAT-only Confusion (after defense τ3c) [{name3}]")
            else:
                acc_nat_ad = float("nan"); f1_nat_ad = float("nan")
                per_class_nat_ad = {str(k): {"precision": float("nan"), "recall": float("nan"), "f1": float("nan"), "support": 0} for k in LABELS_NAT}
                cm_nat_ad = {"labels": NAT_CM_LABELS, "matrix": [[0,0,0],[0,0,0],[0,0,0]], "total": 0, "accuracy_from_confusion": float("nan")}

            det_rate_nd   = float((P3_adv.argmax(axis=1) == 2).mean()) if P3_adv.shape[0] else float("nan")
            det_rate_acc  = float((Pacc_adv.argmax(axis=1) == 2).mean()) if Pacc_adv.shape[0] else float("nan")

            # BEFORE already saved above

            # AFTER (legacy τ) — preserved filename/shape (original tree only)
            save_json({
                "scenario": f"mixed_{which}",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "attack_base_model": base_name,
                "threeclass_model": name3,
                "lenient_tau_3c": float(tau_legacy),
                # [FIX] correct total = nat + adv (was nat + nat)
                "stream_counts": {"nat": int(nat_idx_fixed.size), "adv": int(adv_idx_fixed.size), "total": int(nat_idx_fixed.size + adv_idx_fixed.size)},
                "gate": {
                    "accept_rate_nat": float(acc_nat_mask_3c.mean()) if p_adv_nat.size else float("nan"),
                    "accept_rate_adv": float(acc_adv_mask_3c.mean()) if p_adv_adv.size else float("nan"),
                    "counts": {"nat_accepted": int(acc_nat_mask_3c.sum()), "nat_blocked": int((~acc_nat_mask_3c).sum()),
                               "adv_accepted": int(acc_adv_mask_3c.sum()), "adv_blocked": int((~acc_adv_mask_3c).sum())},
                },
                "three_class": {
                    "accuracy": acc_ad, "f1_macro": f1_ad, "auc_ovr": auc_ad,
                    "per_class": per_class_ad,
                    "counts": {"total_accepted": int((Yacc_nat.size + Yacc_adv.size)), "nat_accepted": int(Yacc_nat.size), "adv_accepted": int(Yacc_adv.size)},
                    "adv_pred2_rate_accepted": det_rate_acc,
                    "confusion": cm_ad,
                },
                "nat_only_slice": {
                    "accuracy": acc_nat_ad, "f1_macro": f1_nat_ad,
                    "per_class": per_class_nat_ad, "counts": {"nat_accepted": int(Yacc_nat.size)},
                    "confusion": cm_nat_ad,
                },
                "attack": {"kind": which, "steps": steps_used, "eps_z": float(CFG["eps_z"]), "alpha_z": float(CFG["alpha_z"])},
            }, out_dir / "metrics_after_defense.json")

            # ---------- NEW: End-to-end adaptive attack on composition ----------
            if CFG["adaptive_attack"] and kind3 == "torch":
                print(f"[ADAPT] Composed attack ({which}) against gate+3c with τ=conformal")
                try:
                    Xz_adv_adapt = composed_attack_dl(
                        Xz_in=Xz_adv, scaler_base=scaler_base, model3_torch=est3, gate_surrogate=gate_sur,
                        y_true3=np.full_like(y3_adv, 2), tau=tau_conf, eps=CFG["eps_z"],
                        alpha=CFG["adaptive_alpha"], steps=CFG["adaptive_steps"],
                        device=device, z_min=z_min, z_max=z_max,
                        lambda_gate=CFG["adaptive_lambda_gate"], margin=CFG["adaptive_margin"], use_amp=CFG["use_amp"]
                    )
                except Exception as e:
                    print(f"[WARN] Gradient composed attack failed ({e}); falling back to SPSA.")
                    def loss_fn_spsa(Xz_try):
                        # lower gate prob AND push 3c away from 2
                        X_raw_try = scaler_base.inverse_transform(Xz_try)
                        p_gate = _predict_with_defense(defense, X_raw_try)
                        m_gate = (p_gate - tau_conf)  # want negative
                        if kind3 == "torch":
                            _, P = _torch_logits_and_probs(est3, scaler3c.transform(_align_X_for_3c(X_raw_try, feat_base, feat3c)), device, CFG["torch_eval_bs"], "spsa/3c", False, False)
                        else:
                            P = _chunked_probs_multi(est3, scaler3c.transform(_align_X_for_3c(X_raw_try, feat_base, feat3c)), CFG["eval_bs"], "spsa/3c", False)
                        bad = 1.0 - (P[:,0:2].max(axis=1))  # want low (prefer NAT)
                        return m_gate + bad
                    Xz_adv_adapt = spsa_attack(
                        Xz_adv, loss_fn=lambda Xz_try: loss_fn_spsa(Xz_try),
                        steps=CFG["spsa_iters"], lr=CFG["spsa_lr"], delta=CFG["spsa_delta"], eps=CFG["eps_z"],
                        z_min=z_min, z_max=z_max, rng=np.random.RandomState(7)
                    )

                # Evaluate after gate with τ_conf
                p_adv_adapt = defense_p_adv_from_z(
                    defense, scaler_base, Xz_adv_adapt, batch=CFG["eval_bs"], calibrator=calibrator,
                    desc=f"defense p_adv ADAPT/{base_name}/{which}", use_progress=CFG["use_progress"],
                    fast_predict=CFG["FAST_PREDICT"], early_stop_margin=CFG["EARLY_STOP_MARGIN"]
                )
                acc_adv_mask_adapt = (p_adv_adapt < tau_conf)
                X_raw_adapt = scaler_base.inverse_transform(Xz_adv_adapt)
                Xz3_adapt = scaler3c.transform(_align_X_for_3c(X_raw_adapt, feat_base, feat3c)).astype(np.float32)
                if kind3 == "torch":
                    _, P3_adapt = _torch_logits_and_probs(est3, Xz3_adapt, device, CFG["torch_eval_bs"], f"3c/{name3} ADAPT", False, CFG["use_amp"])
                else:
                    P3_adapt = _chunked_probs_multi(est3, Xz3_adapt, CFG["eval_bs"], f"3c/{name3} ADAPT", False)
                if acc_adv_mask_adapt.any():
                    Pacc_adapt = P3_adapt[acc_adv_mask_adapt]
                    yhat_adapt = Pacc_adapt.argmax(axis=1)
                    slip_rate = float(np.isin(yhat_adapt, [0,1]).mean())
                else:
                    slip_rate = float("nan")
                adapt_payload = {
                    "scenario": f"mixed_{which}",
                    "gate": "gate_conformal",
                    "tau_used": float(tau_conf),
                    "adaptive": {
                        "steps": CFG["adaptive_steps"], "alpha": CFG["adaptive_alpha"],
                        "lambda_gate": CFG["adaptive_lambda_gate"], "margin": CFG["adaptive_margin"]
                    },
                    "adv_accept_rate_adaptive": float(acc_adv_mask_adapt.mean()),
                    "adv_slip_as_nat_rate_adaptive": slip_rate
                }
                save_json(adapt_payload, out_dir / "metrics_adaptive_attack.json")
                save_json(adapt_payload, out_dir_3cd / "metrics_adaptive_attack.json")

            # free per-attack memory
            del Xz_adv, p_adv_nat, p_adv_adv, P3_nat, P3_adv, P3_mixed, y3_mixed
            cleanup_cuda()

    print("\n[DONE] 3-class evaluation with Q1 extensions complete.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[FATAL]", e)
        print(traceback.format_exc())
        raise
