"""
Adversarial evaluation (DL-MLP, Random Forest, LightGBM) with rich diagnostics — array-free JSON.

Per model & attack (NAT/FGSM/PGD):
  results_evaluation/<ModelName>/adv_<ATTACK>/
    - metrics.json        (only scalars + confusion counts; NO big arrays)
    - classification_report.txt          # kept (contains per-class lines not in JSON)
    - confusion_matrix.png
    - roc_curve.png (if scores available)
    - pr_curve.png  (if scores available)
    - attack_trace.json   (FGSM/PGD: small sample; PGD logs early step snapshots)

Global:
  results_evaluation/_metrics_all.json   (records of all model×attack; scalars only)
  results_evaluation/_summary_global_metrics.json (averages over models per attack; scalars only)
  results_evaluation/_combined/combined_<ATTACK>.png  # single combined figure per attack
"""

from __future__ import annotations
from pathlib import Path
from datetime import datetime, timezone
import os, gc, json, warnings, math
import numpy as np
import pandas as pd
import joblib

# cut fragmentation *before* CUDA alloc
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Matplotlib headless
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score, confusion_matrix,
    log_loss, roc_curve, precision_recall_curve, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
)

# Optional libs
try:
    import lightgbm as lgb  # noqa: F401
except Exception:
    lgb = None

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

if torch.cuda.is_available():
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# -------------------------------------------------------------------
# Config
# -------------------------------------------------------------------
CFG = {
    "models_dir": "models",
    "results_dir": "results_evaluation",
    "dataset_csv": "features_extracted.csv",
    "label_col": "label",
    "url_col": "url",                 # unused for tabular attacks; kept for parity
    "train_ratio": 0.70,
    "val_ratio":   0.10,
    "test_ratio":  0.20,
    "random_state": 42,
    "max_attack_samples_tab": 50000,  # limit for runtime sanity (applies to test slice)
    "use_gpu": True,
    "num_workers": 2,
}

ATTACKCFG = {
    "eps_tab_fgsm": 0.5,
    "eps_tab_pgd":  0.5,
    "alpha_tab_pgd": 0.05,
    "steps_tab_pgd": 10,
}

# -------------------------------------------------------------------
# Utility
# -------------------------------------------------------------------
def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True); return p

def save_json(obj, p: Path):
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")

def save_text(txt, p: Path):
    p.write_text(txt, encoding="utf-8")

def cleanup_cuda():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def set_seeds(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# -------------------------------------------------------------------
# Plot helpers (saved as PNGs)
# -------------------------------------------------------------------
def plot_confusion(y_true, y_pred, out_path: Path, title: str):
    try:
        fig, ax = plt.subplots(figsize=(5.5, 5))
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred, labels=[0,1], ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass

def plot_roc(y_true, y_score, out_path: Path, title: str):
    try:
        if len(np.unique(y_true)) < 2: return
        if y_score is None: return
        s = np.asarray(y_score)
        if np.unique(s).size <= 2:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        RocCurveDisplay.from_predictions(y_true, s, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass

def plot_pr(y_true, y_score, out_path: Path, title: str):
    try:
        if len(np.unique(y_true)) < 2: return
        if y_score is None: return
        s = np.asarray(y_score)
        if np.unique(s).size <= 2:
            return
        fig, ax = plt.subplots(figsize=(6, 5))
        PrecisionRecallDisplay.from_predictions(y_true, s, ax=ax)
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(out_path, dpi=200)
        plt.close(fig)
    except Exception:
        pass

# -------------------------------------------------------------------
# Combined single-figure renderer (ROC + PR + Metrics bars)
# -------------------------------------------------------------------
def render_combined_figure(attack_tag: str, entries: list[dict], out_dir: Path):
    """
    entries: list of dicts with keys:
        - name: model name
        - y_true: np.array
        - y_score: np.array or None (skip in curves if None/label-like)
        - metrics: metrics dict (scalars)
    Produces ONE PNG: ROC (top-left), PR (top-right), grouped bars bottom.
    """
    if not entries:
        return
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.2], hspace=0.28, wspace=0.22)

    # ROC
    ax_roc = fig.add_subplot(gs[0, 0])
    any_roc = False
    for e in entries:
        y_true = np.asarray(e["y_true"])
        y_score = e["y_score"]
        if y_score is None: continue
        s = np.asarray(y_score)
        if len(np.unique(y_true)) < 2 or np.unique(s).size <= 2:  # skip label-like
            continue
        fpr, tpr, _ = roc_curve(y_true, s)
        try:
            auc = roc_auc_score(y_true, s)
            label = f"{e['name']} (AUC={auc:.3f})"
        except Exception:
            label = e["name"]
        ax_roc.plot(fpr, tpr, label=label)
        any_roc = True
    ax_roc.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax_roc.set_xlabel("False Positive Rate"); ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title(f"ROC — {attack_tag}")
    if any_roc: ax_roc.legend()

    # PR
    ax_pr = fig.add_subplot(gs[0, 1])
    any_pr = False
    for e in entries:
        y_true = np.asarray(e["y_true"])
        y_score = e["y_score"]
        if y_score is None: continue
        s = np.asarray(y_score)
        if len(np.unique(y_true)) < 2 or np.unique(s).size <= 2:
            continue
        p, r, _ = precision_recall_curve(y_true, s)
        try:
            ap = average_precision_score(y_true, s)
            label = f"{e['name']} (AP={ap:.3f})"
        except Exception:
            label = e["name"]
        ax_pr.plot(r, p, label=label)
        any_pr = True
    ax_pr.set_xlabel("Recall"); ax_pr.set_ylabel("Precision")
    ax_pr.set_title(f"Precision–Recall — {attack_tag}")
    if any_pr: ax_pr.legend()

    # Grouped bars of scalar metrics
    key_metrics = ["accuracy", "f1", "roc_auc", "pr_auc", "balanced_accuracy", "mcc"]
    ax_bar = fig.add_subplot(gs[1, :])
    models = [e["name"] for e in entries]
    x = np.arange(len(key_metrics), dtype=float)
    width = max(0.8 / max(1, len(models)), 0.06)

    for i, e in enumerate(entries):
        vals = []
        for k in key_metrics:
            v = e["metrics"].get(k)
            if v is None or not isinstance(v, (int, float)) or not np.isfinite(v):
                v = 0.0
            vals.append(v)
        ax_bar.bar(x + i * width, vals, width, label=e["name"])

    ax_bar.set_xticks(x + (len(models) - 1) * width / 2)
    ax_bar.set_xticklabels(key_metrics, rotation=20)
    ax_bar.set_ylim(0, 1.05)
    ax_bar.set_ylabel("Score (0–1)")
    ax_bar.set_title(f"Key Metrics — {attack_tag}")
    ax_bar.legend(ncols=2)

    fig.suptitle(f"Combined Summary — {attack_tag}", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    ensure_dir(out_dir)
    fig.savefig(out_dir / f"combined_{attack_tag}.png", dpi=220)
    plt.close(fig)

# -------------------------------------------------------------------
# Calibration / rich metrics helpers
# -------------------------------------------------------------------
def expected_calibration_error(y_true, y_prob, n_bins=15):
    y_true = np.asarray(y_true).astype(int)
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

def safe_div(n, d):
    return float(n / d) if d else 0.0

def rich_metrics(y_true, y_score, y_pred, proba=None):
    """
    Detailed, scalar-only metrics (no big arrays).
    """
    out = {}
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    y_score = np.asarray(y_score) if y_score is not None else y_pred

    out["samples"] = int(len(y_true))
    out["accuracy"] = float(accuracy_score(y_true, y_pred))
    out["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    out["recall"] = float(recall_score(y_true, y_pred))  # = TPR / sensitivity
    out["f1"] = float(f1_score(y_true, y_pred))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, y_pred))
    out["mcc"] = float(matthews_corrcoef(y_true, y_pred))
    out["kappa"] = float(cohen_kappa_score(y_true, y_pred))

    # Confusion-derived
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    tpr = safe_div(tp, tp + fn)  # recall
    tnr = safe_div(tn, tn + fp)  # specificity
    fpr = safe_div(fp, fp + tn)
    fnr = safe_div(fn, fn + tp)
    ppv = safe_div(tp, tp + fp)  # precision
    npv = safe_div(tn, tn + fn)
    fdr = 1.0 - ppv
    _for = safe_div(fn, fn + tn)  # false omission rate
    prevalence = safe_div(tp + fn, tp + fn + tn + fp)
    pred_pos_rate = safe_div(tp + fp, tp + fp + tn + fn)
    youden_j = tpr + tnr - 1.0
    gmean = math.sqrt(max(tpr,0.0) * max(tnr,0.0))
    threat = safe_div(tp, tp + fn + fp)  # TS / CSI
    f0_5 = float(fbeta_score(y_true, y_pred, beta=0.5, zero_division=0))
    f2 = float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0))
    lr_plus = safe_div(tpr, fpr) if fpr > 0 else (float("inf") if tpr > 0 else 0.0)
    lr_minus = safe_div(fnr, tnr) if tnr > 0 else (float("inf") if fnr > 0 else 0.0)

    out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    out["specificity"] = tnr
    out["tpr"] = tpr
    out["tnr"] = tnr
    out["fpr"] = fpr
    out["fnr"] = fnr
    out["npv"] = npv
    out["fdr"] = fdr
    out["for"] = _for
    out["prevalence"] = prevalence
    out["pred_pos_rate"] = pred_pos_rate
    out["youden_j"] = youden_j
    out["gmean"] = gmean
    out["threat_score"] = threat
    out["f0_5"] = f0_5
    out["f2"] = f2
    out["lr_plus"] = lr_plus if np.isfinite(lr_plus) else None
    out["lr_minus"] = lr_minus if np.isfinite(lr_minus) else None

    # AUCs (no arrays saved)
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        out["roc_auc"] = None
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        out["pr_auc"] = None

    # Calibration metrics (kept as scalars if probs provided; no plot)
    if proba is not None:
        proba = np.asarray(proba).clip(1e-7, 1-1e-7)
        out["brier"] = float(np.mean((proba - y_true)**2))
        try:
            out["log_loss_ce"] = float(log_loss(y_true, proba, labels=[0,1]))
        except Exception:
            out["log_loss_ce"] = None
        out["ece"] = float(expected_calibration_error(y_true, proba, n_bins=15))
    else:
        out["brier"] = None
        out["log_loss_ce"] = None
        out["ece"] = None

    return out

# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def map_label(v):
    if isinstance(v, (int, np.integer)): return int(v)
    return 1 if str(v).strip().lower() in {"1","true","malicious","phishing","malware","bad"} else 0

def load_split_indices(global_dir: Path, y_all: np.ndarray, cfg: dict):
    split_path = global_dir / "split_indices.json"
    if split_path.exists():
        info = json.loads(split_path.read_text(encoding="utf-8"))
        return np.array(info["train_idx"]), np.array(info["val_idx"]), np.array(info["test_idx"])
    idx_all = np.arange(len(y_all))
    idx_tr, idx_tmp, y_tr, y_tmp = train_test_split(
        idx_all, y_all, test_size=(1.0 - cfg["train_ratio"]),
        random_state=cfg["random_state"], stratify=y_all
    )
    idx_val, idx_te, _, _ = train_test_split(
        idx_tmp, y_tmp,
        test_size=(cfg["test_ratio"] / (cfg["val_ratio"] + cfg["test_ratio"])),
        random_state=cfg["random_state"], stratify=y_tmp
    )
    save_json({"train_idx": idx_tr.tolist(), "val_idx": idx_val.tolist(), "test_idx": idx_te.tolist()}, split_path)
    return idx_tr, idx_val, idx_te

def load_tabular_scaled_from_df(df: pd.DataFrame, cfg: dict, global_dir: Path):
    """
    Load scaler + schema from training time. Be robust to missing columns / NaNs by filling
    with training means (so scaled z ~ 0 for missing).
    """
    assert (global_dir / "scaler.joblib").exists(), "Missing models/_global/scaler.joblib"
    assert (global_dir / "feature_columns.json").exists(), "Missing models/_global/feature_columns.json"

    y = df[cfg["label_col"]].apply(map_label).astype(np.int64).to_numpy()
    scaler: StandardScaler = joblib.load(global_dir / "scaler.joblib")
    feat_cols = json.loads((global_dir / "feature_columns.json").read_text())["feature_columns"]

    # Build in saved order with fill at training mean
    means = getattr(scaler, "mean_", np.zeros(len(feat_cols), dtype=np.float64))
    X_df = pd.DataFrame(index=df.index)
    for i, col in enumerate(feat_cols):
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
            s = s.fillna(float(means[i]))
        else:
            s = pd.Series(np.full(len(df), float(means[i]), dtype=np.float32), index=df.index)
        X_df[col] = s.astype(np.float32)

    X = X_df[feat_cols].to_numpy(dtype=np.float32)
    Xz = scaler.transform(X).astype(np.float32)
    return Xz, y

# -------------------------------------------------------------------
# Models
# -------------------------------------------------------------------
class MLPNet(nn.Module):
    def __init__(self, in_dim: int, hidden=(256,128,64), p_drop=0.2):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.BatchNorm1d(h), nn.Dropout(p_drop)]
            prev=h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev,1)
    def forward(self,x): return self.head(self.backbone(x)).squeeze(-1)

class SurrogateMLP(nn.Module):
    def __init__(self, in_dim, hidden=(256,128,64), p_drop=0.1):
        super().__init__()
        layers=[]; prev=in_dim
        for h in hidden:
            layers += [nn.Linear(prev,h), nn.ReLU(), nn.Dropout(p_drop)]
            prev=h
        self.f = nn.Sequential(*layers, nn.Linear(prev,1))
    def forward(self,x): return self.f(x).squeeze(-1)

def distill_surrogate(teacher, X, y, device, tag, epochs=6, lr=3e-4, wd=1e-5):
    # Soft labels from teacher
    if hasattr(teacher, "predict_proba"):
        soft = teacher.predict_proba(X)[:,1].astype(np.float32)
    else:
        soft = teacher.predict(X).astype(np.float32)

    model = SurrogateMLP(in_dim=X.shape[1]).to(device)
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(soft).float())
    dl = DataLoader(ds, batch_size=8192, shuffle=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    lossbce = nn.BCEWithLogitsLoss()

    model.train()
    for _ in range(epochs):
        for xb, sb in dl:
            xb, sb = xb.to(device), sb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = lossbce(logits, sb)
            loss.backward(); opt.step()
    model.eval()
    return model

# -------------------------------------------------------------------
# Attacks
# -------------------------------------------------------------------
def attack_fgsm(model, X, y, eps, device):
    model.eval()
    X_t = torch.from_numpy(X).to(device).float()
    y_t = torch.from_numpy(y).to(device).float()
    X_adv = X_t.clone().detach().requires_grad_(True)
    logits = model(X_adv)
    loss = nn.BCEWithLogitsLoss()(logits, y_t)
    loss.backward()
    grad_sign = X_adv.grad.sign()
    adv = X_adv + eps * grad_sign
    return adv.detach().cpu().numpy()

def attack_pgd(model, X, y, eps, alpha, steps, device):
    """
    Returns:
        adv_np: np.ndarray
        steps_info: dict[int -> list[dict]], keys are absolute indices (0..N-1),
                    each list holds early step snapshots {step, pred, prob, vector}.
    """
    model.eval()
    X0 = torch.from_numpy(X).to(device).float()
    Y = torch.from_numpy(y).to(device).float()
    adv = X0.clone().detach()
    steps_info = {}

    tracked_idx = list(range(min(5, len(X0))))  # log first few examples

    for step in range(steps):
        adv.requires_grad_(True)
        logits = model(adv)
        loss = nn.BCEWithLogitsLoss()(logits, Y)
        loss.backward()
        adv = adv + alpha * adv.grad.sign()
        adv = torch.max(torch.min(adv, X0+eps), X0-eps).detach()

        if step < 3:  # keep logs small
            sig = torch.sigmoid(logits).detach()
            for i in tracked_idx:
                rec = {
                    "step": int(step),
                    "pred": int((sig[i] >= 0.5).item()),
                    "prob": float(sig[i].item()),
                    "vector": adv[i].cpu().numpy().tolist(),
                }
                steps_info.setdefault(i, []).append(rec)

    return adv.cpu().numpy(), steps_info

# -------------------------------------------------------------------
# Metric writer for attacks (saves plots too)
# -------------------------------------------------------------------
def write_metrics_and_plots(name, attack_tag, y_true, y_pred, y_score, proba, out_dir: Path):
    metrics = rich_metrics(y_true, y_score, y_pred, proba)
    metrics.update({
        "model": name,
        "attack": attack_tag,
        "samples_evaluated": int(len(y_true)),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    save_json(metrics, out_dir / "metrics.json")

    # Text report (kept; contains per-class stats not fully in JSON)
    report = classification_report(y_true, y_pred, labels=[0,1], digits=4, zero_division=0)
    save_text(report, out_dir / "classification_report.txt")

    # Plots (no calibration)
    plot_confusion(y_true, y_pred, out_dir / "confusion_matrix.png", f"Confusion Matrix: {name} [{attack_tag}]")
    plot_roc(y_true, y_score, out_dir / "roc_curve.png", f"ROC: {name} [{attack_tag}]")
    plot_pr(y_true, y_score, out_dir / "pr_curve.png", f"Precision–Recall: {name} [{attack_tag}]")

    return metrics

def save_attack_trace(name, attack, X_nat, X_adv, y_true, y_pred_nat, y_pred_adv, out_dir: Path, steps_info=None):
    rng = np.random.default_rng(CFG["random_state"])
    idxs = rng.choice(len(y_true), size=min(10, len(y_true)), replace=False)

    trace = []
    for i in idxs:
        rec = {
            "sample_id": int(i),
            "true_label": int(y_true[i]),
            "nat_pred": int(y_pred_nat[i]),
            "adv_pred": int(y_pred_adv[i]),
            "nat_vector": X_nat[i].tolist(),
            "adv_vector": X_adv[i].tolist(),
        }
        if steps_info is not None and i in steps_info:
            rec["pgd_steps"] = steps_info[i]
        trace.append(rec)

    save_json(trace, out_dir / "attack_trace.json")

# -------------------------------------------------------------------
# Helpers for global CSV flattening (removed CSV export entirely)
# -------------------------------------------------------------------
def _flatten_for_csv(rec: dict) -> dict:
    flat = {}
    for k, v in rec.items():
        if isinstance(v, dict) and k == "confusion":
            flat.update({f"confusion_{ik}": iv for ik, iv in v.items()})
        else:
            flat[k] = v
    return flat

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
def main():
    set_seeds(CFG["random_state"])
    device = torch.device("cuda" if (CFG["use_gpu"] and torch.cuda.is_available()) else "cpu")
    models_root = Path(CFG["models_dir"])
    results_root = Path(CFG["results_dir"]); ensure_dir(results_root)
    global_dir = models_root / "_global"

    # Load dataset & scaled features
    df = pd.read_csv(CFG["dataset_csv"])
    assert CFG["label_col"] in df.columns, f"Missing label column: {CFG['label_col']}"
    df[CFG["label_col"]] = df[CFG["label_col"]].apply(map_label)
    Xz_all, y_all = load_tabular_scaled_from_df(df, CFG, global_dir)

    # Reuse training split
    idx_tr, idx_val, idx_te = load_split_indices(global_dir, y_all, CFG)
    X_tr, y_tr = Xz_all[idx_tr], y_all[idx_tr]
    X_te, y_te = Xz_all[idx_te], y_all[idx_te]

    # Optional cap on test size (runtime control)
    if CFG["max_attack_samples_tab"] and len(X_te) > CFG["max_attack_samples_tab"]:
        rng = np.random.default_rng(CFG["random_state"])
        sel = rng.choice(np.arange(len(X_te)), size=CFG["max_attack_samples_tab"], replace=False)
        X_te = X_te[sel]; y_te = y_te[sel]

    model_list = ["DL-MLP", "Random_Forest", "LightGBM"]
    all_records = []          # one row per (model, attack) — scalars only
    summary_accumulator = {}  # attack -> list of metrics dicts
    combined_entries = {"NAT": [], "FGSM": [], "PGD": []}  # for single combined figs

    for name in model_list:
        mdir = models_root / name
        if not mdir.exists():
            print(f"[WARN] Skipping {name}: {mdir} not found")
            continue

        print(f"\n==== Evaluating & Attacking: {name} ====")
        out_base = ensure_dir(results_root / name)

        if name == "DL-MLP":
            mdl = MLPNet(in_dim=X_te.shape[1]).to(device)
            mdl.load_state_dict(torch.load(mdir / "model.pt", map_location=device))
            mdl.eval()

            # NAT
            with torch.no_grad():
                p_nat = torch.sigmoid(
                    mdl(torch.from_numpy(X_te).to(device).float())
                ).cpu().numpy().ravel()
            y_pred_nat = (p_nat >= 0.5).astype(int)
            metrics_nat = write_metrics_and_plots(name, "NAT", y_te, y_pred_nat, p_nat, p_nat, ensure_dir(out_base / "adv_NAT"))
            combined_entries["NAT"].append({"name": name, "y_true": y_te, "y_score": p_nat, "metrics": metrics_nat})

            # FGSM
            X_adv = attack_fgsm(mdl, X_te, y_te, ATTACKCFG["eps_tab_fgsm"], device)
            with torch.no_grad():
                p_adv = torch.sigmoid(
                    mdl(torch.from_numpy(X_adv).to(device).float())
                ).cpu().numpy().ravel()
            y_pred_adv = (p_adv >= 0.5).astype(int)
            metrics_fgsm = write_metrics_and_plots(name, "FGSM", y_te, y_pred_adv, p_adv, p_adv, ensure_dir(out_base / "adv_FGSM"))
            save_attack_trace(name, "FGSM", X_te, X_adv, y_te, y_pred_nat, y_pred_adv, ensure_dir(out_base / "adv_FGSM"))
            combined_entries["FGSM"].append({"name": name, "y_true": y_te, "y_score": p_adv, "metrics": metrics_fgsm})

            # PGD
            X_adv, steps_info = attack_pgd(
                mdl, X_te, y_te,
                ATTACKCFG["eps_tab_pgd"], ATTACKCFG["alpha_tab_pgd"], ATTACKCFG["steps_tab_pgd"], device
            )
            with torch.no_grad():
                p_adv = torch.sigmoid(
                    mdl(torch.from_numpy(X_adv).to(device).float())
                ).cpu().numpy().ravel()
            y_pred_adv = (p_adv >= 0.5).astype(int)
            metrics_pgd = write_metrics_and_plots(name, "PGD", y_te, y_pred_adv, p_adv, p_adv, ensure_dir(out_base / "adv_PGD"))
            save_attack_trace(name, "PGD", X_te, X_adv, y_te, y_pred_nat, y_pred_adv, ensure_dir(out_base / "adv_PGD"), steps_info)
            combined_entries["PGD"].append({"name": name, "y_true": y_te, "y_score": p_adv, "metrics": metrics_pgd})

        else:
            # Random_Forest / LightGBM (teacher) + surrogate attacker
            mdl = joblib.load(mdir / "model.joblib")

            # NAT
            if hasattr(mdl, "predict_proba"):
                p_nat = mdl.predict_proba(X_te)[:, 1]
                y_pred_nat = (p_nat >= 0.5).astype(int)
                score_nat = p_nat
                proba_nat = p_nat
            else:
                y_pred_nat = mdl.predict(X_te)
                score_nat = y_pred_nat
                proba_nat = None
            metrics_nat = write_metrics_and_plots(name, "NAT", y_te, y_pred_nat, score_nat, proba_nat, ensure_dir(out_base / "adv_NAT"))
            combined_entries["NAT"].append({"name": name, "y_true": y_te, "y_score": score_nat, "metrics": metrics_nat})

            # surrogate distilled on TRAIN
            surr = distill_surrogate(mdl, X_tr, y_tr, device, tag=name)

            # FGSM via surrogate, evaluate with teacher
            X_adv = attack_fgsm(surr, X_te, y_te, ATTACKCFG["eps_tab_fgsm"], device)
            if hasattr(mdl, "predict_proba"):
                p_adv = mdl.predict_proba(X_adv)[:, 1]
                y_pred_adv = (p_adv >= 0.5).astype(int)
                score_adv = p_adv
                proba_adv = p_adv
            else:
                y_pred_adv = mdl.predict(X_adv)
                score_adv = y_pred_adv
                proba_adv = None
            metrics_fgsm = write_metrics_and_plots(name, "FGSM", y_te, y_pred_adv, score_adv, proba_adv, ensure_dir(out_base / "adv_FGSM"))
            save_attack_trace(name, "FGSM", X_te, X_adv, y_te, y_pred_nat, y_pred_adv, ensure_dir(out_base / "adv_FGSM"))
            combined_entries["FGSM"].append({"name": name, "y_true": y_te, "y_score": score_adv, "metrics": metrics_fgsm})

            # PGD via surrogate, evaluate with teacher
            X_adv, steps_info = attack_pgd(
                surr, X_te, y_te,
                ATTACKCFG["eps_tab_pgd"], ATTACKCFG["alpha_tab_pgd"], ATTACKCFG["steps_tab_pgd"], device
            )
            if hasattr(mdl, "predict_proba"):
                p_adv = mdl.predict_proba(X_adv)[:, 1]
                y_pred_adv = (p_adv >= 0.5).astype(int)
                score_adv = p_adv
                proba_adv = p_adv
            else:
                y_pred_adv = mdl.predict(X_adv)
                score_adv = y_pred_adv
                proba_adv = None
            metrics_pgd = write_metrics_and_plots(name, "PGD", y_te, y_pred_adv, score_adv, proba_adv, ensure_dir(out_base / "adv_PGD"))
            save_attack_trace(name, "PGD", X_te, X_adv, y_te, y_pred_nat, y_pred_adv, ensure_dir(out_base / "adv_PGD"), steps_info)
            combined_entries["PGD"].append({"name": name, "y_true": y_te, "y_score": score_adv, "metrics": metrics_pgd})

        # collect for global summaries
        for rec in (metrics_nat, metrics_fgsm, metrics_pgd):
            all_records.append(rec)
            summary_accumulator.setdefault(rec["attack"], []).append(rec)

        cleanup_cuda()

    # --- combined single-figure per attack ---
    comb_dir = ensure_dir(results_root / "_combined")
    for atk in ["NAT", "FGSM", "PGD"]:
        render_combined_figure(atk, combined_entries.get(atk, []), comb_dir)

    # --- global exports (scalars only) ---
    save_json(all_records, results_root / "_metrics_all.json")

    # attack-wise averages over models (only numeric fields)
    summary = {}
    numeric_keys = [
        "accuracy","precision","recall","f1","balanced_accuracy","mcc","kappa",
        "roc_auc","pr_auc","specificity","tpr","tnr","fpr","fnr",
        "npv","fdr","for","prevalence","pred_pos_rate","youden_j","gmean",
        "threat_score","f0_5","f2","lr_plus","lr_minus","brier","log_loss_ce","ece"
    ]
    for atk, rows in summary_accumulator.items():
        agg = {}
        for k in numeric_keys:
            vals = [r.get(k) for r in rows if r.get(k) is not None and isinstance(r.get(k), (int, float))]
            agg[k] = float(np.mean(vals)) if vals else None
        summary[atk] = agg
    save_json(summary, results_root / "_summary_global_metrics.json")

    print("\n[DONE] Adversarial evaluation complete with rich metrics and single combined figures (no arrays, no calibration curve, no CSV).")

if __name__ == "__main__":
    main()
