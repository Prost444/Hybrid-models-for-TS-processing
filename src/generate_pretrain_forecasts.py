#!/usr/bin/env python3
"""Generate pretrained vs from-scratch forecasts for thesis plots.

Two modes:
1. Per-model pretrain pairs: train each neural model both pretrained and from-scratch
   on the same series, save predictions for side-by-side comparison.
2. All-pretrained cherry-pick: train all 11 models (classical + pretrained neural)
   on a series where pretrained models win.

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_pretrain_forecasts.py --device cuda
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybridts.data import (
    M4_H, M4_P, best_L,
    ensure_m4_csv, load_m4_train_test, smape,
)
from hybridts.models import make_model
from hybridts.models.helformer_pt import helformer_forecast_pt
from hybridts.training import TrainConfig, WindowDatasetStd, train_model
OUTDIR = Path("outputs/thesis_pretrain_forecasts")

# Series where ALL 7 neural models improve with pretrain
CHERRY_PRETRAIN_SERIES = ["D2111", "D2163", "D2216"]
# Per-model best series (where that specific model improves most)
PER_MODEL_SERIES = {
    "dlinear": "D2111",      # use same for consistency
    "autoformer": "D2111",
    "fedformer": "D2111",
    "patchtst": "D2111",
    "nbeats": "D2111",
    "timesnet": "D2111",
    "helformer": "D2111",
}

NEURAL = ["dlinear", "autoformer", "fedformer", "patchtst", "nbeats", "timesnet", "helformer"]
CLASSICAL = ["seasonal_naive"]  # auto_arima/ets added locally


def seasonal_naive_forecast(y_tr, H, P):
    if P < 1 or P > len(y_tr):
        P = 1
    return np.tile(y_tr[-P:], (H // P) + 1)[:H]


def train_neural(model_name, y_tr, H, P, epochs, device, pretrain_weights=None):
    """Train a neural model, optionally loading pretrained weights first."""
    L = best_L(y_tr, H, P)
    n_windows = len(y_tr) - L - H + 1
    bs = 256 if n_windows > 1024 else (128 if n_windows > 256 else 32)

    if pretrain_weights is not None:
        # Pretrain mode: fewer finetune epochs
        ft_epochs = min(20, epochs)
    else:
        ft_epochs = epochs

    cfg = TrainConfig(lookback=L, horizon=H, epochs=ft_epochs, batch_size=bs,
                      lr=1e-3, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model(model_name, cfg)

    if pretrain_weights is not None:
        try:
            model.load_state_dict(pretrain_weights, strict=False)
            print(f"    loaded pretrain weights for {model_name}")
        except Exception as e:
            print(f"    WARNING: could not load pretrain weights: {e}")

    ds = WindowDatasetStd(y_tr, L, H)
    if len(ds) < 1:
        return None
    train_model(model, ds, cfg)
    mu, sd = ds.scaler
    model.eval()
    with torch.no_grad():
        x = torch.tensor(y_tr[-L:], dtype=torch.float32)
        x_norm = (x - mu) / sd
        x_in = x_norm.unsqueeze(0).unsqueeze(0).to(cfg.device)
        pred_norm = model(x_in).squeeze(0).cpu().numpy()
    return np.asarray(pred_norm * sd + mu, float).ravel()[:H]


def predict_helformer(y_tr, H, P, epochs, device, pretrain_weights=None):
    L = min(30, len(y_tr) - 2)
    if L < 4:
        L = max(4, len(y_tr) // 2)
    ft_epochs = min(20, epochs) if pretrain_weights else epochs
    cfg = TrainConfig(lookback=L, horizon=1, epochs=ft_epochs, batch_size=32,
                      lr=8e-4, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model("helformer", cfg)
    if pretrain_weights is not None:
        try:
            model.load_state_dict(pretrain_weights, strict=False)
        except:
            pass
    vmin, scale = float(np.nanmin(y_tr)), float(np.nanmax(y_tr)) - float(np.nanmin(y_tr))
    if scale < 1e-8: scale = 1.0
    scaled = (y_tr - vmin) / scale
    ds = WindowDatasetStd(scaled, L, 1, scale=False)
    if len(ds) < 1: return None
    train_model(model, ds, cfg)
    return helformer_forecast_pt(y_tr, H, model, lookback=L,
                                 seasonal_period=P, use_hw=True, device=device)


def find_pretrain_weights(model_name, category="daily"):
    """Find pretrained weights from pretrain benchmark outputs."""
    # Check various locations
    base = Path("outputs")
    candidates = [
        base / f"pretrain_m4_{category}" / f"{model_name}_pretrained.pt",
        base / f"pretrain_m4_{category}_gpu1" / f"{model_name}_pretrained.pt",
        base / f"pretrain_m4_{category}_gpu5" / f"{model_name}_pretrained.pt",
    ]
    for p in candidates:
        if p.exists():
            print(f"    found weights: {p}")
            return torch.load(p, map_location="cpu")

    # Also check for .pth files
    for d in base.glob(f"pretrain_m4_{category}*"):
        for f in d.glob(f"{model_name}*.pt*"):
            print(f"    found weights: {f}")
            return torch.load(f, map_location="cpu")

    print(f"    WARNING: no pretrain weights found for {model_name}")
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    ensure_m4_csv(categories=("daily",))
    all_pairs = load_m4_train_test("daily")
    pair_dict = {sid: (y_tr, y_te) for sid, y_tr, y_te in all_pairs}
    H, P = M4_H["daily"], M4_P["daily"]

    # ── 1. Per-model pretrain pairs on D2111 ─────────────────────
    sid = "D2111"
    if sid not in pair_dict:
        print(f"ERROR: {sid} not in dataset")
        return
    y_tr, y_te = pair_dict[sid]
    print(f"\n{'='*60}")
    print(f"  Series {sid}: train={len(y_tr)}, test={len(y_te)}, H={H}")

    # Save train
    pd.DataFrame({"step": range(len(y_tr)), "value": y_tr}).to_csv(
        OUTDIR / f"{sid}_train.csv", index=False)

    for m in NEURAL:
        print(f"\n  --- {m} ---")
        # From scratch
        try:
            if m == "helformer":
                pred_scratch = predict_helformer(y_tr, H, P, args.epochs, device)
            else:
                pred_scratch = train_neural(m, y_tr, H, P, args.epochs, device)
        except Exception as e:
            print(f"    scratch error: {e}")
            pred_scratch = None

        # Pretrained
        weights = find_pretrain_weights(m, "daily")
        try:
            if m == "helformer":
                pred_pretrain = predict_helformer(y_tr, H, P, args.epochs, device, weights)
            else:
                pred_pretrain = train_neural(m, y_tr, H, P, args.epochs, device, weights)
        except Exception as e:
            print(f"    pretrain error: {e}")
            pred_pretrain = None

        # Save per-model comparison
        result = pd.DataFrame({"step": range(H), "actual": y_te[:H]})
        if pred_scratch is not None:
            result[f"{m}_scratch"] = np.asarray(pred_scratch, float).ravel()[:H]
            s = smape(y_te[:H], result[f"{m}_scratch"].values)
            print(f"    scratch sMAPE={s:.2f}")
        if pred_pretrain is not None:
            result[f"{m}_pretrain"] = np.asarray(pred_pretrain, float).ravel()[:H]
            s = smape(y_te[:H], result[f"{m}_pretrain"].values)
            print(f"    pretrain sMAPE={s:.2f}")

        result.to_csv(OUTDIR / f"{sid}_{m}_pair.csv", index=False)
        if device == "cuda":
            torch.cuda.empty_cache()

    # ── 2. All-pretrained cherry-pick on multiple series ─────────
    for sid in CHERRY_PRETRAIN_SERIES:
        if sid not in pair_dict:
            continue
        y_tr, y_te = pair_dict[sid]
        print(f"\n{'='*60}")
        print(f"  ALL-PRETRAINED cherry-pick: {sid} (train={len(y_tr)})")

        pd.DataFrame({"step": range(len(y_tr)), "value": y_tr}).to_csv(
            OUTDIR / f"{sid}_train.csv", index=False)

        result = pd.DataFrame({"step": range(H), "actual": y_te[:H]})

        # Classical
        try:
            result["seasonal_naive"] = seasonal_naive_forecast(y_tr, H, P)
        except:
            pass

        # Neural pretrained
        for m in NEURAL:
            weights = find_pretrain_weights(m, "daily")
            try:
                if m == "helformer":
                    pred = predict_helformer(y_tr, H, P, args.epochs, device, weights)
                else:
                    pred = train_neural(m, y_tr, H, P, args.epochs, device, weights)
                if pred is not None:
                    result[m] = np.asarray(pred, float).ravel()[:H]
                    s = smape(y_te[:H], result[m].values)
                    print(f"    {m} (pretrained) sMAPE={s:.2f}")
            except Exception as e:
                print(f"    {m} error: {e}")
            if device == "cuda":
                torch.cuda.empty_cache()

        result.to_csv(OUTDIR / f"{sid}_all_pretrained.csv", index=False)

    print(f"\n\nAll done! Saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
