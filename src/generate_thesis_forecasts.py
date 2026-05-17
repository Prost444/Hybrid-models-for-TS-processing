#!/usr/bin/env python3
"""Generate per-series forecasts for thesis cherry-pick plots.

Trains all 11 models on selected series, saves predictions as CSV.

Usage:
    CUDA_VISIBLE_DEVICES=0 python generate_thesis_forecasts.py --device cuda
    python generate_thesis_forecasts.py --device cpu
"""
import argparse, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybridts.data import (
    M4_H, M4_P, best_L,
    ensure_m4_csv, load_m4_train_test,
    smape,
)
from hybridts.models import make_model
from hybridts.models.helformer_pt import helformer_forecast_pt
from hybridts.training import TrainConfig, WindowDatasetStd, train_model

import torch

OUTDIR = Path("outputs/thesis_forecasts")

# ── Cherry-pick series ──────────────────────────────────────────
# Daily: ALL 7 neural beat classical
DAILY_PICKS = ["D1117", "D1816", "D1941", "D2133", "D2361", "D4121",
               "D2517", "D2612"]
# Daily: single-model dramatic wins
DAILY_DRAMATIC = ["D2173", "D2172", "D2052", "D4166"]

NEURAL = ["dlinear", "autoformer", "fedformer", "patchtst", "nbeats", "timesnet", "helformer"]
CLASSICAL = ["seasonal_naive", "auto_arima", "ets"]
ALL_MODELS = CLASSICAL + NEURAL


def seasonal_naive_forecast(y_tr, H, P):
    if P < 1 or P > len(y_tr):
        P = 1
    last_season = y_tr[-P:]
    reps = (H // P) + 1
    return np.tile(last_season, reps)[:H]


def forecast_auto_arima(y_tr, H):
    from statsforecast.models import AutoARIMA
    m = AutoARIMA(season_length=1)
    m.fit(y_tr)
    return m.predict(h=H)["mean"]


def forecast_ets(y_tr, H, P):
    from statsforecast.models import AutoETS
    m = AutoETS(season_length=P)
    m.fit(y_tr)
    return m.predict(h=H)["mean"]


def train_neural(model_name, y_tr, H, P, epochs, device):
    L = best_L(y_tr, H, P)
    n_windows = len(y_tr) - L - H + 1
    bs = 256 if n_windows > 1024 else (128 if n_windows > 256 else 32)
    cfg = TrainConfig(lookback=L, horizon=H, epochs=epochs, batch_size=bs,
                      lr=1e-3, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model(model_name, cfg)
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


def predict_helformer(y_tr, H, P, epochs, device):
    L = min(30, len(y_tr) - 2)
    if L < 4:
        L = max(4, len(y_tr) // 2)
    cfg = TrainConfig(lookback=L, horizon=1, epochs=epochs, batch_size=32,
                      lr=8e-4, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model("helformer", cfg)
    vmin, scale = float(np.nanmin(y_tr)), float(np.nanmax(y_tr)) - float(np.nanmin(y_tr))
    if scale < 1e-8: scale = 1.0
    scaled = (y_tr - vmin) / scale
    ds = WindowDatasetStd(scaled, L, 1, scale=False)
    if len(ds) < 1: return None
    train_model(model, ds, cfg)
    return helformer_forecast_pt(y_tr, H, model, lookback=L,
                                 seasonal_period=P, use_hw=True, device=device)


def forecast_one_series(sid, y_tr, y_te, H, P, epochs, device):
    """Train all models and return dict {model: predictions}."""
    results = {}

    # Seasonal Naive
    try:
        results["seasonal_naive"] = seasonal_naive_forecast(y_tr, H, P)
    except Exception as e:
        print(f"  seasonal_naive error: {e}")

    # Auto-ARIMA
    try:
        results["auto_arima"] = np.asarray(forecast_auto_arima(y_tr, H), float)
    except Exception as e:
        print(f"  auto_arima error: {e}")

    # ETS
    try:
        results["ets"] = np.asarray(forecast_ets(y_tr, H, P), float)
    except Exception as e:
        print(f"  ets error: {e}")

    # Neural models
    for m in NEURAL:
        try:
            if m == "helformer":
                pred = predict_helformer(y_tr, H, P, epochs, device)
            else:
                pred = train_neural(m, y_tr, H, P, epochs, device)
            if pred is not None:
                results[m] = np.asarray(pred, float).ravel()[:H]
        except Exception as e:
            print(f"  {m} error: {e}")

        if device == "cuda":
            torch.cuda.empty_cache()

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--category", default="daily", choices=["daily", "quarterly", "monthly"])
    args = parser.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    # Load data
    cat = args.category
    if cat == "daily":
        ensure_m4_csv(categories=("daily",))
        all_pairs = load_m4_train_test("daily")
        H, P = M4_H["daily"], M4_P["daily"]
        picks = DAILY_PICKS + DAILY_DRAMATIC
    else:
        ensure_m4_csv(categories=(cat,))
        all_pairs = load_m4_train_test(cat)
        H, P = M4_H[cat], M4_P[cat]
        picks = []  # will add later if needed

    pair_dict = {sid: (y_tr, y_te) for sid, y_tr, y_te in all_pairs}
    print(f"Loaded {len(pair_dict)} {cat} series, processing {len(picks)} picks")

    for sid in picks:
        if sid not in pair_dict:
            print(f"  {sid}: NOT FOUND, skipping")
            continue

        y_tr, y_te = pair_dict[sid]
        print(f"\n{'='*60}")
        print(f"  {sid}: train={len(y_tr)}, test={len(y_te)}, H={H}")
        t0 = time.time()

        results = forecast_one_series(sid, y_tr, y_te, H, P, args.epochs, device)

        # Save train data
        train_df = pd.DataFrame({"step": range(len(y_tr)), "value": y_tr})
        train_df.to_csv(OUTDIR / f"{sid}_train.csv", index=False)

        # Save predictions
        pred_df = pd.DataFrame({"step": range(H), "actual": y_te[:H]})
        for m, pred in results.items():
            pred_df[m] = pred[:H]
            s = smape(y_te[:H], pred[:H])
            print(f"    {m}: sMAPE={s:.2f}")
        pred_df.to_csv(OUTDIR / f"{sid}_predictions.csv", index=False)

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.0f}s, {len(results)} models")

    print(f"\n\nAll done! Predictions saved to {OUTDIR}/")


if __name__ == "__main__":
    main()
