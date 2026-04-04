#!/usr/bin/env python3
"""Generate the 3 remaining M4 daily forecast plots.

Uses speed optimizations:
- Cap training data to last 500 points
- 15 epochs instead of 30
- Replace timesnet with a lighter model set
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybridts.data import (
    M4_H, M4_P, best_L, load_m4_train_test,
    plot_forecast,
)
from hybridts.models import make_model, ets_forecast
from hybridts.models.helformer_pt import helformer_forecast_pt
from hybridts.training import TrainConfig, WindowDatasetStd, train_model

OUT_DIR = ROOT / "outputs" / "forecast_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

M4D_METRICS = ROOT / "outputs" / "m4_daily_merged.csv"

NEURAL_MODELS = ["nbeats", "dlinear", "patchtst", "timesnet"]
EPOCHS = 15
MAX_TRAIN_LEN = 500  # Cap training data for speed

CLASSICAL_COLS = ["seasonal_naive_sMAPE", "auto_arima_sMAPE", "ets_sMAPE", "prophet_sMAPE"]
NEURAL_COLS = ["dlinear_sMAPE", "autoformer_sMAPE", "fedformer_sMAPE", "nbeats_sMAPE",
               "patchtst_sMAPE", "timesnet_sMAPE"]


def select_3_series(metrics_df, cat):
    df = metrics_df[metrics_df["category"] == cat].copy()
    classical_present = [c for c in CLASSICAL_COLS if c in df.columns]
    neural_present = [c for c in NEURAL_COLS if c in df.columns]
    df["classical_mean"] = df[classical_present].mean(axis=1)
    df["neural_mean"] = df[neural_present].mean(axis=1)
    df["gap"] = df["neural_mean"] - df["classical_mean"]
    df = df.dropna(subset=["gap"])
    df = df.sort_values("gap")
    neural_best_id = df.iloc[0]["series_id"]
    classical_best_id = df.iloc[-1]["series_id"]
    mid_idx = len(df) // 2
    median_id = df.iloc[mid_idx]["series_id"]
    return [
        (neural_best_id, "neural_best"),
        (classical_best_id, "classical_best"),
        (median_id, "median"),
    ]


def train_and_forecast_neural(model_name, y_tr, H, per):
    try:
        # Cap training data for speed
        if len(y_tr) > MAX_TRAIN_LEN:
            y_tr = y_tr[-MAX_TRAIN_LEN:]
        L = best_L(y_tr, H, per)
        cfg = TrainConfig(lookback=L, horizon=H, epochs=EPOCHS, batch_size=64, lr=3e-3)
        ds = WindowDatasetStd(y_tr, L, H)
        if len(ds) == 0:
            return None
        mu, sd = ds.scaler
        model = make_model(model_name, cfg)
        train_model(model, ds, cfg)
        model.eval()

        z_tr = (y_tr - mu) / sd
        if len(z_tr) >= L:
            x = z_tr[-L:]
        else:
            pad_val = z_tr[0] if len(z_tr) > 0 else 0.0
            x = np.concatenate([np.full(L - len(z_tr), pad_val), z_tr])

        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            pred_z = model(x_t).squeeze().cpu().numpy()

        pred = pred_z * sd + mu
        return pred[:H]
    except Exception as e:
        print(f"    [WARN] {model_name} failed: {e}")
        return None


def train_and_forecast_helformer(y_tr, H, per):
    try:
        if len(y_tr) > MAX_TRAIN_LEN:
            y_tr = y_tr[-MAX_TRAIN_LEN:]
        L = best_L(y_tr, 1, per)
        cfg = TrainConfig(lookback=L, horizon=1, epochs=EPOCHS, batch_size=64, lr=3e-3)
        ds = WindowDatasetStd(y_tr, L, 1)
        if len(ds) == 0:
            return None
        model = make_model("helformer", cfg)
        train_model(model, ds, cfg)
        pred = helformer_forecast_pt(
            y_tr, horizon=H, model=model, lookback=L,
            seasonal_period=per, use_hw=True, device="cpu",
        )
        return pred[:H]
    except Exception as e:
        print(f"    [WARN] helformer failed: {e}")
        return None


def forecast_ets(y_tr, H, per):
    try:
        pred = ets_forecast(y_tr, H, per)
        return pred[:H]
    except Exception as e:
        print(f"    [WARN] ets failed: {e}")
        return None


if __name__ == "__main__":
    freq = "daily"
    H = M4_H[freq]
    per = M4_P[freq]

    metrics = pd.read_csv(M4D_METRICS)
    selected = select_3_series(metrics, freq)
    print(f"Selected daily series: {selected}")

    all_data = load_m4_train_test(freq)
    data_dict = {sid: (y_tr, y_te) for sid, y_tr, y_te in all_data}

    for series_id, plot_type in selected:
        print(f"\n  Series {series_id} ({plot_type})")
        if series_id not in data_dict:
            print(f"    [SKIP] Series {series_id} not found")
            continue

        y_tr, y_te = data_dict[series_id]
        print(f"    Train length: {len(y_tr)}, Test length: {len(y_te)}")

        forecasts = {}

        # ETS
        ets_pred = forecast_ets(y_tr, H, per)
        if ets_pred is not None:
            forecasts["ETS"] = ets_pred

        for mname in NEURAL_MODELS:
            print(f"    Training {mname}...")
            pred = train_and_forecast_neural(mname, y_tr, H, per)
            if pred is not None:
                label_map = {
                    "timesnet": "TimesNet", "nbeats": "N-BEATS",
                    "dlinear": "DLinear", "patchtst": "PatchTST",
                }
                forecasts[label_map.get(mname, mname)] = pred

        print(f"    Training helformer...")
        helf_pred = train_and_forecast_helformer(y_tr, H, per)
        if helf_pred is not None:
            forecasts["Helformer"] = helf_pred

        title = f"M4 daily -- {series_id} ({plot_type.replace('_',' ')})"
        fname = f"m4_daily_{series_id}_{plot_type}.png"
        save_path = OUT_DIR / fname
        plot_forecast(title, y_tr, y_te, forecasts, save_path=save_path)
        print(f"    Saved: {save_path}")

    print("\n=== M4 daily forecast plots done ===")
