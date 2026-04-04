#!/usr/bin/env python3
"""Generate 18 forecast plots (3 per category x 6 categories).

For each category, selects:
  1. Series where neural models do best vs classical (min gap)
  2. Series where classical models dominate (max gap)
  3. Median series

Retrains top 4 neural models + ETS + Helformer and plots actual vs predicted.
"""
from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

# Suppress noisy output
os.environ["TQDM_DISABLE"] = "1"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

# ── project imports ────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from hybridts.data import (
    M3_H, M3_P, M4_H, M4_P,
    best_L, load_train_tsts, load_m4_train_test,
    seasonal_naive, plot_forecast,
)
from hybridts.models import make_model, ets_forecast
from hybridts.models.helformer_pt import helformer_forecast_pt, HelformerPT
from hybridts.training import TrainConfig, WindowDatasetStd, train_model

# ── paths ──────────────────────────────────────────────────────────────
OUT_DIR = ROOT / "outputs" / "forecast_plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Metrics CSVs
M3_METRICS = ROOT / "outputs" / "m3_benchmark_v3" / "metrics.csv"
M4QM_METRICS = ROOT / "outputs" / "m4_benchmark_qm" / "metrics.csv"
M4D_METRICS = ROOT / "outputs" / "m4_daily_merged.csv"
HELF_M3_METRICS = ROOT / "outputs" / "helformer_m3_benchmark" / "metrics.csv"
HELF_M4_METRICS = ROOT / "outputs" / "helformer_m4_benchmark" / "metrics.csv"

# ── config ─────────────────────────────────────────────────────────────
NEURAL_MODELS = ["timesnet", "nbeats", "dlinear", "patchtst"]
EPOCHS = 30

CLASSICAL_COLS = ["seasonal_naive_sMAPE", "auto_arima_sMAPE", "ets_sMAPE", "prophet_sMAPE"]
NEURAL_COLS = ["dlinear_sMAPE", "autoformer_sMAPE", "fedformer_sMAPE", "nbeats_sMAPE",
               "patchtst_sMAPE", "timesnet_sMAPE"]

# ── category definitions ──────────────────────────────────────────────
CATEGORIES = [
    # (label, dataset_prefix, freq, H_dict, P_dict, loader_func, metrics_csv)
    ("m3", "yearly",    M3_H, M3_P, load_train_tsts, M3_METRICS),
    ("m3", "quarterly", M3_H, M3_P, load_train_tsts, M3_METRICS),
    ("m3", "monthly",   M3_H, M3_P, load_train_tsts, M3_METRICS),
    ("m4", "quarterly", M4_H, M4_P, load_m4_train_test, M4QM_METRICS),
    ("m4", "monthly",   M4_H, M4_P, load_m4_train_test, M4QM_METRICS),
    ("m4", "daily",     M4_H, M4_P, load_m4_train_test, M4D_METRICS),
]


def select_3_series(metrics_df: pd.DataFrame, cat: str) -> list[tuple[str, str]]:
    """Select 3 representative series: neural_best, classical_best, median.

    Returns list of (series_id, type_label).
    """
    df = metrics_df[metrics_df["category"] == cat].copy()

    # Compute mean classical vs neural sMAPE per series
    classical_present = [c for c in CLASSICAL_COLS if c in df.columns]
    neural_present = [c for c in NEURAL_COLS if c in df.columns]

    df["classical_mean"] = df[classical_present].mean(axis=1)
    df["neural_mean"] = df[neural_present].mean(axis=1)
    df["gap"] = df["neural_mean"] - df["classical_mean"]

    # Drop series with NaN gap
    df = df.dropna(subset=["gap"])
    if len(df) == 0:
        return []

    df = df.sort_values("gap")

    # 1. Min gap = neural best relative to classical
    neural_best_id = df.iloc[0]["series_id"]

    # 2. Max gap = classical best
    classical_best_id = df.iloc[-1]["series_id"]

    # 3. Median
    mid_idx = len(df) // 2
    median_id = df.iloc[mid_idx]["series_id"]

    return [
        (neural_best_id, "neural_best"),
        (classical_best_id, "classical_best"),
        (median_id, "median"),
    ]


def train_and_forecast_neural(model_name: str, y_tr: np.ndarray, H: int, per: int) -> np.ndarray | None:
    """Train a neural model and produce H-step forecast."""
    try:
        L = best_L(y_tr, H, per)
        cfg = TrainConfig(lookback=L, horizon=H, epochs=EPOCHS, batch_size=32, lr=3e-3)
        ds = WindowDatasetStd(y_tr, L, H)
        if len(ds) == 0:
            return None
        mu, sd = ds.scaler
        model = make_model(model_name, cfg)
        train_model(model, ds, cfg)
        model.eval()

        # Produce forecast using the last window
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


def train_and_forecast_helformer(y_tr: np.ndarray, H: int, per: int) -> np.ndarray | None:
    """Train Helformer and produce H-step forecast via rollout."""
    try:
        L = best_L(y_tr, 1, per)  # horizon=1 for helformer training
        cfg = TrainConfig(lookback=L, horizon=1, epochs=EPOCHS, batch_size=32, lr=3e-3)
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


def forecast_ets(y_tr: np.ndarray, H: int, per: int) -> np.ndarray | None:
    """Produce ETS forecast."""
    try:
        pred = ets_forecast(y_tr, H, per)
        return pred[:H]
    except Exception as e:
        print(f"    [WARN] ets failed: {e}")
        return None


def process_category(ds_name: str, freq: str, H_dict: dict, P_dict: dict,
                     loader, metrics_path: Path):
    """Process one category: select 3 series, train models, plot."""
    cat_label = f"{ds_name}_{freq}"
    print(f"\n=== Processing {cat_label} ===")

    H = H_dict[freq]
    per = P_dict[freq]

    # Load metrics
    metrics = pd.read_csv(metrics_path)

    # Select 3 representative series
    selected = select_3_series(metrics, freq)
    if not selected:
        print(f"  No valid series found for {cat_label}")
        return

    # Load all data for this category
    all_data = loader(freq)
    data_dict = {sid: (y_tr, y_te) for sid, y_tr, y_te in all_data}

    for series_id, plot_type in selected:
        print(f"  Series {series_id} ({plot_type})")

        if series_id not in data_dict:
            print(f"    [SKIP] Series {series_id} not found in loaded data")
            continue

        y_tr, y_te = data_dict[series_id]

        forecasts = {}

        # ETS
        ets_pred = forecast_ets(y_tr, H, per)
        if ets_pred is not None:
            forecasts["ETS"] = ets_pred

        # Neural models
        for mname in NEURAL_MODELS:
            print(f"    Training {mname}...")
            pred = train_and_forecast_neural(mname, y_tr, H, per)
            if pred is not None:
                # Use nice labels
                label_map = {
                    "timesnet": "TimesNet", "nbeats": "N-BEATS",
                    "dlinear": "DLinear", "patchtst": "PatchTST",
                }
                forecasts[label_map.get(mname, mname)] = pred

        # Helformer
        print(f"    Training helformer...")
        helf_pred = train_and_forecast_helformer(y_tr, H, per)
        if helf_pred is not None:
            forecasts["Helformer"] = helf_pred

        # Plot
        title = f"{ds_name.upper()} {freq} -- {series_id} ({plot_type.replace('_',' ')})"
        fname = f"{ds_name}_{freq}_{series_id}_{plot_type}.png"
        save_path = OUT_DIR / fname

        plot_forecast(title, y_tr, y_te, forecasts, save_path=save_path)
        print(f"    Saved: {save_path}")


if __name__ == "__main__":
    for ds_name, freq, H_dict, P_dict, loader, metrics_path in CATEGORIES:
        process_category(ds_name, freq, H_dict, P_dict, loader, metrics_path)

    print(f"\n=== All forecast plots saved to {OUT_DIR} ===")
