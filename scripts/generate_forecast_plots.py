#!/usr/bin/env python3
"""Generate forecast visualisation plots: 3 representative series × 6 categories.

For each category, selects:
  1. Series where neural models perform best relative to classical
  2. Series where classical models dominate
  3. Median-quality series

Re-trains all 10 models on the selected series and plots actual vs predicted.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- project bootstrap ----------------------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_paths import ensure_src_on_path
ensure_src_on_path()

import torch
from hybridts.config.settings import settings
from hybridts.data import (
    M3_H, M3_P, M4_H, M4_P, best_L,
    ensure_m3_csv, load_train_tsts, ensure_m4_csv, load_m4_train_test,
    seasonal_naive, smape, plot_forecast,
)
from hybridts.models import make_model, auto_arima_forecast, ets_forecast, prophet_forecast
from hybridts.training import TrainConfig, WindowDatasetStd, train_model


def _train_and_predict(model_name, y_tr, H, P, epochs, device):
    L = best_L(y_tr, H, P)
    cfg = TrainConfig(lookback=L, horizon=H, epochs=epochs, batch_size=32,
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
    return pred_norm * sd + mu


def _select_representative(metrics_df, cat, models_neural, models_classical):
    """Pick 3 series: best-neural, best-classical, median."""
    sub = metrics_df[metrics_df["category"] == cat].copy()
    if sub.empty:
        return []

    neural_cols = [f"{m}_sMAPE" for m in models_neural if f"{m}_sMAPE" in sub.columns]
    classical_cols = [f"{m}_sMAPE" for m in models_classical if f"{m}_sMAPE" in sub.columns]

    if not neural_cols or not classical_cols:
        return sub["series_id"].tolist()[:3]

    sub["neural_mean"] = sub[neural_cols].mean(axis=1)
    sub["classical_mean"] = sub[classical_cols].mean(axis=1)
    sub["gap"] = sub["classical_mean"] - sub["neural_mean"]  # positive = neural better

    best_neural = sub.loc[sub["gap"].idxmax(), "series_id"]
    best_classical = sub.loc[sub["gap"].idxmin(), "series_id"]
    sub_sorted = sub.sort_values("gap")
    median_series = sub_sorted.iloc[len(sub_sorted) // 2]["series_id"]

    return [best_neural, best_classical, median_series]


def generate_plots(
    metrics_csv: str,
    dataset: str,
    categories: list[str],
    neural_models: list[str],
    classical_models: list[str],
    epochs: int = 30,
    device: str = "cpu",
    out_dir: str = "src/outputs/forecast_plots",
):
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(metrics_csv)
    freq_map = {"yearly": "YE", "quarterly": "QE", "monthly": "ME",
                "weekly": "W", "daily": "D", "hourly": "h"}

    if dataset == "m3":
        ensure_m3_csv()
        H_map, P_map = M3_H, M3_P
        load_fn = load_train_tsts
    else:
        ensure_m4_csv()
        H_map, P_map = M4_H, M4_P
        load_fn = load_m4_train_test

    for cat in categories:
        H = H_map[cat]
        P = P_map[cat]
        pairs = load_fn(cat)
        pairs_dict = {sid: (y_tr, y_te) for sid, y_tr, y_te in pairs}

        selected = _select_representative(df, cat, neural_models, classical_models)
        print(f"\n[{dataset} {cat}] Selected: {selected}")

        for sid in selected:
            if sid not in pairs_dict:
                print(f"  {sid} not found in data, skipping")
                continue
            y_tr, y_te = pairs_dict[sid]
            forecasts = {}

            # Classical
            try:
                forecasts["Seasonal Naive"] = seasonal_naive(y_tr, H, P)
            except Exception:
                pass
            try:
                forecasts["auto-ARIMA"] = np.asarray(auto_arima_forecast(y_tr, H), float)
            except Exception:
                pass
            try:
                forecasts["ETS"] = np.asarray(ets_forecast(y_tr, H, seasonal_periods=P), float)
            except Exception:
                pass

            # Neural (top 4 by name for readability)
            for nm in neural_models[:4]:
                try:
                    pred = _train_and_predict(nm, y_tr, H, P, epochs, device)
                    if pred is not None:
                        forecasts[nm] = np.asarray(pred, float).ravel()[:H]
                except Exception as e:
                    print(f"  {nm} failed on {sid}: {e}")

            save_path = out_path / f"{dataset}_{cat}_{sid}.png"
            plot_forecast(
                f"{dataset.upper()} {cat.capitalize()} — {sid} (H={H})",
                y_tr, y_te, forecasts, save_path=save_path,
            )
            print(f"  Saved {save_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate forecast plots")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--metrics", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="m3")
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="src/outputs/forecast_plots")
    args = parser.parse_args()

    if args.config and args.config.exists():
        with open(args.config) as f:
            cfg = json.load(f)
    else:
        cfg = {}

    dataset = cfg.get("dataset", args.dataset)
    categories = cfg.get("categories", args.categories)
    if categories is None:
        categories = ["yearly", "quarterly", "monthly"] if dataset == "m3" else ["quarterly", "monthly", "daily"]

    metrics_csv = cfg.get("metrics", args.metrics)
    if metrics_csv is None:
        if dataset == "m3":
            metrics_csv = "src/outputs/m3_benchmark_v3/metrics.csv"
        else:
            metrics_csv = "src/outputs/m4_benchmark_qm/metrics.csv"

    neural_models = cfg.get("neural_models", ["dlinear", "timesnet", "nbeats", "patchtst"])
    classical_models = cfg.get("classical_models", ["seasonal_naive", "auto_arima", "ets"])

    generate_plots(
        metrics_csv=metrics_csv,
        dataset=dataset,
        categories=categories,
        neural_models=neural_models,
        classical_models=classical_models,
        epochs=cfg.get("epochs", args.epochs),
        device=cfg.get("device", args.device),
        out_dir=cfg.get("out_dir", args.out_dir),
    )


if __name__ == "__main__":
    main()
