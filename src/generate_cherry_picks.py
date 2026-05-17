#!/usr/bin/env python3
"""Generate cherry-pick forecast plots for thesis.

Selects daily series where neural networks beat classical models,
trains all models, and produces comparative forecast plots.
"""
import sys, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybridts.data import (
    M4_H, M4_P, best_L,
    ensure_m4_csv, load_m4_train_test,
    smape, seasonal_naive as snaive_forecast,
)
from hybridts.models import make_model
from hybridts.models.helformer_pt import helformer_forecast_pt
from hybridts.training import TrainConfig, WindowDatasetStd, train_model

import torch

ROOT = Path(__file__).resolve().parent.parent
FIGURES = ROOT / "thesis" / "figures"
FIGURES.mkdir(exist_ok=True)

H = M4_H["daily"]  # 14
P = M4_P["daily"]  # 7

NEURAL_MODELS = ["dlinear", "nbeats", "timesnet", "patchtst", "autoformer"]
COLORS = {
    "ground_truth": "#333333",
    "train": "#CCCCCC",
    "seasonal_naive": "#888888",
    "auto_arima": "#2196F3",
    "ets": "#4CAF50",
    "dlinear": "#E74C3C",
    "autoformer": "#F48FB1",
    "patchtst": "#6EBF6E",
    "nbeats": "#3498DB",
    "timesnet": "#FF6F00",
    "helformer": "#9B59B6",
}
LABELS = {
    "seasonal_naive": "S. Naive", "auto_arima": "Auto-ARIMA", "ets": "ETS",
    "dlinear": "DLinear", "autoformer": "Autoformer",
    "patchtst": "PatchTST", "nbeats": "N-BEATS",
    "timesnet": "TimesNet", "helformer": "Helformer",
}


def train_and_forecast(model_name, y_tr, epochs=30, device="cpu"):
    L = best_L(y_tr, H, P)
    # Cap training data for speed
    if len(y_tr) > 500:
        y_tr_use = y_tr[-500:]
    else:
        y_tr_use = y_tr
    L = min(L, len(y_tr_use) - H - 1)
    if L < 4:
        L = max(4, len(y_tr_use) // 3)

    n_windows = len(y_tr_use) - L - H + 1
    bs = min(128, max(16, n_windows))
    cfg = TrainConfig(lookback=L, horizon=H, epochs=epochs, batch_size=bs,
                      lr=1e-3, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model(model_name, cfg)
    ds = WindowDatasetStd(y_tr_use, L, H)
    if len(ds) < 1:
        return None
    train_model(model, ds, cfg)
    mu, sd = ds.scaler
    model.eval()
    with torch.no_grad():
        x = torch.tensor(y_tr_use[-L:], dtype=torch.float32)
        x_norm = (x - mu) / sd
        x_in = x_norm.unsqueeze(0).unsqueeze(0).to(cfg.device)
        pred_norm = model(x_in).squeeze(0).cpu().numpy()
    return np.asarray(pred_norm * sd + mu, float).ravel()[:H]


def forecast_ets(y_tr):
    try:
        from statsforecast.models import AutoETS
        model = AutoETS(season_length=P)
        model.fit(y_tr)
        return model.predict(h=H)["mean"]
    except:
        return None


def forecast_arima(y_tr):
    try:
        from statsforecast.models import AutoARIMA
        model = AutoARIMA(season_length=P)
        model.fit(y_tr)
        return model.predict(h=H)["mean"]
    except:
        return None


def plot_forecast(y_tr, y_te, forecasts, series_id, output_path, show_last=100):
    """Plot train tail + test + forecasts."""
    fig, ax = plt.subplots(figsize=(14, 5))

    # Train data (last N points)
    n_show = min(show_last, len(y_tr))
    train_x = range(len(y_tr) - n_show, len(y_tr))
    ax.plot(train_x, y_tr[-n_show:], color=COLORS["train"], linewidth=1.5,
            label="Обучение", zorder=1)

    # Test data
    test_x = range(len(y_tr), len(y_tr) + H)
    ax.plot(test_x, y_te[:H], color=COLORS["ground_truth"], linewidth=2.5,
            linestyle="--", label="Факт", zorder=10)

    # Forecasts
    for model_name, pred in sorted(forecasts.items(), key=lambda x: x[0]):
        if pred is not None:
            s = smape(y_te[:H], pred)
            lbl = f"{LABELS.get(model_name, model_name)} ({s:.1f}%)"
            ax.plot(test_x, pred, linewidth=1.5, label=lbl,
                    color=COLORS.get(model_name, "#999"), alpha=0.85, zorder=5)

    ax.set_xlabel("Время")
    ax.set_ylabel("Значение")
    ax.set_title(f"Прогнозы моделей на ряде {series_id} (M4 Daily, H={H})")
    ax.legend(loc="upper left", fontsize=9, ncol=2)
    ax.axvline(x=len(y_tr) - 0.5, color="#DDD", linestyle=":", zorder=0)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  -> {output_path.name}")


def main():
    print("Loading M4 daily data...")
    ensure_m4_csv(categories=("daily",))
    all_pairs = load_m4_train_test("daily")
    pair_dict = {sid: (y_tr, y_te) for sid, y_tr, y_te in all_pairs}

    # Cherry-pick series (from analysis)
    cherry_picks = [
        ("D3735", "cherry_daily_neural_win_1"),   # TN=0.71 vs ARIMA=3.47
        ("D1816", "cherry_daily_all_neural_win"),  # ALL 7 neural beat ALL 3 classical
        ("D2461", "cherry_daily_neural_win_2"),    # TN=2.09 vs ARIMA=6.08
        ("D1117", "cherry_daily_neural_win_3"),    # ALL neural win, best=0.56
    ]

    for series_id, filename in cherry_picks:
        if series_id not in pair_dict:
            print(f"  SKIP {series_id}: not in dataset")
            continue

        y_tr, y_te = pair_dict[series_id]
        print(f"\n=== {series_id} (len={len(y_tr)}) ===")

        forecasts = {}

        # Classical
        print("  S. Naive...")
        forecasts["seasonal_naive"] = snaive_forecast(y_tr, H, P)

        print("  Auto-ARIMA...")
        forecasts["auto_arima"] = forecast_arima(y_tr)

        print("  ETS...")
        forecasts["ets"] = forecast_ets(y_tr)

        # Neural
        for m in NEURAL_MODELS:
            print(f"  {m}...")
            try:
                forecasts[m] = train_and_forecast(m, y_tr, epochs=30)
            except Exception as e:
                print(f"    ERROR: {e}")
                forecasts[m] = None

        output_path = FIGURES / f"{filename}.png"
        plot_forecast(y_tr, y_te, forecasts, series_id, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()
