#!/usr/bin/env python3
"""Plot cherry-pick time series forecasts for thesis.

Reads prediction CSVs from outputs/thesis_forecasts/ and generates
publication-quality plots showing train tail + test + model predictions.

Usage:
    python plot_cherry_picks.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

PRED_DIR = Path("outputs/thesis_forecasts")
FIG_DIR = Path("../thesis/figures")

# Colors
COLORS = {
    "seasonal_naive": "#95A5A6",
    "auto_arima": "#E74C3C",
    "ets": "#F39C12",
    "dlinear": "#3498DB",
    "autoformer": "#E67E22",
    "fedformer": "#D35400",
    "patchtst": "#2ECC71",
    "nbeats": "#1ABC9C",
    "timesnet": "#9B59B6",
    "helformer": "#E91E63",
}

LABELS = {
    "seasonal_naive": "S. Naive",
    "auto_arima": "ARIMA",
    "ets": "ETS",
    "dlinear": "DLinear",
    "autoformer": "Autoformer",
    "fedformer": "FEDformer",
    "patchtst": "PatchTST",
    "nbeats": "N-BEATS",
    "timesnet": "TimesNet",
    "helformer": "Helformer",
}

plt.rcParams.update({
    'font.size': 12,
    'figure.dpi': 150,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def smape(actual, pred):
    return 200 * np.mean(np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8))


def plot_forecast(sid, train_df, pred_df, fig_dir, tail_len=100):
    """Create a single forecast plot for a cherry-pick series."""
    train = train_df["value"].values
    H = len(pred_df)
    actual = pred_df["actual"].values

    # Get available models
    models = [c for c in pred_df.columns if c not in ["step", "actual"]]

    # Compute sMAPE for each model
    model_smapes = {}
    for m in models:
        pred = pred_df[m].values
        if not np.any(np.isnan(pred)):
            model_smapes[m] = smape(actual, pred)

    # Sort by sMAPE
    sorted_models = sorted(model_smapes.keys(), key=lambda m: model_smapes[m])

    # Determine best neural and best classical
    classical = ["seasonal_naive", "auto_arima", "ets"]
    best_classical = [m for m in sorted_models if m in classical]
    best_neural = [m for m in sorted_models if m not in classical]

    # Show ALL available models, sorted by sMAPE
    show_models = sorted_models

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Train tail
    tail = train[-tail_len:]
    t_train = np.arange(len(train) - tail_len, len(train))
    ax.plot(t_train, tail, color='#BDC3C7', linewidth=1, label='train')

    # Test actual
    t_test = np.arange(len(train), len(train) + H)
    ax.plot(t_test, actual, 'k-', linewidth=2.5, label='actual', zorder=10)

    # Model predictions — all models, best ones thicker
    for rank, m in enumerate(show_models):
        pred = pred_df[m].values
        color = COLORS.get(m, '#666')
        label = f"{LABELS.get(m, m)} ({model_smapes[m]:.1f}%)"
        # Top 3 thicker, rest thinner
        lw = 2.0 if rank < 3 else 1.0
        ls = '--' if m in classical else '-'
        alpha = 0.9 if rank < 3 else 0.5
        ax.plot(t_test, pred, color=color, linewidth=lw, linestyle=ls,
                label=label, alpha=alpha)

    # Vertical line at forecast start
    ax.axvline(x=len(train), color='black', linestyle=':', alpha=0.3)

    ax.set_xlabel('Временной шаг')
    ax.set_ylabel('Значение')
    ax.set_title(f'Ряд {sid}: прогнозы 11 моделей на горизонте H={H} '
                 f'(train={len(train)} точек)')
    ax.legend(loc='best', fontsize=7, framealpha=0.9, ncol=2)

    plt.tight_layout()
    fig_path = fig_dir / f"cherry_{sid}.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"  saved {fig_path}")
    return model_smapes


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Find all prediction files
    pred_files = sorted(PRED_DIR.glob("*_predictions.csv"))

    if not pred_files:
        print(f"No prediction files found in {PRED_DIR}/")
        return

    print(f"Found {len(pred_files)} prediction files")

    for pf in pred_files:
        sid = pf.stem.replace("_predictions", "")
        train_file = PRED_DIR / f"{sid}_train.csv"

        if not train_file.exists():
            print(f"  {sid}: missing train file, skipping")
            continue

        print(f"\n{sid}:")
        train_df = pd.read_csv(train_file)
        pred_df = pd.read_csv(pf)

        smapes = plot_forecast(sid, train_df, pred_df, FIG_DIR, tail_len=120)

        # Print sMAPE comparison
        classical = ["seasonal_naive", "auto_arima", "ets"]
        neural = [m for m in smapes if m not in classical]

        best_cl = min((smapes[m] for m in classical if m in smapes), default=999)
        best_nn = min((smapes[m] for m in neural if m in smapes), default=999)

        if best_nn < best_cl:
            print(f"  >>> NEURAL WINS: best_nn={best_nn:.2f} vs best_cl={best_cl:.2f} "
                  f"(gap={best_cl - best_nn:.2f} pp)")
        else:
            print(f"  Classical wins: best_cl={best_cl:.2f} vs best_nn={best_nn:.2f}")

    print(f"\n\nAll plots saved to {FIG_DIR}/")


if __name__ == "__main__":
    main()
