#!/usr/bin/env python3
"""Plot pretrained vs from-scratch forecast comparisons for thesis.

Two types of plots:
1. Per-model pair: train tail + actual + scratch prediction + pretrained prediction
2. All-pretrained cherry-pick: train tail + actual + classical + all pretrained neural

Usage:
    python plot_pretrain_forecasts.py
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

PRED_DIR = Path("outputs/thesis_pretrain_forecasts")
FIG_DIR = Path("../thesis/figures")

LABELS = {
    "dlinear": "DLinear", "autoformer": "Autoformer", "fedformer": "FEDformer",
    "patchtst": "PatchTST", "nbeats": "N-BEATS", "timesnet": "TimesNet",
    "helformer": "Helformer", "seasonal_naive": "S. Naive",
    "auto_arima": "ARIMA", "ets": "ETS",
}
COLORS = {
    "dlinear": "#3498DB", "autoformer": "#E67E22", "fedformer": "#D35400",
    "patchtst": "#2ECC71", "nbeats": "#1ABC9C", "timesnet": "#9B59B6",
    "helformer": "#E91E63", "seasonal_naive": "#95A5A6",
    "auto_arima": "#E74C3C", "ets": "#F39C12",
}

plt.rcParams.update({'font.size': 12, 'figure.dpi': 150, 'axes.grid': True, 'grid.alpha': 0.3})


def smape(actual, pred):
    return 200 * np.mean(np.abs(actual - pred) / (np.abs(actual) + np.abs(pred) + 1e-8))


def plot_pretrain_pair(sid, model_name, train_df, pair_df, fig_dir, tail_len=100):
    """Plot scratch vs pretrained prediction for one model."""
    train = train_df["value"].values
    H = len(pair_df)
    actual = pair_df["actual"].values

    scratch_col = f"{model_name}_scratch"
    pretrain_col = f"{model_name}_pretrain"

    fig, ax = plt.subplots(figsize=(14, 5))

    # Train tail
    tail = train[-tail_len:]
    t_train = np.arange(len(train) - tail_len, len(train))
    ax.plot(t_train, tail, color='#BDC3C7', linewidth=1, label='train')

    # Actual
    t_test = np.arange(len(train), len(train) + H)
    ax.plot(t_test, actual, 'k-', linewidth=2.5, label='actual', zorder=10)

    # Vertical line
    ax.axvline(x=len(train), color='black', linestyle=':', alpha=0.3)

    color = COLORS.get(model_name, '#666')
    label_name = LABELS.get(model_name, model_name)

    if scratch_col in pair_df.columns:
        scratch_pred = pair_df[scratch_col].values
        s_smape = smape(actual, scratch_pred)
        ax.plot(t_test, scratch_pred, color=color, linewidth=2, linestyle='--',
                alpha=0.7, label=f'{label_name} с нуля ({s_smape:.1f}%)')

    if pretrain_col in pair_df.columns:
        pretrain_pred = pair_df[pretrain_col].values
        p_smape = smape(actual, pretrain_pred)
        ax.plot(t_test, pretrain_pred, color=color, linewidth=2.5, linestyle='-',
                label=f'{label_name} предобучен ({p_smape:.1f}%)')

    ax.set_xlabel('Временной шаг')
    ax.set_ylabel('Значение')
    ax.set_title(f'{label_name}: предобученный vs с нуля на ряде {sid} '
                 f'(N={len(train)}, H={H})')
    ax.legend(loc='best', fontsize=10, framealpha=0.9)
    plt.tight_layout()

    fig_path = fig_dir / f"pretrain_forecast_{model_name}_{sid}.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"  saved {fig_path.name}")


def plot_all_pretrained(sid, train_df, pred_df, fig_dir, tail_len=100):
    """Plot all 11 models where neural ones are pretrained."""
    train = train_df["value"].values
    H = len(pred_df)
    actual = pred_df["actual"].values

    models = [c for c in pred_df.columns if c not in ["step", "actual"]]
    model_smapes = {}
    for m in models:
        pred = pred_df[m].values
        if not np.any(np.isnan(pred)):
            model_smapes[m] = smape(actual, pred)

    sorted_models = sorted(model_smapes.keys(), key=lambda m: model_smapes[m])
    classical = ["seasonal_naive", "auto_arima", "ets"]

    fig, ax = plt.subplots(figsize=(14, 6))

    # Train tail
    tail = train[-tail_len:]
    t_train = np.arange(len(train) - tail_len, len(train))
    ax.plot(t_train, tail, color='#BDC3C7', linewidth=1, label='train')

    # Actual
    t_test = np.arange(len(train), len(train) + H)
    ax.plot(t_test, actual, 'k-', linewidth=2.5, label='actual', zorder=10)
    ax.axvline(x=len(train), color='black', linestyle=':', alpha=0.3)

    for rank, m in enumerate(sorted_models):
        pred = pred_df[m].values
        color = COLORS.get(m, '#666')
        label_name = LABELS.get(m, m)
        is_classical = m in classical
        suffix = "" if is_classical else " (pretrained)"
        label = f"{label_name}{suffix} ({model_smapes[m]:.1f}%)"
        lw = 2.0 if rank < 3 else 1.0
        ls = '--' if is_classical else '-'
        alpha = 0.9 if rank < 3 else 0.5
        ax.plot(t_test, pred, color=color, linewidth=lw, linestyle=ls,
                label=label, alpha=alpha)

    ax.set_xlabel('Временной шаг')
    ax.set_ylabel('Значение')
    ax.set_title(f'Ряд {sid}: предобученные нейросети vs классические модели '
                 f'(N={len(train)}, H={H})')
    ax.legend(loc='best', fontsize=7, framealpha=0.9, ncol=2)
    plt.tight_layout()

    fig_path = fig_dir / f"cherry_pretrained_{sid}.png"
    plt.savefig(fig_path, bbox_inches='tight')
    plt.close()
    print(f"  saved {fig_path.name}")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Per-model pairs
    print("=== Per-model pretrain pairs ===")
    pair_files = sorted(PRED_DIR.glob("*_pair.csv"))
    for pf in pair_files:
        parts = pf.stem.split("_")
        # e.g. D2111_dlinear_pair
        sid = parts[0]
        model_name = parts[1]

        train_file = PRED_DIR / f"{sid}_train.csv"
        if not train_file.exists():
            continue

        train_df = pd.read_csv(train_file)
        pair_df = pd.read_csv(pf)
        plot_pretrain_pair(sid, model_name, train_df, pair_df, FIG_DIR)

    # 2. All-pretrained cherry-picks
    print("\n=== All-pretrained cherry-picks ===")
    cherry_files = sorted(PRED_DIR.glob("*_all_pretrained.csv"))
    for cf in cherry_files:
        sid = cf.stem.replace("_all_pretrained", "")
        train_file = PRED_DIR / f"{sid}_train.csv"
        if not train_file.exists():
            continue

        train_df = pd.read_csv(train_file)
        pred_df = pd.read_csv(cf)
        plot_all_pretrained(sid, train_df, pred_df, FIG_DIR)

    print("\nAll plots saved!")


if __name__ == "__main__":
    main()
