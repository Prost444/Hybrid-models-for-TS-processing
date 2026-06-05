#!/usr/bin/env python3
"""Defense-grade forecast charts (predзащита fixes).

Fixes addressed:
  * forecast now CONNECTS to the last training point — no 1-step gap;
  * explicit annotated boundary "конец обучения → прогноз";
  * large, readable legend (also duplicated as text on the slide);
  * one dedicated large all-pretrained chart for the standalone forecast slide.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "src" / "outputs" / "thesis_pretrain_forecasts"
FIG = ROOT / "thesis" / "figures"

NAVY = "#1B2649"
ACCENT = "#E07A5F"
GREY = "#9AA3B2"
BLACK = "#16181D"

LABELS = {
    "dlinear": "DLinear", "autoformer": "Autoformer", "fedformer": "FEDformer",
    "patchtst": "PatchTST", "nbeats": "N-BEATS", "timesnet": "TimesNet",
    "helformer": "Helformer", "seasonal_naive": "Seasonal Naive",
    "auto_arima": "Auto-ARIMA", "ets": "ETS",
}
COLORS = {
    "dlinear": "#3498DB", "autoformer": "#E67E22", "fedformer": "#16A085",
    "patchtst": "#27AE60", "nbeats": "#8E44AD", "timesnet": "#9B59B6",
    "helformer": "#E91E63", "seasonal_naive": "#95A5A6",
    "auto_arima": "#C0392B", "ets": "#D4881A",
}
CLASSICAL = ["seasonal_naive", "auto_arima", "ets"]

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.grid": True,
                     "grid.alpha": 0.25})


def smape(a, p):
    return 200 * np.mean(np.abs(a - p) / (np.abs(a) + np.abs(p) + 1e-8))


def _boundary(ax, x0, ytop):
    """Mark train/forecast boundary clearly."""
    ax.axvline(x=x0, color=NAVY, linestyle="--", linewidth=1.3, alpha=0.7)
    ax.annotate("конец обучающей выборки → прогноз",
                xy=(x0, ytop), xytext=(6, -4), textcoords="offset points",
                fontsize=10, color=NAVY, fontweight="bold",
                ha="left", va="top", rotation=0)


def plot_pair_fixed(sid, model, tail=70):
    """Scratch vs pretrained for one model — forecast connected to train end."""
    tr = pd.read_csv(DATA / f"{sid}_train.csv")["value"].values
    pair = pd.read_csv(DATA / f"{sid}_{model}_pair.csv")
    H = len(pair)
    actual = pair["actual"].values
    N = len(tr)

    # x grids; forecast lines start ONE step early (at N-1) with train[-1]
    t_train = np.arange(N - tail, N)
    t_fc = np.arange(N - 1, N + H)          # includes the bridge point N-1
    bridge = tr[-1]

    fig, ax = plt.subplots(figsize=(11.6, 6.2))
    ax.plot(t_train, tr[-tail:], color=GREY, lw=1.6, label="обучающая история")
    ax.plot(np.concatenate([[N - 1], np.arange(N, N + H)]),
            np.concatenate([[bridge], actual]),
            color=BLACK, lw=3.0, label="факт", zorder=10)

    color = COLORS.get(model, "#666")
    name = LABELS.get(model, model)
    sc = pair[f"{model}_scratch"].values
    pr = pair[f"{model}_pretrain"].values
    ax.plot(t_fc, np.concatenate([[bridge], sc]), color=color, lw=2.2,
            ls="--", alpha=0.85,
            label=f"{name} с нуля — sMAPE {smape(actual, sc):.1f}%")
    ax.plot(t_fc, np.concatenate([[bridge], pr]), color=color, lw=3.0,
            ls="-", label=f"{name} предобучен — sMAPE {smape(actual, pr):.1f}%")

    ymax = max(tr[-tail:].max(), actual.max())
    ymin = min(tr[-tail:].min(), actual.min())
    _boundary(ax, N - 1, ymax)
    ax.set_xlabel("Временной шаг", fontsize=12)
    ax.set_ylabel("Значение ряда", fontsize=12)
    ax.set_title(f"{name}: эффект предобучения на ряде {sid} (N={N}, H={H})",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower left", fontsize=11.5, framealpha=0.95)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = FIG / f"pretrain_forecast_{model}_{sid}.png"
    fig.savefig(out, dpi=170, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved", out.name)


def plot_large_forecast(sid, tail=60, data_dir=DATA, pred_name="all_pretrained",
                        out_name=None, title=None):
    """One big clean chart: all pretrained neural + classical, connected."""
    tr = pd.read_csv(data_dir / f"{sid}_train.csv")["value"].values
    df = pd.read_csv(data_dir / f"{sid}_{pred_name}.csv")
    H = len(df)
    actual = df["actual"].values
    N = len(tr)
    bridge = tr[-1]

    models = [c for c in df.columns if c not in ("step", "actual")
              and not df[c].isna().any()]
    sm = {m: smape(actual, df[m].values) for m in models}
    order = sorted(models, key=lambda m: sm[m])

    fig, ax = plt.subplots(figsize=(12.8, 6.6))
    t_train = np.arange(N - tail, N)
    ax.plot(t_train, tr[-tail:], color=GREY, lw=1.6, label="обучающая история")
    xb = np.concatenate([[N - 1], np.arange(N, N + H)])
    ax.plot(xb, np.concatenate([[bridge], actual]), color=BLACK, lw=3.2,
            label="факт", zorder=12)

    for rank, m in enumerate(order):
        y = np.concatenate([[bridge], df[m].values])
        is_cl = m in CLASSICAL
        ax.plot(xb, y, color=COLORS.get(m, "#666"),
                lw=2.6 if rank < 3 else 1.4,
                ls="--" if is_cl else "-",
                alpha=0.95 if rank < 3 else 0.55,
                zorder=8 if rank < 3 else 4)

    ymax = max(tr[-tail:].max(), actual.max())
    _boundary(ax, N - 1, ymax)
    ax.set_xlabel("Временной шаг", fontsize=12.5)
    ax.set_ylabel("Значение ряда", fontsize=12.5)
    ax.set_title(title or (f"Прогноз ежедневного ряда {sid}: предобученные "
                 f"нейросети против классики (N={N}, H={H})"),
                 fontsize=13.5, fontweight="bold")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = FIG / (out_name or f"forecast_large_{sid}.png")
    fig.savefig(out, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close()
    # also return ranking text for the slide legend-as-text
    print("saved", out.name, "| ranking:",
          ", ".join(f"{LABELS[m]} {sm[m]:.1f}" for m in order[:6]))


THESIS_FC = ROOT / "src" / "outputs" / "thesis_forecasts"


def plot_pnl():
    """3-year P&L bar chart for the economic slide (shared startup «ГМ-ПВР»)."""
    years = ["Год 1", "Год 2", "Год 3"]
    revenue = [8550, 35000, 63800]
    opex = [2605.1, 6310.2, 10515.3]
    profit = [5345.9, 26239.8, 48818.7]
    x = np.arange(3)
    w = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 6.0))
    ax.bar(x - w, revenue, w, label="Выручка", color=NAVY)
    ax.bar(x, opex, w, label="Опер. расходы", color=ACCENT)
    ax.bar(x + w, profit, w, label="Чистая прибыль", color="#2E7D6B")
    ax.set_xticks(x)
    ax.set_xticklabels(years, fontsize=13)
    ax.set_ylabel("тыс. ₽", fontsize=12)
    ax.set_title("Прогноз выручки и чистой прибыли (P&L), тыс. ₽",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=12, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    plt.tight_layout()
    out = FIG / "econ_pnl.png"
    fig.savefig(out, dpi=175, bbox_inches="tight", facecolor="white")
    plt.close()
    print("saved", out.name)


if __name__ == "__main__":
    plot_pnl()
    plot_pair_fixed("D2111", "autoformer")
    plot_pair_fixed("D2111", "patchtst")
    plot_large_forecast("D2111")
    # "анализ отдельных рядов" — clean connected version of the cherry chart
    plot_large_forecast(
        "D2052", tail=60, data_dir=THESIS_FC, pred_name="predictions",
        out_name="forecast_cherry_D2052.png",
        title="Ряд D2052 (M4 daily): нейросети точнее классики (N=174, H=14)")
