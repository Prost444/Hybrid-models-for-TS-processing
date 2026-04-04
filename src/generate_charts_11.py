#!/usr/bin/env python3
"""Generate the 5 main benchmark charts with 11 models (including Helformer).

Charts produced in src/outputs/visualizations/:
  1. heatmap_smape.png   -- Cross-dataset sMAPE heatmap (11 models x 6 categories)
  2. bar_m3_smape.png    -- M3 overall sMAPE bar chart (11 models)
  3. bar_m3_by_freq.png  -- M3 by frequency (3 panels, 11 models each)
  4. bar_m4_daily_full.png -- M4 daily sMAPE bar chart (11 models)
  5. heatmap_rank.png    -- Mean rank heatmap (11 models x 6 categories)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

# ── paths ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "outputs" / "visualizations"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── load CSVs ──────────────────────────────────────────────────────────
m3 = pd.read_csv(ROOT / "outputs" / "m3_benchmark_v3" / "metrics.csv")
m4qm = pd.read_csv(ROOT / "outputs" / "m4_benchmark_qm" / "metrics.csv")
m4d = pd.read_csv(ROOT / "outputs" / "m4_daily_merged.csv")
hm3 = pd.read_csv(ROOT / "outputs" / "helformer_m3_benchmark" / "metrics.csv")
hm4 = pd.read_csv(ROOT / "outputs" / "helformer_m4_benchmark" / "metrics.csv")

# ── model definitions ──────────────────────────────────────────────────
MODELS = [
    "seasonal_naive", "auto_arima", "ets", "prophet",
    "dlinear", "autoformer", "fedformer", "nbeats",
    "patchtst", "timesnet", "helformer",
]

MODEL_LABELS = [
    "Сезон.\nнаивн.", "auto-\nARIMA", "ETS", "Prophet",
    "DLinear", "Auto-\nformer", "FED-\nformer", "N-BEATS",
    "PatchTST", "TimesNet", "Hel-\nformer",
]

COLORS = {
    "seasonal_naive": "#4e79a7",
    "auto_arima":     "#4e79a7",
    "ets":            "#4e79a7",
    "prophet":        "#4e79a7",
    "dlinear":        "#e15759",
    "autoformer":     "#e15759",
    "fedformer":      "#e15759",
    "nbeats":         "#e15759",
    "patchtst":       "#59a14f",
    "timesnet":       "#59a14f",
    "helformer":      "#b07aa1",
}

BAR_COLORS = [COLORS[m] for m in MODELS]

CATEGORIES_M3 = ["yearly", "quarterly", "monthly"]
CATEGORIES_M4_QM = ["quarterly", "monthly"]
CATEGORIES_M4_D = ["daily"]
ALL_CATEGORIES = [
    ("M3 yearly", "m3", "yearly"),
    ("M3 quarterly", "m3", "quarterly"),
    ("M3 monthly", "m3", "monthly"),
    ("M4 quarterly", "m4", "quarterly"),
    ("M4 monthly", "m4", "monthly"),
    ("M4 daily", "m4", "daily"),
]

CAT_LABELS_SHORT = [
    "M3\nyearly", "M3\nquarterly", "M3\nmonthly",
    "M4\nquarterly", "M4\nmonthly", "M4\ndaily",
]


# ── merge Helformer into main tables ──────────────────────────────────
def _merge_helformer(main_df: pd.DataFrame, helf_df: pd.DataFrame) -> pd.DataFrame:
    """Add helformer_sMAPE column to main_df by joining on (category, series_id)."""
    helf = helf_df[["category", "series_id", "helformer_sMAPE"]].copy()
    helf = helf.dropna(subset=["helformer_sMAPE"])
    merged = main_df.merge(helf, on=["category", "series_id"], how="left")
    return merged


m3_full = _merge_helformer(m3, hm3)
m4qm_full = _merge_helformer(m4qm, hm4)
m4d_full = _merge_helformer(m4d, hm4)


# ── helper: compute mean sMAPE per model per category ─────────────────
def _smape_col(model: str) -> str:
    return f"{model}_sMAPE"


def mean_smape_table() -> pd.DataFrame:
    """Return DataFrame with columns = models, rows = 6 categories."""
    rows = []
    for label, ds, cat in ALL_CATEGORIES:
        if ds == "m3":
            df = m3_full[m3_full["category"] == cat]
        elif ds == "m4" and cat == "daily":
            df = m4d_full[m4d_full["category"] == cat]
        else:
            df = m4qm_full[m4qm_full["category"] == cat]

        row = {}
        for m in MODELS:
            col = _smape_col(m)
            if col in df.columns:
                vals = df[col].dropna()
                row[m] = vals.mean() if len(vals) > 0 else np.nan
            else:
                row[m] = np.nan
        rows.append(row)

    return pd.DataFrame(rows, index=[c[0] for c in ALL_CATEGORIES])


def rank_table(smape_tbl: pd.DataFrame) -> pd.DataFrame:
    """Rank models per category (1 = best)."""
    return smape_tbl.rank(axis=1, method="min")


# ── compute tables ────────────────────────────────────────────────────
smape_tbl = mean_smape_table()
rank_tbl = rank_table(smape_tbl)

print("=== Mean sMAPE table ===")
print(smape_tbl.round(2).to_string())
print()
print("=== Rank table ===")
print(rank_tbl.round(1).to_string())
print()

# ── Style helpers ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
})
DPI = 150
FIGW_CM = 16
FIGW = FIGW_CM / 2.54  # inches


# ── Chart 1: heatmap_smape ────────────────────────────────────────────
def chart_heatmap_smape():
    data = smape_tbl.values  # shape (6, 11)
    fig, ax = plt.subplots(figsize=(FIGW * 1.35, FIGW * 0.55))

    cmap = LinearSegmentedColormap.from_list("gyr", ["#2ca02c", "#ffffcc", "#d62728"], N=256)
    vmin_val = np.nanmin(data)
    vmax_val = np.nanpercentile(data[np.isfinite(data)], 95) if np.any(np.isfinite(data)) else 100
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin_val, vmax=vmax_val)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_LABELS, fontsize=7, ha="center")
    ax.set_yticks(range(len(CAT_LABELS_SHORT)))
    ax.set_yticklabels(CAT_LABELS_SHORT, fontsize=7)

    # Annotate cells; bold the best (min) per row
    for i in range(data.shape[0]):
        row = data[i]
        if np.all(np.isnan(row)):
            best_j = -1
        else:
            best_j = int(np.nanargmin(row))
        for j in range(data.shape[1]):
            val = row[j]
            if np.isnan(val):
                txt = "--"
                weight = "normal"
            else:
                txt = f"{val:.1f}"
                weight = "bold" if j == best_j else "normal"
            color = "white" if (not np.isnan(val) and val > (vmax_val * 0.7)) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, fontweight=weight, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("sMAPE", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("sMAPE по моделям и категориям", fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "heatmap_smape.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Chart 2: bar_m3_smape ────────────────────────────────────────────
def chart_bar_m3_smape():
    # Overall M3 mean sMAPE across 3 categories
    m3_cats = smape_tbl.loc[["M3 yearly", "M3 quarterly", "M3 monthly"]]
    overall = m3_cats.mean(axis=0)

    fig, ax = plt.subplots(figsize=(FIGW * 1.1, FIGW * 0.5))
    xs = np.arange(len(MODELS))
    bars = ax.bar(xs, overall.values, color=BAR_COLORS, edgecolor="white", linewidth=0.5, width=0.7)

    # Annotate
    for bar_obj, val in zip(bars, overall.values):
        if np.isnan(val):
            txt = "--"
        else:
            txt = f"{val:.1f}"
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2, bar_obj.get_height() + 0.5,
                txt, ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(MODEL_LABELS, fontsize=7)
    ax.set_ylabel("sMAPE")
    ax.set_title("M3: среднее sMAPE по трём категориям (90 рядов)", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    path = OUT_DIR / "bar_m3_smape.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Chart 3: bar_m3_by_freq ──────────────────────────────────────────
def chart_bar_m3_by_freq():
    fig, axes = plt.subplots(1, 3, figsize=(FIGW * 1.65, FIGW * 0.52), sharey=False)
    freq_names = {"yearly": "Yearly (30 рядов)", "quarterly": "Quarterly (30 рядов)", "monthly": "Monthly (30 рядов)"}

    for idx, cat in enumerate(CATEGORIES_M3):
        ax = axes[idx]
        row_label = f"M3 {cat}"
        vals = smape_tbl.loc[row_label].values
        xs = np.arange(len(MODELS))
        bars = ax.bar(xs, vals, color=BAR_COLORS, edgecolor="white", linewidth=0.4, width=0.75)

        for bar_obj, v in zip(bars, vals):
            if np.isnan(v):
                txt = "--"
            else:
                txt = f"{v:.1f}"
            ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                    bar_obj.get_height() + 0.3, txt,
                    ha="center", va="bottom", fontsize=5.5)

        ax.set_xticks(xs)
        ax.set_xticklabels(MODEL_LABELS, fontsize=5.5)
        ax.set_title(freq_names[cat], fontsize=8)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        if idx == 0:
            ax.set_ylabel("sMAPE", fontsize=8)

    fig.suptitle("M3: sMAPE по частотам", fontsize=10, y=1.02)
    fig.tight_layout()
    path = OUT_DIR / "bar_m3_by_freq.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Chart 4: bar_m4_daily_full ───────────────────────────────────────
def chart_bar_m4_daily():
    vals = smape_tbl.loc["M4 daily"].values
    fig, ax = plt.subplots(figsize=(FIGW * 1.1, FIGW * 0.5))
    xs = np.arange(len(MODELS))
    bars = ax.bar(xs, vals, color=BAR_COLORS, edgecolor="white", linewidth=0.5, width=0.7)

    for bar_obj, v in zip(bars, vals):
        if np.isnan(v):
            txt = "--"
        else:
            txt = f"{v:.1f}"
        ax.text(bar_obj.get_x() + bar_obj.get_width() / 2,
                bar_obj.get_height() + 0.03, txt,
                ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(xs)
    ax.set_xticklabels(MODEL_LABELS, fontsize=7)
    ax.set_ylabel("sMAPE")
    ax.set_title("M4 Daily: sMAPE (30 рядов)", fontsize=9)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    path = OUT_DIR / "bar_m4_daily_full.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Chart 5: heatmap_rank ────────────────────────────────────────────
def chart_heatmap_rank():
    data = rank_tbl.values  # (6, 11)
    fig, ax = plt.subplots(figsize=(FIGW * 1.35, FIGW * 0.55))

    cmap = LinearSegmentedColormap.from_list("rank_cmap", ["#2ca02c", "#ffffcc", "#d62728"], N=256)
    vmin_val = 1
    vmax_val = len(MODELS)
    im = ax.imshow(data, cmap=cmap, aspect="auto", vmin=vmin_val, vmax=vmax_val)

    ax.set_xticks(range(len(MODELS)))
    ax.set_xticklabels(MODEL_LABELS, fontsize=7, ha="center")
    ax.set_yticks(range(len(CAT_LABELS_SHORT)))
    ax.set_yticklabels(CAT_LABELS_SHORT, fontsize=7)

    for i in range(data.shape[0]):
        row = data[i]
        if np.all(np.isnan(row)):
            best_j = -1
        else:
            best_j = int(np.nanargmin(row))
        for j in range(data.shape[1]):
            val = row[j]
            if np.isnan(val):
                txt = "--"
                weight = "normal"
            else:
                txt = f"{val:.0f}"
                weight = "bold" if j == best_j else "normal"
            color = "white" if (not np.isnan(val) and val > vmax_val * 0.7) else "black"
            ax.text(j, i, txt, ha="center", va="center", fontsize=6.5, fontweight=weight, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.04)
    cbar.set_label("Ранг", fontsize=8)
    cbar.ax.tick_params(labelsize=7)

    ax.set_title("Средний ранг по категориям", fontsize=10)
    fig.tight_layout()
    path = OUT_DIR / "heatmap_rank.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    chart_heatmap_smape()
    chart_bar_m3_smape()
    chart_bar_m3_by_freq()
    chart_bar_m4_daily()
    chart_heatmap_rank()
    print("\nAll 5 charts generated.")
