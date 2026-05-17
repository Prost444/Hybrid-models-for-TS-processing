#!/usr/bin/env python3
"""Compute aggregated M4 benchmark statistics for thesis tables."""

import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("outputs/parallel_m4_1000/metrics_checkpoint.csv")

models = [
    "seasonal_naive", "auto_arima", "ets", "prophet",
    "dlinear", "autoformer", "fedformer", "patchtst",
    "nbeats", "timesnet", "helformer"
]
model_labels = {
    "seasonal_naive": "Seasonal Naive",
    "auto_arima": "Auto-ARIMA",
    "ets": "ETS",
    "prophet": "Prophet",
    "dlinear": "DLinear",
    "autoformer": "Autoformer",
    "fedformer": "FEDformer",
    "patchtst": "PatchTST",
    "nbeats": "N-BEATS",
    "timesnet": "TimesNet",
    "helformer": "Helformer",
}

categories = ["quarterly", "monthly", "daily"]

print("=" * 80)
print("M4 BENCHMARK RESULTS — PARALLEL_M4_1000")
print("=" * 80)

# Count series per category
for cat in categories:
    sub = df[df["category"] == cat]
    print(f"\n{cat}: {len(sub)} series")
    for m in models:
        col = f"{m}_sMAPE"
        if col in sub.columns:
            valid = sub[col].dropna()
            print(f"  {m}: {len(valid)} valid sMAPE values")

# ============================================================
# TABLE 1: M4 Q+M sMAPE (like tab:m4qm-smape)
# ============================================================
print("\n" + "=" * 80)
print("TABLE: M4 Quarterly + Monthly sMAPE (%)")
print("=" * 80)

results_qm = {}
for m in models:
    results_qm[m] = {}
    for cat in ["quarterly", "monthly"]:
        col = f"{m}_sMAPE"
        sub = df[df["category"] == cat]
        val = sub[col].dropna().mean()
        results_qm[m][cat] = val
    results_qm[m]["overall"] = np.mean([results_qm[m]["quarterly"], results_qm[m]["monthly"]])

# Sort by overall
sorted_models = sorted(results_qm.keys(), key=lambda m: results_qm[m]["overall"])

print(f"\n{'Model':<20} {'Quarterly':>12} {'Monthly':>12} {'Overall':>12}")
print("-" * 60)
for m in sorted_models:
    print(f"{model_labels[m]:<20} {results_qm[m]['quarterly']:>12.2f} {results_qm[m]['monthly']:>12.2f} {results_qm[m]['overall']:>12.2f}")

# ============================================================
# TABLE 2: M4 Daily sMAPE + MASE (like tab:m4d-smape)
# ============================================================
print("\n" + "=" * 80)
print("TABLE: M4 Daily sMAPE + MASE")
print("=" * 80)

daily = df[df["category"] == "daily"]
results_d = {}
for m in models:
    smape_col = f"{m}_sMAPE"
    mase_col = f"{m}_MASE"
    smape = daily[smape_col].dropna()
    mase = daily[mase_col].dropna()
    results_d[m] = {
        "smape": smape.mean() if len(smape) > 0 else None,
        "mase": mase.mean() if len(mase) > 0 else None,
        "n": len(smape),
    }

sorted_daily = sorted([m for m in models if results_d[m]["smape"] is not None],
                       key=lambda m: results_d[m]["smape"])

print(f"\n{'Model':<20} {'sMAPE':>10} {'MASE':>10} {'N':>6}")
print("-" * 50)
for m in sorted_daily:
    r = results_d[m]
    mase_str = f"{r['mase']:.2f}" if r['mase'] is not None else "---"
    print(f"{model_labels[m]:<20} {r['smape']:>10.2f} {mase_str:>10} {r['n']:>6}")

# ============================================================
# TABLE 3: Detailed per-category (sMAPE, MASE, RMSE)
# ============================================================
for cat in categories:
    print(f"\n{'=' * 80}")
    print(f"TABLE: M4 {cat.title()} Detail (sMAPE, MASE, RMSE)")
    print(f"{'=' * 80}")

    sub = df[df["category"] == cat]
    detail = {}
    for m in models:
        s = sub[f"{m}_sMAPE"].dropna()
        ma = sub[f"{m}_MASE"].dropna()
        r = sub[f"{m}_RMSE"].dropna()
        if len(s) > 0:
            detail[m] = {
                "smape": s.mean(),
                "mase": ma.mean() if len(ma) > 0 else None,
                "rmse": r.mean() if len(r) > 0 else None,
                "n": len(s),
            }

    sorted_m = sorted(detail.keys(), key=lambda m: detail[m]["smape"])

    print(f"\n{'Model':<20} {'sMAPE':>10} {'MASE':>10} {'RMSE':>12} {'N':>6}")
    print("-" * 65)
    for m in sorted_m:
        r = detail[m]
        mase_str = f"{r['mase']:.2f}" if r['mase'] is not None else "---"
        rmse_str = f"{r['rmse']:.2f}" if r['rmse'] is not None else "---"
        print(f"{model_labels[m]:<20} {r['smape']:>10.2f} {mase_str:>10} {rmse_str:>12} {r['n']:>6}")

# ============================================================
# TABLE 4: Cross-dataset (for M4 columns only)
# ============================================================
print(f"\n{'=' * 80}")
print(f"M4 COLUMNS for cross-dataset table")
print(f"{'=' * 80}")

print(f"\n{'Model':<20} {'M4-Q':>10} {'M4-M':>10} {'M4-D':>10}")
print("-" * 55)
for m in sorted_models:
    m4q = results_qm[m]["quarterly"]
    m4m = results_qm[m]["monthly"]
    m4d = results_d[m]["smape"] if results_d[m]["smape"] is not None else float('nan')
    print(f"{model_labels[m]:<20} {m4q:>10.2f} {m4m:>10.2f} {m4d:>10.2f}")

# ============================================================
# TABLE 5: Mean rank across all 6 categories (M3 + M4)
# ============================================================
print(f"\n{'=' * 80}")
print(f"RANK TABLE (M4 only, for reference)")
print(f"{'=' * 80}")

# Build sMAPE by category for ranking
all_cats = {}
for cat in categories:
    sub = df[df["category"] == cat]
    cat_smape = {}
    for m in models:
        vals = sub[f"{m}_sMAPE"].dropna()
        if len(vals) > 10:  # only rank models with enough data
            cat_smape[m] = vals.mean()
    # Sort and assign ranks
    ranked = sorted(cat_smape.keys(), key=lambda x: cat_smape[x])
    for rank, m in enumerate(ranked, 1):
        if m not in all_cats:
            all_cats[m] = {}
        all_cats[m][cat] = rank

print(f"\n{'Model':<20} {'M4-Q rank':>10} {'M4-M rank':>10} {'M4-D rank':>10} {'Mean':>8}")
print("-" * 65)
for m in models:
    if m in all_cats:
        ranks = list(all_cats[m].values())
        mean_r = np.mean(ranks)
        q = all_cats[m].get("quarterly", "---")
        mo = all_cats[m].get("monthly", "---")
        d = all_cats[m].get("daily", "---")
        print(f"{model_labels[m]:<20} {q:>10} {mo:>10} {d:>10} {mean_r:>8.2f}")
