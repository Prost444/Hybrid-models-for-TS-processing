#!/usr/bin/env python3
"""Generate forecast predictions (saved to CSV) and plots for ALL 11 models.

Covers:
  1. Standard forecast plots (3 series × 6 categories = 18 plots)
  2. Rollout comparison plots (direct vs rollout for neural models)
  3. Pretrain comparison plots (pretrain vs from-scratch)
  4. Synthetic series benchmark (6 profiles × 11 models)

All predictions saved to CSV for reproducibility. Plots built from CSVs.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from project_paths import ensure_src_on_path
ensure_src_on_path()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from hybridts.config.settings import settings
from hybridts.data import (
    M3_H, M3_P, M4_H, M4_P, best_L,
    ensure_m3_csv, load_train_tsts, ensure_m4_csv, load_m4_train_test,
    seasonal_naive, smape, mape, mse, rmse, mae, mase,
)
from hybridts.models import make_model, auto_arima_forecast, ets_forecast, prophet_forecast
from hybridts.models.helformer_pt import HelformerPT, helformer_forecast_pt
from hybridts.training import TrainConfig, WindowDatasetStd, train_model

DEVICE = "cpu"
EPOCHS = 50
HF_EPOCHS = 40
OUT_DIR = Path("src/outputs/full_forecasts")

NEURAL_MODELS = ["dlinear", "autoformer", "fedformer", "patchtst", "nbeats", "timesnet"]
FREQ_MAP = {"yearly": "YE", "quarterly": "QE", "monthly": "ME",
            "weekly": "W", "daily": "D", "hourly": "h"}


def train_and_predict(model_name, y_tr, H, P, epochs, device):
    L = best_L(y_tr, H, P)
    cfg = TrainConfig(lookback=L, horizon=H, epochs=epochs, batch_size=32,
                      lr=1e-3, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model(model_name, cfg)
    ds = WindowDatasetStd(y_tr, L, H)
    if len(ds) < 1:
        return None, None
    train_model(model, ds, cfg)
    mu, sd = ds.scaler
    model.eval()
    with torch.no_grad():
        x = torch.tensor(y_tr[-L:], dtype=torch.float32)
        x_norm = (x - mu) / sd
        x_in = x_norm.unsqueeze(0).unsqueeze(0).to(cfg.device)
        pred_norm = model(x_in).squeeze(0).cpu().numpy()
    pred = pred_norm * sd + mu
    return np.asarray(pred, float).ravel()[:H], model


def rollout_predict(model, y_tr, L, H, mu, sd, device):
    model.eval()
    history = list(y_tr)
    preds = []
    for _ in range(H):
        x = np.array(history[-L:], dtype=np.float32)
        x_norm = (x - mu) / sd
        x_t = torch.tensor(x_norm).unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            out = model(x_t).squeeze(0).cpu().numpy()
        val = float(out[0]) * sd + mu
        if not np.isfinite(val):
            val = history[-1]
        preds.append(val)
        history.append(val)
    return np.array(preds)


def predict_helformer(y_tr, H, P, epochs, device):
    L = min(30, len(y_tr) - 2)
    if L < 4:
        L = max(4, len(y_tr) // 2)
    cfg = TrainConfig(lookback=L, horizon=1, epochs=epochs, batch_size=32,
                      lr=8e-4, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model("helformer", cfg)
    # MinMax normalization
    vmin = float(np.nanmin(y_tr))
    scale = float(np.nanmax(y_tr)) - vmin
    if scale < 1e-8:
        scale = 1.0
    scaled = (y_tr - vmin) / scale
    ds = WindowDatasetStd(scaled, L, 1, scale=False)
    if len(ds) < 1:
        return None
    train_model(model, ds, cfg)
    pred = helformer_forecast_pt(y_tr, H, model, lookback=L,
                                  seasonal_period=P, use_hw=True, device=device)
    return np.asarray(pred, float).ravel()[:H]


def forecast_all_models(y_tr, y_te, H, P, cat, device, epochs):
    forecasts = {}
    # Classical
    try:
        forecasts["seasonal_naive"] = seasonal_naive(y_tr, H, P)
    except Exception:
        pass
    try:
        forecasts["auto_arima"] = np.asarray(auto_arima_forecast(y_tr, H), float)
    except Exception:
        pass
    try:
        forecasts["ets"] = np.asarray(ets_forecast(y_tr, H, seasonal_periods=P), float)
    except Exception:
        pass
    try:
        freq = FREQ_MAP.get(cat, "D")
        forecasts["prophet"] = np.asarray(prophet_forecast(y_tr, H, freq=freq), float)
    except Exception:
        pass
    # Neural
    trained_models = {}
    for nm in NEURAL_MODELS:
        try:
            pred, model = train_and_predict(nm, y_tr, H, P, epochs, device)
            if pred is not None:
                forecasts[nm] = pred
                trained_models[nm] = (model, best_L(y_tr, H, P))
        except Exception as e:
            print(f"  {nm} failed: {e}")
    # Helformer
    try:
        hf = predict_helformer(y_tr, H, P, HF_EPOCHS, device)
        if hf is not None:
            forecasts["helformer"] = hf
    except Exception as e:
        print(f"  helformer failed: {e}")

    return forecasts, trained_models


def save_forecasts_csv(records, path):
    df = pd.DataFrame(records)
    df.to_csv(path, index=False)
    print(f"  Saved {path} ({len(df)} rows)")


def plot_forecast_11(title, y_tr, y_te, forecasts, save_path):
    H = len(y_te)
    xs_tr = np.arange(len(y_tr))
    xs_te = np.arange(len(y_tr) - 1, len(y_tr) + H)
    last_tr = np.array([y_tr[-1]])

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(xs_tr, y_tr, label="train", lw=1.8, color="C0")
    ax.plot(xs_te, np.hstack([last_tr, y_te]), label="test", lw=2.0, color="C3")

    styles = {"seasonal_naive": ("--", "gray", 0.7), "auto_arima": ("--", "#4e79a7", 0.8),
              "ets": ("-", "#4e79a7", 1.2), "prophet": (":", "#4e79a7", 0.8),
              "dlinear": ("--", "#e15759", 0.9), "autoformer": ("--", "#ff9d9a", 0.8),
              "fedformer": (":", "#e15759", 0.8), "nbeats": ("-", "#59a14f", 0.9),
              "patchtst": ("--", "#59a14f", 0.8), "timesnet": ("-.", "#59a14f", 0.8),
              "helformer": ("-", "#b07aa1", 1.1)}

    for k, v in forecasts.items():
        ls, c, lw = styles.get(k, ("--", "gray", 0.7))
        ax.plot(xs_te, np.hstack([last_tr, v[:H]]), label=k, ls=ls, color=c, lw=lw, alpha=0.9)

    ax.axvline(x=len(y_tr) - 1, color="0.35", ls="--", lw=1.0, alpha=0.5)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=3, loc="upper left")
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_rollout_comparison(title, y_tr, y_te, direct_preds, rollout_preds, save_path):
    H = len(y_te)
    xs_te = np.arange(H)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xs_te, y_te, label="actual", lw=2.0, color="black")
    for nm in direct_preds:
        if nm in rollout_preds:
            ax.plot(xs_te, direct_preds[nm][:H], label=f"{nm} (direct)", ls="--", alpha=0.7)
            ax.plot(xs_te, rollout_preds[nm][:H], label=f"{nm} (rollout)", ls="-", alpha=0.9)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_pretrain_comparison(title, y_tr, y_te, scratch_pred, pretrain_pred, model_name, save_path):
    H = len(y_te)
    xs_te = np.arange(H)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(xs_te, y_te, label="actual", lw=2.0, color="black")
    ax.plot(xs_te, scratch_pred[:H], label=f"{model_name} (с нуля)", ls="--", color="#e15759", lw=1.5)
    ax.plot(xs_te, pretrain_pred[:H], label=f"{model_name} (предобучение)", ls="-", color="#59a14f", lw=1.5)
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


# ---- Synthetic series generation ----
def generate_synthetic():
    np.random.seed(42)
    t = np.arange(200)
    profiles = {
        "trend_only": 50 + 0.5 * t + np.random.randn(200) * 2,
        "seasonal": 100 + 20 * np.sin(2 * np.pi * t / 12) + np.random.randn(200) * 3,
        "trend_seasonal": 50 + 0.3 * t + 15 * np.sin(2 * np.pi * t / 12) + np.random.randn(200) * 2,
        "noisy": 100 + np.random.randn(200) * 20,
        "changepoint": np.where(t < 100, 50 + 0.5 * t, 100 - 0.3 * (t - 100)) + np.random.randn(200) * 3,
        "complex": 50 + 0.2 * t + 10 * np.sin(2 * np.pi * t / 12) + 5 * np.sin(2 * np.pi * t / 4) + np.random.randn(200) * 2,
    }
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--out-dir", default="src/outputs/full_forecasts")
    parser.add_argument("--task", default="all", choices=["all", "standard", "rollout", "pretrain", "synthetic"])
    args = parser.parse_args()

    global DEVICE, EPOCHS, OUT_DIR
    DEVICE = args.device
    EPOCHS = args.epochs
    OUT_DIR = Path(args.out_dir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plots_dir = OUT_DIR / "plots"
    plots_dir.mkdir(exist_ok=True)

    torch.manual_seed(42)
    np.random.seed(42)

    tasks = [args.task] if args.task != "all" else ["standard", "rollout", "pretrain", "synthetic"]

    if "standard" in tasks:
        print("\n=== STANDARD FORECAST PLOTS (11 models) ===")
        ensure_m3_csv()
        ensure_m4_csv(categories=["quarterly", "monthly", "daily"])

        datasets = [
            ("m3", ["yearly", "quarterly", "monthly"], M3_H, M3_P, load_train_tsts),
            ("m4", ["quarterly", "monthly"], M4_H, M4_P, load_m4_train_test),
        ]
        # M4 daily separately with cap
        rng = np.random.default_rng(42)

        for ds_name, cats, H_map, P_map, load_fn in datasets:
            for cat in cats:
                H, P = H_map[cat], P_map[cat]
                pairs = load_fn(cat)
                # Same selection as benchmark
                rng2 = np.random.default_rng(42)
                idx = rng2.choice(len(pairs), size=min(30, len(pairs)), replace=False)
                selected = [pairs[int(i)] for i in sorted(idx)]
                # Pick 3 representative: first, middle, last
                pick = [selected[0], selected[len(selected)//2], selected[-1]]

                for sid, y_tr, y_te in pick:
                    # Cap training for daily
                    y_tr_use = y_tr[-500:] if len(y_tr) > 500 else y_tr
                    print(f"[{ds_name} {cat} {sid}] len={len(y_tr_use)}, H={H}")
                    forecasts, _ = forecast_all_models(y_tr_use, y_te, H, P, cat, DEVICE, EPOCHS)

                    # Save predictions to CSV
                    rec = {"dataset": ds_name, "category": cat, "series_id": sid}
                    for step in range(H):
                        rec[f"actual_{step}"] = y_te[step] if step < len(y_te) else np.nan
                        for m, pred in forecasts.items():
                            rec[f"{m}_{step}"] = pred[step] if step < len(pred) else np.nan
                    pd.DataFrame([rec]).to_csv(
                        OUT_DIR / f"pred_{ds_name}_{cat}_{sid}.csv", index=False)

                    plot_forecast_11(
                        f"{ds_name.upper()} {cat} — {sid} (H={H})",
                        y_tr_use, y_te, forecasts,
                        plots_dir / f"{ds_name}_{cat}_{sid}.png")

        # M4 daily
        pairs = load_m4_train_test("daily")
        rng3 = np.random.default_rng(42)
        idx = rng3.choice(len(pairs), size=min(30, len(pairs)), replace=False)
        selected = [pairs[int(i)] for i in sorted(idx)]
        pick = [selected[0], selected[len(selected)//2], selected[-1]]
        H, P = M4_H["daily"], M4_P["daily"]
        for sid, y_tr, y_te in pick:
            y_tr_use = y_tr[-300:]
            print(f"[m4 daily {sid}] len={len(y_tr_use)}, H={H}")
            forecasts, _ = forecast_all_models(y_tr_use, y_te, H, P, "daily", DEVICE, min(EPOCHS, 30))
            rec = {"dataset": "m4", "category": "daily", "series_id": sid}
            for step in range(H):
                rec[f"actual_{step}"] = y_te[step]
                for m, pred in forecasts.items():
                    rec[f"{m}_{step}"] = pred[step] if step < len(pred) else np.nan
            pd.DataFrame([rec]).to_csv(OUT_DIR / f"pred_m4_daily_{sid}.csv", index=False)
            plot_forecast_11(f"M4 daily — {sid} (H={H})", y_tr_use, y_te, forecasts,
                            plots_dir / f"m4_daily_{sid}.png")

    if "rollout" in tasks:
        print("\n=== ROLLOUT COMPARISON PLOTS ===")
        ensure_m3_csv()
        for cat in ["quarterly", "monthly"]:
            H, P = M3_H[cat], M3_P[cat]
            pairs = load_train_tsts(cat)
            rng_r = np.random.default_rng(42)
            idx = rng_r.choice(len(pairs), size=min(30, len(pairs)), replace=False)
            sel = [pairs[int(i)] for i in sorted(idx)]
            sid, y_tr, y_te = sel[len(sel)//2]  # median series
            print(f"[rollout m3 {cat} {sid}]")

            direct_preds = {}
            rollout_preds = {}
            for nm in ["dlinear", "timesnet", "nbeats", "patchtst"]:
                try:
                    pred, model = train_and_predict(nm, y_tr, H, P, EPOCHS, DEVICE)
                    if pred is not None and model is not None:
                        direct_preds[nm] = pred
                        L = best_L(y_tr, H, P)
                        ds = WindowDatasetStd(y_tr, L, H)
                        mu, sd = ds.scaler
                        rollout_preds[nm] = rollout_predict(model, y_tr, L, H, mu, sd, DEVICE)
                except Exception as e:
                    print(f"  {nm} rollout failed: {e}")

            plot_rollout_comparison(
                f"M3 {cat} — {sid}: прямой vs итеративный прогноз",
                y_tr, y_te, direct_preds, rollout_preds,
                plots_dir / f"rollout_m3_{cat}_{sid}.png")

    if "pretrain" in tasks:
        print("\n=== PRETRAIN COMPARISON PLOTS ===")
        ensure_m3_csv()
        H, P = M3_H["monthly"], M3_P["monthly"]
        pairs = load_train_tsts("monthly")
        rng_p = np.random.default_rng(42)
        idx = rng_p.choice(len(pairs), size=min(30, len(pairs)), replace=False)
        sel = [pairs[int(i)] for i in sorted(idx)]

        # Pretrain on first 100 series
        from hybridts.training import MultiSeriesWindowDataset
        L_avg = int(np.mean([best_L(y_tr, H, P) for _, y_tr, _ in pairs[:200]]))
        pretrain_series = [y_tr for _, y_tr, _ in pairs[:200]]
        pretrain_ds = MultiSeriesWindowDataset(pretrain_series, L_avg, H)
        print(f"  Pretrain dataset: {len(pretrain_ds)} windows, L_avg={L_avg}")

        for nm in ["patchtst", "autoformer"]:
            # Pretrain
            cfg_pt = TrainConfig(lookback=L_avg, horizon=H, epochs=30, batch_size=256,
                                lr=1e-3, weight_decay=1e-4, clip=1.0, device=DEVICE)
            model_pt = make_model(nm, cfg_pt)
            train_model(model_pt, pretrain_ds, cfg_pt)
            pretrained_state = {k: v.clone() for k, v in model_pt.state_dict().items()}

            sid, y_tr, y_te = sel[len(sel)//2]  # median
            print(f"[pretrain {nm} on {sid}]")

            # From-scratch
            pred_scratch, _ = train_and_predict(nm, y_tr, H, P, EPOCHS, DEVICE)

            # Pretrain + finetune
            L = best_L(y_tr, H, P)
            cfg_ft = TrainConfig(lookback=L_avg, horizon=H, epochs=20, batch_size=32,
                                lr=3e-4, weight_decay=1e-4, clip=1.0, device=DEVICE)
            model_ft = make_model(nm, cfg_ft)
            model_ft.load_state_dict(pretrained_state)
            # Pad/truncate y_tr for L_avg
            if len(y_tr) > L_avg + H:
                ds_ft = WindowDatasetStd(y_tr, L_avg, H)
            else:
                ds_ft = WindowDatasetStd(y_tr, min(L_avg, len(y_tr) - H - 1), H)
            train_model(model_ft, ds_ft, cfg_ft)
            mu, sd = ds_ft.scaler
            model_ft.eval()
            with torch.no_grad():
                x = torch.tensor(y_tr[-L_avg:], dtype=torch.float32)
                x_norm = (x - mu) / sd
                x_in = x_norm.unsqueeze(0).unsqueeze(0).to(DEVICE)
                pred_pt_norm = model_ft(x_in).squeeze(0).cpu().numpy()
            pred_pretrain = pred_pt_norm * sd + mu

            if pred_scratch is not None and pred_pretrain is not None:
                plot_pretrain_comparison(
                    f"M3 monthly — {sid}: {nm} с~предобучением и~без",
                    y_tr, y_te, pred_scratch, pred_pretrain, nm,
                    plots_dir / f"pretrain_m3_monthly_{nm}_{sid}.png")

    if "synthetic" in tasks:
        print("\n=== SYNTHETIC SERIES (11 models) ===")
        profiles = generate_synthetic()
        H, P = 24, 12
        synth_records = []

        for pname, series in profiles.items():
            y_tr = series[:176]
            y_te = series[176:]
            print(f"[synth {pname}] len_tr={len(y_tr)}, len_te={len(y_te)}")
            forecasts, _ = forecast_all_models(y_tr, y_te, H, P, "monthly", DEVICE, EPOCHS)

            rec = {"profile": pname}
            for m, pred in forecasts.items():
                rec[f"{m}_sMAPE"] = smape(y_te, pred)
                rec[f"{m}_MASE"] = mase(y_te, pred, y_tr, P)
            synth_records.append(rec)

            plot_forecast_11(f"Синтетический ряд: {pname} (H={H})",
                            y_tr, y_te, forecasts,
                            plots_dir / f"synth_{pname}.png")

        save_forecasts_csv(synth_records, OUT_DIR / "synth_metrics.csv")

    print(f"\n=== ALL DONE. Output: {OUT_DIR} ===")


if __name__ == "__main__":
    main()
