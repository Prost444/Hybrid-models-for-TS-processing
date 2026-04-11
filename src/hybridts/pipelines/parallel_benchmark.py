"""Parallelized per-series benchmark using joblib.

Speeds up the standard benchmark by training multiple series in parallel.
Each worker handles all models for one series, avoiding GPU contention.
"""
from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

try:
    from joblib import Parallel, delayed
except ImportError:
    Parallel = None

from ..config.settings import settings
from ..data import (
    M3_H, M3_P, M4_H, M4_P, best_L,
    ensure_m3_csv, load_train_tsts,
    ensure_m4_csv, load_m4_train_test,
    seasonal_naive,
    smape, mape, mse, rmse, mae, mase,
)
from ..models import make_model, auto_arima_forecast, ets_forecast, prophet_forecast
from ..models.helformer_pt import helformer_forecast_pt
from ..training import TrainConfig, WindowDatasetStd, train_model


def _compute_metrics(y_te, pred, y_tr, P):
    return {
        "sMAPE": smape(y_te, pred),
        "MAPE": mape(y_te, pred),
        "RMSE": rmse(y_te, pred),
        "MSE": mse(y_te, pred),
        "MAE": mae(y_te, pred),
        "MASE": mase(y_te, pred, y_tr, P),
    }


def _train_and_forecast(model_name, y_tr, H, P, epochs, device):
    L = best_L(y_tr, H, P)
    bs = 256 if len(y_tr) - L - H + 1 > 1024 else (128 if len(y_tr) - L - H + 1 > 256 else 32)
    cfg = TrainConfig(lookback=L, horizon=H, epochs=epochs, batch_size=bs,
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
    return np.asarray(pred_norm * sd + mu, float).ravel()[:H]


def _predict_helformer(y_tr, H, P, epochs, device):
    L = min(30, len(y_tr) - 2)
    if L < 4:
        L = max(4, len(y_tr) // 2)
    cfg = TrainConfig(lookback=L, horizon=1, epochs=epochs, batch_size=32,
                      lr=8e-4, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model("helformer", cfg)
    vmin = float(np.nanmin(y_tr))
    scale = float(np.nanmax(y_tr)) - vmin
    if scale < 1e-8:
        scale = 1.0
    scaled = (y_tr - vmin) / scale
    ds = WindowDatasetStd(scaled, L, 1, scale=False)
    if len(ds) < 1:
        return None
    train_model(model, ds, cfg)
    return helformer_forecast_pt(y_tr, H, model, lookback=L,
                                 seasonal_period=P, use_hw=True, device=device)


FREQ_MAP = {"yearly": "YE", "quarterly": "QE", "monthly": "ME",
            "weekly": "W", "daily": "D", "hourly": "h"}


def _process_one_series(sid, y_tr, y_te, H, P, cat, neural_models,
                        classical_models, neural_epochs, device):
    """Process all models for a single series. Returns a dict of metrics."""
    rec = {"category": cat, "series_id": sid}

    # Classical models
    for cname in classical_models:
        try:
            if cname == "seasonal_naive":
                pred = seasonal_naive(y_tr, H, P)
            elif cname == "auto_arima":
                pred = np.asarray(auto_arima_forecast(y_tr, H), float)
            elif cname == "ets":
                pred = np.asarray(ets_forecast(y_tr, H, seasonal_periods=P), float)
            elif cname == "prophet":
                freq = FREQ_MAP.get(cat, "D")
                pred = np.asarray(prophet_forecast(y_tr, H, freq=freq), float)
            else:
                continue
            metrics = _compute_metrics(y_te, pred, y_tr, P)
            for mk, mv in metrics.items():
                rec[f"{cname}_{mk}"] = mv
        except Exception:
            pass

    # Neural models
    for nname in neural_models:
        try:
            if nname == "helformer":
                pred = _predict_helformer(y_tr, H, P, neural_epochs, device)
            else:
                pred = _train_and_forecast(nname, y_tr, H, P, neural_epochs, device)
            if pred is not None:
                pred = np.asarray(pred, float).ravel()[:H]
                metrics = _compute_metrics(y_te, pred, y_tr, P)
                for mk, mv in metrics.items():
                    rec[f"{nname}_{mk}"] = mv
        except Exception:
            pass

    return rec


def run_parallel_benchmark(
    dataset: str = "m3",
    categories: Iterable[str] = ("yearly", "quarterly", "monthly"),
    n_per_cat: int = 0,
    pick: str = "random",
    seed: int = 42,
    neural_epochs: int = 50,
    neural_models: Sequence[str] = ("dlinear", "autoformer", "fedformer",
                                     "patchtst", "nbeats", "timesnet", "helformer"),
    classical_models: Sequence[str] = ("seasonal_naive", "auto_arima", "ets", "prophet"),
    n_jobs: int = 4,
    device: str = "cpu",
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run benchmark with joblib parallelization across series.

    Parameters
    ----------
    n_jobs : int
        Number of parallel workers. Use -1 for all CPUs.
        For GPU: use 1-2 (GPU memory shared).
    device : str
        "cpu" for parallel CPU workers, "cuda" for GPU (use n_jobs=1-2).
    """
    out_path = Path(out_dir or (settings.outputs_dir / f"parallel_{dataset}_benchmark"))
    out_path.mkdir(parents=True, exist_ok=True)

    if dataset == "m3":
        ensure_m3_csv()
        H_MAP, P_MAP = M3_H, M3_P
        load_fn = load_train_tsts
    elif dataset == "m4":
        ensure_m4_csv(categories=tuple(categories))
        H_MAP, P_MAP = M4_H, M4_P
        load_fn = load_m4_train_test
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    rng = np.random.default_rng(seed)
    categories = tuple(categories)
    checkpoint_csv = out_path / "metrics_checkpoint.csv"

    # Resume
    done_keys = set()
    existing_rows = []
    if checkpoint_csv.exists():
        prev = pd.read_csv(checkpoint_csv)
        existing_rows = prev.to_dict("records")
        done_keys = {(r["category"], r["series_id"]) for r in existing_rows}
        print(f"[checkpoint] resuming — {len(existing_rows)} series done")

    all_tasks = []
    for cat in categories:
        H = H_MAP[cat]
        P = P_MAP[cat]
        pairs = load_fn(cat)
        if not pairs:
            continue

        if n_per_cat <= 0:
            selected = pairs
        elif pick == "first":
            selected = pairs[:n_per_cat]
        elif pick == "last":
            selected = pairs[-n_per_cat:]
        else:
            count = min(n_per_cat, len(pairs))
            idx = rng.choice(len(pairs), size=count, replace=False)
            selected = [pairs[int(i)] for i in sorted(idx)]

        for sid, y_tr, y_te in selected:
            if (cat, sid) not in done_keys:
                all_tasks.append((sid, y_tr, y_te, H, P, cat))

    print(f"[parallel] {len(all_tasks)} series to process, n_jobs={n_jobs}")
    t0 = time.time()

    if Parallel is None or n_jobs == 1:
        # Fallback to sequential
        results = []
        for sid, y_tr, y_te, H, P, cat in (tqdm(all_tasks, desc="series") if tqdm else all_tasks):
            rec = _process_one_series(sid, y_tr, y_te, H, P, cat,
                                      neural_models, classical_models, neural_epochs, device)
            results.append(rec)
            existing_rows.append(rec)
            pd.DataFrame(existing_rows).to_csv(checkpoint_csv, index=False)
    else:
        # Parallel execution
        batch_size = max(1, min(n_jobs * 2, len(all_tasks)))
        for i in range(0, len(all_tasks), batch_size):
            batch = all_tasks[i:i + batch_size]
            batch_results = Parallel(n_jobs=n_jobs, backend="loky")(
                delayed(_process_one_series)(
                    sid, y_tr, y_te, H, P, cat,
                    neural_models, classical_models, neural_epochs, device
                ) for sid, y_tr, y_te, H, P, cat in batch
            )
            existing_rows.extend(batch_results)
            pd.DataFrame(existing_rows).to_csv(checkpoint_csv, index=False)
            print(f"  [{i + len(batch)}/{len(all_tasks)}] checkpointed")

    df = pd.DataFrame(existing_rows)
    metrics_csv = out_path / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    elapsed = time.time() - t0
    print(f"\n[saved] {metrics_csv}  ({len(df)} series, {elapsed:.0f}s)")

    # Summary
    if not df.empty:
        for mname in ["sMAPE", "MASE"]:
            cols = [c for c in df.columns if c.endswith(f"_{mname}")]
            if cols:
                print(f"\n--- {mname} mean by category ---")
                print(df.groupby("category")[cols].mean().round(3).to_string())

    return df


__all__ = ["run_parallel_benchmark"]
