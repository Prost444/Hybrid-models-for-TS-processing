"""Unified M4 benchmark pipeline.

Runs all models (classical + neural) on M4 dataset with 6 metrics.
Mirrors the M3 benchmark pipeline structure for consistency.
"""
from __future__ import annotations

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

from ..config.settings import settings
from ..data import (
    M4_H, M4_P, best_L, ensure_m4_csv, load_m4_train_test,
    plot_forecast, seasonal_naive,
    smape, mape, mse, rmse, mae, mase,
)
from ..models import make_model, arima_forecast, auto_arima_forecast, ets_forecast, prophet_forecast
from ..training import TrainConfig, WindowDatasetStd, train_model


def _progress(it, **kw):
    return tqdm(it, **kw) if tqdm is not None else it


# ---- Neural model helpers ------------------------------------------------

def _train_and_forecast(model_name: str, y_tr: np.ndarray, H: int, P: int,
                        epochs: int, model_params: dict | None = None,
                        device: str = "cpu") -> np.ndarray:
    """Train a neural model on one series and return H-step forecast."""
    L = best_L(y_tr, H, P)
    cfg = TrainConfig(lookback=L, horizon=H, epochs=epochs, batch_size=32,
                      lr=1e-3, weight_decay=1e-4, clip=1.0, device=device)

    model = make_model(model_name, cfg, params=model_params)
    ds = WindowDatasetStd(y_tr, L, H)

    if len(ds) < 1:
        raise ValueError(f"Not enough data for {model_name}: len(y_tr)={len(y_tr)}, L={L}, H={H}")

    train_model(model, ds, cfg)

    mu, sd = ds.scaler
    model.eval()
    with torch.no_grad():
        x = torch.tensor(y_tr[-L:], dtype=torch.float32)
        x_norm = (x - mu) / sd
        x_norm = x_norm.unsqueeze(0).unsqueeze(0).to(cfg.device)  # (1, 1, L)
        pred_norm = model(x_norm).squeeze(0).cpu().numpy()
    return pred_norm * sd + mu


# ---- Metric computation ---------------------------------------------------

def _compute_metrics(y_te: np.ndarray, pred: np.ndarray,
                     y_tr: np.ndarray, P: int) -> Dict[str, float]:
    """Compute all 6 metrics."""
    return {
        "sMAPE": smape(y_te, pred),
        "MAPE": mape(y_te, pred),
        "RMSE": rmse(y_te, pred),
        "MSE": mse(y_te, pred),
        "MAE": mae(y_te, pred),
        "MASE": mase(y_te, pred, y_tr, P),
    }


# ---- Main entry point -----------------------------------------------------

def run_m4_benchmark(
    categories: Iterable[str] = ("quarterly", "monthly", "daily"),
    n_per_cat: int = 30,
    pick: str = "random",
    seed: int = 42,
    neural_epochs: int = 50,
    neural_models: Sequence[str] = ("dlinear", "autoformer", "nbeats", "timesnet"),
    classical_models: Sequence[str] = ("seasonal_naive", "auto_arima", "ets", "prophet"),
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
    out_dir: str | Path | None = None,
    visualize: bool = False,
    device: str = "cpu",
    force_rebuild_csv: bool = False,
) -> pd.DataFrame:
    """Run a full M4 benchmark.

    Parameters
    ----------
    categories : sequence of str
        M4 frequency categories to evaluate.
        Available: yearly, quarterly, monthly, weekly, daily, hourly.
    n_per_cat : int
        Number of series per category. 0 = all.
    pick : str
        "random", "first", or "last".
    seed : int
        Random seed for reproducibility.
    neural_epochs : int
        Training epochs for neural models.
    neural_models : sequence of str
        Neural model names (passed to make_model).
    classical_models : sequence of str
        Classical model identifiers.
    model_params : dict
        Per-model parameter overrides, e.g. {"dlinear": {"kernel_size": 15}}.
    out_dir : Path
        Output directory for metrics.csv and plots.
    visualize : bool
        Save per-series forecast plots.
    device : str
        PyTorch device ("cpu" or "cuda").
    force_rebuild_csv : bool
        Force rebuild of M4 CSV files from raw wide format.

    Returns
    -------
    pd.DataFrame with per-series, per-model metrics.
    """
    out_path = Path(out_dir or (settings.outputs_dir / "m4_benchmark"))
    out_path.mkdir(parents=True, exist_ok=True)

    categories = tuple(categories)
    ensure_m4_csv(categories=categories, force_rebuild=force_rebuild_csv)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    freq_map = {
        "yearly": "YE", "quarterly": "QE", "monthly": "ME",
        "weekly": "W", "daily": "D", "hourly": "h",
    }
    all_model_names = list(classical_models) + list(neural_models)

    rows: List[Dict] = []
    checkpoint_csv = out_path / "metrics_checkpoint.csv"
    t0 = time.time()

    # Resume from checkpoint if it exists
    if checkpoint_csv.exists():
        prev = pd.read_csv(checkpoint_csv)
        rows = prev.to_dict("records")
        done_keys = {(r["category"], r["series_id"]) for r in rows}
        print(f"[checkpoint] resuming — {len(rows)} series already done")
    else:
        done_keys = set()

    for cat in _progress(categories, desc="M4 categories"):
        H = M4_H[cat]
        P = M4_P[cat]
        pairs = load_m4_train_test(cat)
        if not pairs:
            print(f"[{cat}] no data found — skipping")
            continue

        # Select series
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

        print(f"[{cat}] H={H}, P={P}, series={len(selected)}")

        for sid, y_tr, y_te in _progress(selected, desc=f"{cat}", leave=False):
            if (cat, sid) in done_keys:
                continue

            rec: Dict[str, Any] = {"category": cat, "series_id": sid}
            forecasts: Dict[str, np.ndarray] = {}

            # --- Classical models ---
            for cname in classical_models:
                try:
                    if cname == "seasonal_naive":
                        pred = seasonal_naive(y_tr, H, P)
                    elif cname == "auto_arima":
                        pred = np.asarray(auto_arima_forecast(y_tr, H), float)
                    elif cname == "ets":
                        pred = np.asarray(ets_forecast(y_tr, H, seasonal_periods=P), float)
                    elif cname == "prophet":
                        freq = freq_map.get(cat, "D")
                        pred = np.asarray(prophet_forecast(y_tr, H, freq=freq), float)
                    elif cname == "arima":
                        pred = np.asarray(arima_forecast(y_tr, H), float)
                    else:
                        print(f"[{cat}:{sid}] unknown classical model: {cname}")
                        continue
                    forecasts[cname] = pred
                    metrics = _compute_metrics(y_te, pred, y_tr, P)
                    for mk, mv in metrics.items():
                        rec[f"{cname}_{mk}"] = mv
                except Exception as exc:
                    print(f"[{cat}:{sid}] {cname} failed: {exc}")

            # --- Neural models ---
            for nname in neural_models:
                try:
                    params = (model_params or {}).get(nname)
                    pred = _train_and_forecast(nname, y_tr, H, P,
                                               epochs=neural_epochs,
                                               model_params=params,
                                               device=device)
                    pred = np.asarray(pred, float).ravel()[:H]
                    forecasts[nname] = pred
                    metrics = _compute_metrics(y_te, pred, y_tr, P)
                    for mk, mv in metrics.items():
                        rec[f"{nname}_{mk}"] = mv
                except Exception as exc:
                    print(f"[{cat}:{sid}] {nname} failed: {exc}")

            rows.append(rec)

            # Incremental checkpoint after each series
            pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)

            if visualize:
                save_png = out_path / f"{cat}_{sid}.png"
                plot_forecast(
                    f"{cat.upper()} {sid} (H={H})",
                    y_tr, y_te, forecasts, save_path=save_png,
                )

    df = pd.DataFrame(rows)
    metrics_csv = out_path / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    elapsed = time.time() - t0
    print(f"\n[saved] {metrics_csv}  ({len(df)} series, {elapsed:.0f}s)")

    # Print summary
    if not df.empty:
        metric_names = ["sMAPE", "MAPE", "RMSE", "MSE", "MAE", "MASE"]
        for mname in metric_names:
            cols = [c for c in df.columns if c.endswith(f"_{mname}")]
            if not cols:
                continue
            print(f"\n--- {mname} mean by category ---")
            summary = df.groupby("category")[cols].mean(numeric_only=True).round(3)
            print(summary.to_string())
        print(f"\n--- Overall means ---")
        metric_cols = [c for c in df.columns if any(c.endswith(f"_{m}") for m in metric_names)]
        if metric_cols:
            overall = df[metric_cols].mean(numeric_only=True).round(3)
            print(overall.to_string())

    return df


__all__ = ["run_m4_benchmark"]
