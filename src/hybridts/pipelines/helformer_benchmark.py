"""Helformer (PyTorch) benchmark pipeline.

Evaluates the HelformerPT model (with optional Holt-Winters decomposition)
on M3 or M4 time-series, using iterative single-step rollout forecasting.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from ..config.settings import settings
from ..data import (
    M3_H, M3_P, M4_H, M4_P, best_L,
    ensure_m3_csv, load_train_tsts,
    ensure_m4_csv, load_m4_train_test,
    plot_forecast, seasonal_naive,
    smape, mape, mse, rmse, mae, mase,
)
from ..models import make_model, auto_arima_forecast, ets_forecast, prophet_forecast
from ..models.helformer_pt import HelformerPT, helformer_forecast_pt, _minmax_fit, _minmax_transform
from ..training import TrainConfig, WindowDatasetStd, train_model


def _progress(it, **kw):
    return tqdm(it, **kw) if tqdm is not None else it


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


# ---- Holt-Winters helpers (reuse from helformer_pt) -----------------------

def _prepare_hw_series(y_tr: np.ndarray, horizon: int, P: Optional[int]):
    """If HW baseline succeeds, return (ratios, base_test); else None."""
    from ..models.helformer_pt import _holt_winters_baseline
    bl = _holt_winters_baseline(y_tr, horizon, P)
    if bl is None:
        return None
    base_train, base_test = bl
    denom = np.where(np.abs(base_train) < 1e-8, 1.0, base_train)
    ratios = y_tr / denom
    return ratios, base_test


# ---- Main entry point -----------------------------------------------------

def run_helformer_benchmark(
    dataset: str = "m3",
    categories: Iterable[str] = ("yearly", "quarterly", "monthly"),
    n_per_cat: int = 30,
    pick: str = "random",
    seed: int = 42,
    epochs: int = 40,
    lookback: int = 30,
    use_hw: bool = True,
    num_blocks: int = 4,
    num_heads: int = 4,
    head_size: int = 56,
    units: int = 25,
    dropout: float = 0.11,
    device: str = "cpu",
    out_dir: str | Path | None = None,
) -> pd.DataFrame:
    """Run a Helformer benchmark on M3 or M4.

    Parameters
    ----------
    dataset : str
        "m3" or "m4".
    categories : sequence of str
        Frequency categories to evaluate.
    n_per_cat : int
        Number of series per category. 0 = all.
    pick : str
        "random", "first", or "last".
    seed : int
        Random seed for reproducibility.
    epochs : int
        Training epochs for each series.
    lookback : int
        Lookback window length.
    use_hw : bool
        Use Holt-Winters decomposition (predict ratios instead of raw).
    num_blocks, num_heads, head_size, units, dropout
        HelformerPT architecture hyperparameters.
    device : str
        PyTorch device.
    out_dir : Path
        Output directory.

    Returns
    -------
    pd.DataFrame with per-series metrics.
    """
    out_path = Path(out_dir or (settings.outputs_dir / f"helformer_{dataset}_benchmark"))
    out_path.mkdir(parents=True, exist_ok=True)

    # Dataset-specific setup
    if dataset == "m3":
        ensure_m3_csv()
        H_MAP, P_MAP = M3_H, M3_P
        load_fn = load_train_tsts
    elif dataset == "m4":
        categories = tuple(categories)
        ensure_m4_csv(categories=categories)
        H_MAP, P_MAP = M4_H, M4_P
        load_fn = load_m4_train_test
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    freq_map = {
        "yearly": "YE", "quarterly": "QE", "monthly": "ME",
        "weekly": "W", "daily": "D", "hourly": "h",
    }
    categories = tuple(categories)

    rows: List[Dict] = []
    checkpoint_csv = out_path / "metrics_checkpoint.csv"
    t0 = time.time()

    # Resume from checkpoint
    if checkpoint_csv.exists():
        prev = pd.read_csv(checkpoint_csv)
        rows = prev.to_dict("records")
        done_keys = {(r["category"], r["series_id"]) for r in rows}
        print(f"[checkpoint] resuming -- {len(rows)} series already done")
    else:
        done_keys = set()

    for cat in _progress(categories, desc=f"Helformer {dataset}"):
        H = H_MAP[cat]
        P = P_MAP[cat]
        pairs = load_fn(cat)
        if not pairs:
            print(f"[{cat}] no data found")
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

        print(f"[{cat}] H={H}, P={P}, series={len(selected)}, lookback={lookback}")

        for sid, y_tr, y_te in _progress(selected, desc=f"{cat}", leave=False):
            if (cat, sid) in done_keys:
                continue

            rec: Dict[str, Any] = {"category": cat, "series_id": sid}
            try:
                # Determine the series to train on
                if use_hw:
                    hw_result = _prepare_hw_series(y_tr, H, P)
                    if hw_result is not None:
                        y_tr_processed = hw_result[0]  # ratios
                        actual_use_hw = True
                    else:
                        y_tr_processed = y_tr.copy()
                        actual_use_hw = False
                else:
                    y_tr_processed = y_tr.copy()
                    actual_use_hw = False

                # MinMax normalisation (NOT z-score) for training
                y_tr_processed = np.where(np.isfinite(y_tr_processed), y_tr_processed, 0.0)
                vmin, scale = _minmax_fit(y_tr_processed)
                scaled = _minmax_transform(y_tr_processed, vmin, scale)

                # Build dataset with scale=False (data is already MinMax-scaled)
                ds = WindowDatasetStd(scaled, lookback, 1, scale=False)

                if len(ds) < 1:
                    print(f"[{cat}:{sid}] not enough data for Helformer (len={len(y_tr)})")
                    rows.append(rec)
                    pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)
                    continue

                # Create and train model
                model = HelformerPT(
                    lookback=lookback,
                    num_blocks=num_blocks,
                    num_heads=num_heads,
                    head_size=head_size,
                    dropout=dropout,
                    units=units,
                )
                cfg = TrainConfig(
                    lookback=lookback, horizon=1, epochs=epochs,
                    batch_size=32, lr=1e-3, weight_decay=1e-4,
                    clip=1.0, device=device,
                )
                train_model(model, ds, cfg)

                # Inference via rollout (handles HW internally)
                pred = helformer_forecast_pt(
                    y_tr, H, model, lookback=lookback,
                    seasonal_period=P, use_hw=use_hw, device=device,
                )
                pred = np.asarray(pred, float).ravel()[:H]

                metrics = _compute_metrics(y_te, pred, y_tr, P)
                for mk, mv in metrics.items():
                    rec[f"helformer_{mk}"] = mv
                rec["use_hw"] = actual_use_hw

            except Exception as exc:
                print(f"[{cat}:{sid}] helformer failed: {exc}")

            rows.append(rec)
            pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)

    df = pd.DataFrame(rows)
    metrics_csv = out_path / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    elapsed = time.time() - t0
    print(f"\n[saved] {metrics_csv}  ({len(df)} series, {elapsed:.0f}s)")

    # Print summary
    if not df.empty:
        metric_names = ["sMAPE", "MAPE", "RMSE", "MSE", "MAE", "MASE"]
        helformer_cols = [c for c in df.columns if c.startswith("helformer_")]
        if helformer_cols:
            print(f"\n--- Helformer mean by category ---")
            summary = df.groupby("category")[helformer_cols].mean(numeric_only=True).round(3)
            print(summary.to_string())
            print(f"\n--- Overall means ---")
            overall = df[helformer_cols].mean(numeric_only=True).round(3)
            print(overall.to_string())

    return df


__all__ = ["run_helformer_benchmark"]
