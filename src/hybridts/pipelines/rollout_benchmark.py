"""Rollout benchmark pipeline.

Evaluates models using iterative single-step rollout forecasting (predict one
step, append to history, repeat) in addition to the standard direct multi-step
forecast.  This lets us compare direct vs. rollout strategies for every model.
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
    M3_H, M3_P, M4_H, M4_P, best_L,
    ensure_m3_csv, load_train_tsts,
    ensure_m4_csv, load_m4_train_test,
    plot_forecast, seasonal_naive,
    smape, mape, mse, rmse, mae, mase,
)
from ..models import make_model, auto_arima_forecast, ets_forecast, prophet_forecast
from ..training import TrainConfig, WindowDatasetStd, train_model


def _progress(it, **kw):
    return tqdm(it, **kw) if tqdm is not None else it


# ---- Metric computation ---------------------------------------------------

def _compute_metrics(y_te: np.ndarray, pred: np.ndarray,
                     y_tr: np.ndarray, P: int) -> Dict[str, float]:
    return {
        "sMAPE": smape(y_te, pred),
        "MAPE": mape(y_te, pred),
        "RMSE": rmse(y_te, pred),
        "MSE": mse(y_te, pred),
        "MAE": mae(y_te, pred),
        "MASE": mase(y_te, pred, y_tr, P),
    }


# ---- Direct forecast (standard H-step) ------------------------------------

def _train_and_forecast_direct(
    model_name: str, y_tr: np.ndarray, H: int, P: int,
    epochs: int, device: str = "cpu",
) -> np.ndarray:
    """Train a neural model on one series and return H-step direct forecast."""
    L = best_L(y_tr, H, P)
    cfg = TrainConfig(
        lookback=L, horizon=H, epochs=epochs, batch_size=32,
        lr=1e-3, weight_decay=1e-4, clip=1.0, device=device,
    )
    model = make_model(model_name, cfg)
    ds = WindowDatasetStd(y_tr, L, H)
    if len(ds) < 1:
        raise ValueError(f"Not enough data for {model_name}: len(y_tr)={len(y_tr)}, L={L}, H={H}")

    train_model(model, ds, cfg)

    mu, sd = ds.scaler
    model.eval()
    with torch.no_grad():
        x = torch.tensor(y_tr[-L:], dtype=torch.float32)
        x_norm = (x - mu) / sd
        x_norm = x_norm.unsqueeze(0).unsqueeze(0).to(cfg.device)
        pred_norm = model(x_norm).squeeze(0).cpu().numpy()
    return pred_norm * sd + mu


# ---- Rollout forecast (neural) --------------------------------------------

def _rollout_neural(
    model: torch.nn.Module, y_tr: np.ndarray, L: int, H: int,
    mu: float, sd: float, device: str,
) -> np.ndarray:
    """Iterative single-step rollout for a neural model trained with H-step output."""
    model.eval()
    history = list(y_tr.astype(float))
    preds = []
    dev = torch.device(device)

    for step in range(H):
        x = np.array(history[-L:], dtype=np.float32)
        x_norm = (x - mu) / sd
        x_t = torch.tensor(x_norm).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            out = model(x_t).squeeze(0).cpu().numpy()
        # Take only the first step prediction
        pred_raw = float(out[0]) * sd + mu
        if not np.isfinite(pred_raw):
            pred_raw = history[-1]
        preds.append(pred_raw)
        history.append(pred_raw)

    return np.array(preds)


# ---- Rollout forecast (classical) -----------------------------------------

def _rollout_auto_arima(y_tr: np.ndarray, H: int) -> np.ndarray:
    """Iterative 1-step ARIMA rollout using pmdarima."""
    try:
        import pmdarima as pm
    except ImportError:
        return auto_arima_forecast(y_tr, H)

    model = pm.auto_arima(y_tr, seasonal=False, suppress_warnings=True,
                          error_action="ignore", stepwise=True)
    preds = []
    for _ in range(H):
        fc = model.predict(n_periods=1)
        val = float(fc[0])
        if not np.isfinite(val):
            val = float(y_tr[-1]) if len(y_tr) > 0 else 0.0
        preds.append(val)
        model.update([val])
    return np.array(preds)


def _rollout_ets(y_tr: np.ndarray, H: int, P: int) -> np.ndarray:
    """Iterative 1-step ETS rollout using statsmodels."""
    try:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
    except ImportError:
        return np.asarray(ets_forecast(y_tr, H, seasonal_periods=P), float)

    sp = P if P > 1 else None
    s_mode = "mul" if sp else None
    t_mode = "mul"
    data = np.asarray(y_tr, float)
    if np.any(data <= 0):
        t_mode = "add"
        s_mode = "add" if s_mode == "mul" else s_mode

    try:
        model = ExponentialSmoothing(data, trend=t_mode, seasonal=s_mode,
                                     seasonal_periods=sp)
        fitted = model.fit(optimized=True, use_brute=True)
    except Exception:
        return np.asarray(ets_forecast(y_tr, H, seasonal_periods=P), float)

    preds = []
    history = list(data)
    for _ in range(H):
        try:
            fc = fitted.forecast(1)
            val = float(fc[0])
        except Exception:
            val = history[-1]
        if not np.isfinite(val):
            val = history[-1]
        preds.append(val)
        history.append(val)
        # Refit with extended history for next step
        try:
            arr = np.array(history, float)
            model2 = ExponentialSmoothing(arr, trend=t_mode, seasonal=s_mode,
                                          seasonal_periods=sp)
            fitted = model2.fit(optimized=True, use_brute=True)
        except Exception:
            pass  # keep previous fitted model

    return np.array(preds)


# ---- Main entry point -----------------------------------------------------

def run_rollout_benchmark(
    dataset: str = "m3",
    categories: Iterable[str] = ("yearly", "quarterly", "monthly"),
    n_per_cat: int = 30,
    pick: str = "random",
    seed: int = 42,
    neural_epochs: int = 80,
    neural_models: Sequence[str] = ("dlinear", "autoformer", "fedformer", "patchtst", "nbeats", "timesnet"),
    classical_models: Sequence[str] = ("seasonal_naive", "auto_arima", "ets"),
    device: str = "cpu",
    out_dir: str | Path | None = None,
    max_train_samples: int = 0,
) -> pd.DataFrame:
    """Run a rollout benchmark comparing direct vs. iterative forecasting.

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
        Random seed.
    neural_epochs : int
        Training epochs for neural models.
    neural_models : sequence of str
        Neural model names.
    classical_models : sequence of str
        Classical model identifiers.
    device : str
        PyTorch device.
    out_dir : Path
        Output directory.
    max_train_samples : int
        If >0, subsample windows for very long series.

    Returns
    -------
    pd.DataFrame with per-series, per-model metrics for both direct and rollout.
    """
    out_path = Path(out_dir or (settings.outputs_dir / f"rollout_{dataset}_benchmark"))
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

    for cat in _progress(categories, desc=f"Rollout {dataset}"):
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

        print(f"[{cat}] H={H}, P={P}, series={len(selected)}")

        for sid, y_tr, y_te in _progress(selected, desc=f"{cat}", leave=False):
            if (cat, sid) in done_keys:
                continue

            rec: Dict[str, Any] = {"category": cat, "series_id": sid}

            # --- Classical models: direct ---
            for cname in classical_models:
                try:
                    if cname == "seasonal_naive":
                        pred_direct = seasonal_naive(y_tr, H, P)
                        # Seasonal naive rollout is the same as direct
                        pred_rollout = pred_direct.copy()
                    elif cname == "auto_arima":
                        pred_direct = np.asarray(auto_arima_forecast(y_tr, H), float)
                        pred_rollout = _rollout_auto_arima(y_tr, H)
                    elif cname == "ets":
                        pred_direct = np.asarray(ets_forecast(y_tr, H, seasonal_periods=P), float)
                        pred_rollout = _rollout_ets(y_tr, H, P)
                    else:
                        print(f"[{cat}:{sid}] unknown classical model: {cname}")
                        continue

                    # Direct metrics
                    m_dir = _compute_metrics(y_te, pred_direct, y_tr, P)
                    for mk, mv in m_dir.items():
                        rec[f"{cname}_direct_{mk}"] = mv

                    # Rollout metrics
                    m_roll = _compute_metrics(y_te, pred_rollout, y_tr, P)
                    for mk, mv in m_roll.items():
                        rec[f"{cname}_rollout_{mk}"] = mv

                except Exception as exc:
                    print(f"[{cat}:{sid}] {cname} failed: {exc}")

            # --- Neural models: direct + rollout ---
            for nname in neural_models:
                try:
                    L = best_L(y_tr, H, P)

                    # Train model
                    cfg = TrainConfig(
                        lookback=L, horizon=H, epochs=neural_epochs,
                        batch_size=32, lr=1e-3, weight_decay=1e-4,
                        clip=1.0, device=device,
                    )
                    model = make_model(nname, cfg)
                    ds = WindowDatasetStd(y_tr, L, H)
                    if len(ds) < 1:
                        print(f"[{cat}:{sid}] {nname}: not enough data")
                        continue

                    train_model(model, ds, cfg)
                    mu, sd = ds.scaler

                    # Direct forecast
                    model.eval()
                    with torch.no_grad():
                        x = torch.tensor(y_tr[-L:], dtype=torch.float32)
                        x_norm = (x - mu) / sd
                        x_norm = x_norm.unsqueeze(0).unsqueeze(0).to(cfg.device)
                        pred_norm = model(x_norm).squeeze(0).cpu().numpy()
                    pred_direct = pred_norm * sd + mu
                    pred_direct = np.asarray(pred_direct, float).ravel()[:H]

                    m_dir = _compute_metrics(y_te, pred_direct, y_tr, P)
                    for mk, mv in m_dir.items():
                        rec[f"{nname}_direct_{mk}"] = mv

                    # Rollout forecast
                    pred_rollout = _rollout_neural(model, y_tr, L, H, mu, sd, device)
                    pred_rollout = np.asarray(pred_rollout, float).ravel()[:H]

                    m_roll = _compute_metrics(y_te, pred_rollout, y_tr, P)
                    for mk, mv in m_roll.items():
                        rec[f"{nname}_rollout_{mk}"] = mv

                except Exception as exc:
                    print(f"[{cat}:{sid}] {nname} failed: {exc}")

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
        for strategy in ("direct", "rollout"):
            print(f"\n{'='*60}")
            print(f"  Strategy: {strategy.upper()}")
            print(f"{'='*60}")
            for mname in metric_names:
                cols = [c for c in df.columns if c.endswith(f"_{strategy}_{mname}")]
                if not cols:
                    continue
                print(f"\n--- {mname} mean by category ({strategy}) ---")
                summary = df.groupby("category")[cols].mean(numeric_only=True).round(3)
                print(summary.to_string())

        # Overall direct vs rollout comparison
        print(f"\n{'='*60}")
        print(f"  Direct vs. Rollout Overall Comparison")
        print(f"{'='*60}")
        for mname in metric_names:
            dir_cols = [c for c in df.columns if c.endswith(f"_direct_{mname}")]
            roll_cols = [c for c in df.columns if c.endswith(f"_rollout_{mname}")]
            if dir_cols and roll_cols:
                dir_mean = df[dir_cols].mean(numeric_only=True).mean()
                roll_mean = df[roll_cols].mean(numeric_only=True).mean()
                print(f"  {mname:>6s}:  direct={dir_mean:.3f}  rollout={roll_mean:.3f}")

    return df


__all__ = ["run_rollout_benchmark"]
