"""Evaluation helpers for M3 Helformer experiments."""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

try:  # pragma: no cover - optional dependency
    import tensorflow as tf
except Exception:  # pragma: no cover - fallback when TF is missing
    tf = None

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover - fallback when statsmodels is missing
    ExponentialSmoothing = None
try:  # pragma: no cover - optional dependency
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except Exception:  # pragma: no cover - fallback when statsmodels is missing
    ConvergenceWarning = None

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - best-effort fallback
    tqdm = None

from ..config.settings import settings
from ..data import (
    M3_H,
    M3_P,
    ensure_m3_csv,
    load_train_tsts,
    plot_forecast,
    seasonal_naive,
    smape,
    mape,
    mse,
    rmse,
)
from ..models import arima_forecast, auto_arima_forecast, ets_forecast, create_helformer_model, prophet_forecast
from ..viz import save_series_viz_bundle_basic


def _progress(iterable, **kwargs):
    if tqdm is None:
        return iterable
    return tqdm(iterable, **kwargs)


def _ensure_min_length(series: np.ndarray, min_len: int) -> np.ndarray:
    series = np.asarray(series, float).ravel()
    if series.size == 0:
        return np.zeros(min_len, dtype=float)
    if series.size >= min_len:
        return series
    pad_len = min_len - series.size
    pad_val = float(series[0])
    pad = np.full(pad_len, pad_val, dtype=float)
    return np.concatenate([pad, series])


def _minmax_fit(series: np.ndarray) -> tuple[float, float]:
    series = np.asarray(series, float).ravel()
    if series.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(series))
    vmax = float(np.nanmax(series))
    scale = vmax - vmin
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        scale = 1.0
    return vmin, float(scale)


def _minmax_transform(series: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return (np.asarray(series, float) - vmin) / scale


def _minmax_inverse(series: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return np.asarray(series, float) * scale + vmin


def _build_xy(series: np.ndarray, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    series = np.asarray(series, float).ravel()
    if series.size <= lookback:
        series = _ensure_min_length(series, lookback + 1)
    X, Y = [], []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
        Y.append(series[i])
    X = np.asarray(X, np.float32).reshape(-1, lookback, 1)
    Y = np.asarray(Y, np.float32).reshape(-1, 1)
    return X, Y


def _build_x(series: np.ndarray, lookback: int) -> np.ndarray:
    series = np.asarray(series, float).ravel()
    if series.size <= lookback:
        series = _ensure_min_length(series, lookback + 1)
    X = []
    for i in range(lookback, len(series)):
        X.append(series[i - lookback : i])
    return np.asarray(X, np.float32).reshape(-1, lookback, 1)


def _hw_baseline(
    y_tr: np.ndarray,
    horizon: int,
    seasonal_period: int | None,
    trend: str,
    seasonal: str,
) -> tuple[np.ndarray, np.ndarray] | None:
    if ExponentialSmoothing is None:
        return None
    data = np.asarray(y_tr, float).ravel()
    if data.size < 4:
        return None
    seasonal_periods = seasonal_period if seasonal_period and seasonal_period > 1 else None
    seasonal_mode = seasonal if seasonal_periods else None
    trend_mode = trend
    if (trend_mode == "mul" or seasonal_mode == "mul") and np.any(data <= 0):
        if trend_mode == "mul":
            trend_mode = "add"
        if seasonal_mode == "mul":
            seasonal_mode = "add"
    try:
        model = ExponentialSmoothing(
            data,
            trend=trend_mode,
            seasonal=seasonal_mode,
            seasonal_periods=seasonal_periods,
        )
        with warnings.catch_warnings():
            if ConvergenceWarning is not None:
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            fitted = model.fit(optimized=True, use_brute=True)
        base_train = np.asarray(fitted.fittedvalues, float)
        if base_train.size != data.size:
            base_train = np.resize(base_train, data.size)
        base_train = np.where(np.isfinite(base_train), base_train, data)
        base_test = np.asarray(fitted.forecast(horizon), float)
        if base_test.size != horizon:
            base_test = np.resize(base_test, horizon)
        last_val = base_train[-1] if base_train.size else (data[-1] if data.size else 0.0)
        base_test = np.where(np.isfinite(base_test), base_test, last_val)
        return base_train, base_test
    except Exception:
        return None


def _predict_test_windows(model, x_test: np.ndarray) -> np.ndarray:
    if x_test.size == 0:
        return np.zeros(0, dtype=float)
    if tf is None:
        raise RuntimeError("TensorFlow is required to run Helformer predictions")
    x_tensor = tf.convert_to_tensor(x_test, dtype=tf.float32)
    preds = model(x_tensor, training=False)
    return np.asarray(preds.numpy(), float).reshape(-1)


def evaluate_m3_helformer(
    categories: Iterable[str] = ("yearly", "quarterly", "monthly"),
    n_per_cat: int | None = None,
    pick: str = "random",
    seed: int = 42,
    helformer_epochs: int = 40,
    helformer_batch_size: int = 32,
    helformer_lr: float = 8e-4,
    helformer_num_blocks: int = 4,
    helformer_num_heads: int = 4,
    helformer_head_size: int = 56,
    helformer_dropout: float = 0.11266524308240201,
    helformer_units: int = 25,
    helformer_lookback: int = 30,
    helformer_verbose: int = 0,
    helformer_use_hw: bool = True,
    helformer_hw_trend: str = "mul",
    helformer_hw_seasonal: str = "mul",
    csv_dir: Path | None = None,
    tsf_dir: Path | None = None,
    out_prefix: Path | None = None,
    force_rebuild_csv: bool = False,
    visualize: bool = False,
    series_override: Mapping[str, Sequence[str]] | None = None,
) -> pd.DataFrame:
    if tf is None:
        raise RuntimeError("TensorFlow is required to train Helformer")

    csv_dir = Path(csv_dir or settings.m3_csv_dir)
    tsf_dir = Path(tsf_dir or settings.m3_tsf_dir)
    out_dir = Path(out_prefix or (settings.outputs_dir / "m3_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_m3_csv(csv_dir=csv_dir, tsf_dir=tsf_dir, force_rebuild=force_rebuild_csv)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

    rows: List[Dict] = []
    categories = tuple(categories)
    freq_map = {"yearly": "YE", "quarterly": "QE", "monthly": "ME"}
    for cat in _progress(categories, desc="Categories"):
        H = M3_H[cat]
        per = M3_P[cat]
        pairs = load_train_tsts(cat, csv_dir=csv_dir)
        if not pairs:
            print(f"[{cat}] no pairs found in CSV dir: {csv_dir}")
            continue
        selected_list: List[tuple[str, np.ndarray, np.ndarray]]
        if series_override and cat in series_override:
            wanted = set(series_override[cat])
            selected_list = [triple for triple in pairs if triple[0] in wanted]
            if n_per_cat and n_per_cat > 0:
                selected_list = selected_list[: min(len(selected_list), n_per_cat)]
        elif n_per_cat is None or n_per_cat <= 0:
            selected_list = pairs
        elif pick == "first":
            selected_list = pairs[:n_per_cat]
        elif pick == "last":
            selected_list = pairs[-n_per_cat:]
        else:
            count = min(n_per_cat, len(pairs))
            idx = rng.choice(len(pairs), size=count, replace=False)
            selected_list = [pairs[int(i)] for i in idx]

        for sid, y_tr, y_te in _progress(selected_list, desc=f"{cat} series", leave=False):
            forecasts: Dict[str, np.ndarray] = {}
            try:
                tf.keras.backend.clear_session()
                model = create_helformer_model(
                    lookback=int(helformer_lookback),
                    num_blocks=int(helformer_num_blocks),
                    num_heads=int(helformer_num_heads),
                    head_size=int(helformer_head_size),
                    dropout_rate=float(helformer_dropout),
                    units=int(helformer_units),
                )
                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=float(helformer_lr)),
                    loss="mean_squared_error",
                )
                y_tr_arr = np.asarray(y_tr, float).ravel()
                y_te_arr = np.asarray(y_te, float).ravel()
                ratio_train = y_tr_arr.copy()
                ratio_test = y_te_arr.copy()
                base_train = None
                base_test = None
                if helformer_use_hw:
                    baseline = _hw_baseline(
                        y_tr_arr,
                        H,
                        seasonal_period=per,
                        trend=helformer_hw_trend,
                        seasonal=helformer_hw_seasonal,
                    )
                    if baseline is not None:
                        base_train, base_test = baseline
                        denom = np.where(np.abs(base_train) < 1e-8, 1.0, base_train)
                        ratio_train = y_tr_arr / denom
                        denom_te = np.where(np.abs(base_test) < 1e-8, 1.0, base_test)
                        ratio_test = y_te_arr / denom_te
                    else:
                        print(f"[{cat}:{sid}] HW baseline unavailable; using raw series")

                ratio_train = _ensure_min_length(ratio_train, int(helformer_lookback) + 1)
                train_values = ratio_train
                vmin, scale = _minmax_fit(train_values)
                scaled_train = _minmax_transform(train_values, vmin, scale)
                X, Y = _build_xy(scaled_train, int(helformer_lookback))
                if X.shape[0] == 0:
                    raise ValueError("not enough data for Helformer windows")
                model.fit(
                    X,
                    Y,
                    batch_size=int(helformer_batch_size),
                    epochs=int(helformer_epochs),
                    verbose=int(helformer_verbose),
                )
                ratio_test = _ensure_min_length(ratio_test, 1)
                test_values = np.concatenate([ratio_train[-int(helformer_lookback) :], ratio_test])
                scaled_test = _minmax_transform(test_values, vmin, scale)
                x_test = _build_x(scaled_test, int(helformer_lookback))
                pred_ratio_scaled = _predict_test_windows(model, x_test)
                pred_ratio = _minmax_inverse(pred_ratio_scaled, vmin, scale)
                if base_test is not None:
                    pred = pred_ratio * base_test
                else:
                    pred = pred_ratio
                forecasts["Helformer"] = pred[:H]
            except Exception as exc:
                print(f"[{cat}:{sid}] Helformer failed: {exc}")
            finally:
                try:
                    tf.keras.backend.clear_session()
                except Exception:
                    pass
            try:
                forecasts["ARIMA"] = arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[{cat}:{sid}] ARIMA failed: {exc}")
            try:
                forecasts["ARIMA_auto"] = auto_arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[{cat}:{sid}] ARIMA_auto failed: {exc}")
            try:
                forecasts["ETS"] = ets_forecast(y_tr, H, seasonal_periods=per)
            except Exception as exc:
                print(f"[{cat}:{sid}] ETS failed: {exc}")
            try:
                freq = freq_map.get(cat, "D")
                forecasts["Prophet"] = prophet_forecast(y_tr, H, freq=freq)
            except Exception as exc:
                print(f"[{cat}:{sid}] Prophet failed: {exc}")
            if not forecasts:
                forecasts["Naive"] = seasonal_naive(y_tr, H, per)

            rec = {"category": cat, "series_id": sid}
            for name, pred in forecasts.items():
                key = name.replace(" ", "_")
                rec[f"{key}_sMAPE"] = smape(y_te, pred)
                rec[f"{key}_MAPE"] = mape(y_te, pred)
                rec[f"{key}_RMSE"] = rmse(y_te, pred)
                rec[f"{key}_MSE"] = mse(y_te, pred)
            rows.append(rec)

            title = f"{cat.upper()} {sid} (H={H}, L={helformer_lookback})"
            if visualize:
                series_key = f"{cat}_{sid}"
                save_series_viz_bundle_basic(
                    out_dir=out_dir / "viz",
                    series_key=series_key,
                    title_prefix=title,
                    y_tr=y_tr,
                    y_te=y_te,
                    forecasts=forecasts,
                )
            else:
                save_png = out_dir / f"{cat}_{sid}.png"
                plot_forecast(title, y_tr, y_te, forecasts, save_path=save_png)

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[saved] metrics: {metrics_csv}")
    if not df.empty:
        metric_suffixes = {
            "sMAPE": "_sMAPE",
            "MAPE": "_MAPE",
            "RMSE": "_RMSE",
            "MSE": "_MSE",
        }
        for metric, suffix in metric_suffixes.items():
            cols = [c for c in df.columns if c.endswith(suffix)]
            if not cols:
                continue
            print(f"[{metric}] mean by category")
            print(df.groupby("category")[cols].mean(numeric_only=True).round(3))
            overall = df[cols].mean(numeric_only=True)
            print(f"[{metric} overall]")
            print(overall.round(3))
    else:
        print("No results generated — check CSV/logs.")
    return df


__all__ = ["evaluate_m3_helformer"]
