"""Helformer inference helpers (TensorFlow/Keras)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Tuple

import numpy as np
import json

try:  # pragma: no cover - optional dependency
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except Exception:  # pragma: no cover - fallback when TF is missing
    tf = None
    load_model = None

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover - fallback when statsmodels is missing
    ExponentialSmoothing = None

try:  # pragma: no cover - optional dependency
    import h5py
except Exception:  # pragma: no cover - fallback when h5py is missing
    h5py = None


def mish(x):
    if tf is None:  # pragma: no cover - TF required for inference
        raise RuntimeError("TensorFlow is required for mish activation")
    return x * tf.math.tanh(tf.math.softplus(x))


def _extract_helformer_config(weights_path: Path) -> dict:
    if h5py is None:
        raise RuntimeError("h5py is required to inspect Helformer weights")
    with h5py.File(weights_path, "r") as handle:
        model_config = handle.attrs.get("model_config")
    if model_config is None:
        raise RuntimeError("model_config not found in Helformer weights")
    if isinstance(model_config, (bytes, bytearray)):
        model_config = model_config.decode("utf-8")
    cfg = json.loads(model_config)
    layers = cfg.get("config", {}).get("layers", [])
    input_layer = next((layer for layer in layers if layer.get("class_name") == "InputLayer"), None)
    if input_layer is None:
        raise RuntimeError("InputLayer not found in Helformer config")
    batch_shape = input_layer.get("config", {}).get("batch_shape", [None, None, None])
    lookback = int(batch_shape[1]) if len(batch_shape) > 1 and batch_shape[1] is not None else 30

    mha_layers = [layer for layer in layers if layer.get("class_name") == "MultiHeadAttention"]
    if not mha_layers:
        raise RuntimeError("MultiHeadAttention layers not found in Helformer config")
    mha_cfg = mha_layers[0].get("config", {})
    num_blocks = len(mha_layers)
    num_heads = int(mha_cfg.get("num_heads", 4))
    head_size = int(mha_cfg.get("key_dim", 32))
    dropout_rate = float(mha_cfg.get("dropout", 0.0))

    lstm_layer = next((layer for layer in layers if layer.get("class_name") == "LSTM"), None)
    if lstm_layer is None:
        raise RuntimeError("LSTM layer not found in Helformer config")
    units = int(lstm_layer.get("config", {}).get("units", 32))

    return {
        "lookback": lookback,
        "num_blocks": num_blocks,
        "num_heads": num_heads,
        "head_size": head_size,
        "dropout_rate": dropout_rate,
        "units": units,
    }


def create_helformer_model(
    *,
    lookback: int,
    num_blocks: int,
    num_heads: int,
    head_size: int,
    dropout_rate: float,
    units: int,
):
    if tf is None:  # pragma: no cover - TF required for inference
        raise RuntimeError("TensorFlow is required for Helformer inference")
    lookback = int(lookback)
    num_blocks = int(num_blocks)
    num_heads = int(num_heads)
    head_size = int(head_size)
    dropout_rate = float(dropout_rate)
    units = int(units)

    inputs = tf.keras.layers.Input(shape=(lookback, 1))
    x = inputs
    for _ in range(num_blocks):
        x_norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads,
            key_dim=head_size,
            dropout=dropout_rate,
        )(x_norm1, x_norm1)
        x = tf.keras.layers.Add()([x, attention])
        x_norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x)
        x = tf.keras.layers.Add()([x, x_norm2])
    x = tf.keras.layers.LSTM(units, activation=mish, return_sequences=False)(x)
    outputs = tf.keras.layers.Dense(1)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def load_helformer_model(weights_path: Path | str):
    if load_model is None:  # pragma: no cover - TF required for inference
        raise RuntimeError("TensorFlow is required to load the Helformer model")
    weights_path = Path(weights_path)
    if not weights_path.exists():
        raise FileNotFoundError(f"Helformer weights not found: {weights_path}")
    try:
        return load_model(weights_path, custom_objects={"mish": mish}, compile=False)
    except Exception:
        cfg = _extract_helformer_config(weights_path)
        model = create_helformer_model(**cfg)
        model.load_weights(weights_path)
        return model


def _minmax_fit(series: np.ndarray) -> Tuple[float, float]:
    if series.size == 0:
        return 0.0, 1.0
    vmin = float(np.nanmin(series))
    vmax = float(np.nanmax(series))
    scale = vmax - vmin
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        scale = 1.0
    return vmin, float(scale)


def _minmax_transform(series: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return (series - vmin) / scale


def _minmax_inverse(series: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return series * scale + vmin


def _prepare_window(history: Iterable[float], lookback: int) -> np.ndarray:
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    hist = list(history)
    if not hist:
        return np.zeros(lookback, dtype=np.float32)
    if len(hist) >= lookback:
        window = hist[-lookback:]
    else:
        pad_val = hist[-1]
        window = [pad_val] * (lookback - len(hist)) + hist
    return np.asarray(window, dtype=np.float32)


def _holt_winters_baseline(
    y_tr: np.ndarray,
    horizon: int,
    seasonal_period: Optional[int],
    trend: str,
    seasonal: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ExponentialSmoothing is None:
        return None
    data = np.asarray(y_tr, float).ravel()
    if data.size < 4:
        return None
    seasonal_periods = seasonal_period if seasonal_period and seasonal_period > 1 else None
    seasonal = seasonal if seasonal_periods else None
    trend_mode = trend
    seasonal_mode = seasonal
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


def _holt_winters_fit(
    y_tr: np.ndarray,
    *,
    seasonal_period: Optional[int],
    trend: str,
    seasonal: str,
) -> Optional[Tuple[Any, str, str, Optional[int]]]:
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
        fitted = model.fit(optimized=True, use_brute=True)
    except Exception:
        return None
    return fitted, trend_mode, (seasonal_mode or "none"), seasonal_periods


def helformer_forecast(
    y_tr: Iterable[float],
    horizon: int,
    *,
    model,
    lookback: int = 30,
    seasonal_period: Optional[int] = None,
    use_hw: bool = False,
    hw_trend: str = "mul",
    hw_seasonal: str = "mul",
) -> np.ndarray:
    y_tr = np.asarray(list(y_tr), dtype=float).ravel()
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if y_tr.size == 0:
        return np.zeros(horizon, dtype=float)

    series = y_tr
    base_test = None
    if use_hw:
        baseline = _holt_winters_baseline(
            y_tr=y_tr,
            horizon=horizon,
            seasonal_period=seasonal_period,
            trend=hw_trend,
            seasonal=hw_seasonal,
        )
        if baseline is not None:
            base_train, base_test = baseline
            denom = np.where(np.abs(base_train) < 1e-8, 1.0, base_train)
            series = y_tr / denom
        else:
            use_hw = False

    series = np.where(np.isfinite(series), series, 0.0)
    vmin, scale = _minmax_fit(series)
    scaled_series = _minmax_transform(series, vmin, scale)
    history = scaled_series.tolist()

    preds_scaled: list[float] = []
    for _ in range(horizon):
        window = _prepare_window(history, lookback).reshape(1, lookback, 1)
        pred = model.predict(window, verbose=0)
        pred_val = float(np.ravel(pred)[0])
        if not np.isfinite(pred_val):
            pred_val = history[-1] if history else 0.0
        preds_scaled.append(pred_val)
        history.append(pred_val)

    preds = _minmax_inverse(np.asarray(preds_scaled, float), vmin, scale)
    if use_hw and base_test is not None:
        base_test = np.asarray(base_test, float).ravel()
        if base_test.size < horizon:
            pad_val = base_test[-1] if base_test.size else 1.0
            base_test = np.pad(base_test, (0, horizon - base_test.size), constant_values=pad_val)
        elif base_test.size > horizon:
            base_test = base_test[:horizon]
        preds = preds * base_test
    return preds


@dataclass(frozen=True)
class HelformerHWComponents:
    base_train: np.ndarray
    base_test: np.ndarray
    ratio_train: np.ndarray
    ratio_test: np.ndarray
    level: Optional[np.ndarray]
    trend: Optional[np.ndarray]
    season: Optional[np.ndarray]
    trend_mode: str
    seasonal_mode: str
    seasonal_periods: Optional[int]


def helformer_hw_decompose(
    y_tr: Iterable[float],
    y_te: Iterable[float],
    *,
    seasonal_period: Optional[int] = None,
    hw_trend: str = "mul",
    hw_seasonal: str = "mul",
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Compute the Holt-Winters ratio decomposition used by Helformer.

    Returns base (train fitted, test forecast) and multiplicative ratios
    y/base for both train and test segments. Returns None if the baseline
    cannot be fit (e.g., statsmodels missing or convergence issues).
    """
    y_tr_arr = np.asarray(list(y_tr), dtype=float).ravel()
    y_te_arr = np.asarray(list(y_te), dtype=float).ravel()
    horizon = int(y_te_arr.size)
    baseline = _holt_winters_baseline(
        y_tr=y_tr_arr,
        horizon=horizon,
        seasonal_period=seasonal_period,
        trend=hw_trend,
        seasonal=hw_seasonal,
    )
    if baseline is None:
        return None
    base_train, base_test = baseline
    denom_tr = np.where(np.abs(base_train) < 1e-8, 1.0, base_train)
    ratio_train = y_tr_arr / denom_tr
    denom_te = np.where(np.abs(base_test) < 1e-8, 1.0, base_test)
    ratio_test = y_te_arr / denom_te
    ratio_train = np.where(np.isfinite(ratio_train), ratio_train, 1.0)
    ratio_test = np.where(np.isfinite(ratio_test), ratio_test, 1.0)
    return base_train, base_test, ratio_train, ratio_test


def helformer_hw_components(
    y_tr: Iterable[float],
    y_te: Iterable[float],
    *,
    seasonal_period: Optional[int] = None,
    hw_trend: str = "mul",
    hw_seasonal: str = "mul",
) -> Optional[HelformerHWComponents]:
    y_tr_arr = np.asarray(list(y_tr), dtype=float).ravel()
    y_te_arr = np.asarray(list(y_te), dtype=float).ravel()
    horizon = int(y_te_arr.size)
    fit = _holt_winters_fit(
        y_tr_arr,
        seasonal_period=seasonal_period,
        trend=hw_trend,
        seasonal=hw_seasonal,
    )
    if fit is None:
        return None
    fitted, trend_mode, seasonal_mode, seasonal_periods = fit

    base_train = np.asarray(fitted.fittedvalues, float)
    if base_train.size != y_tr_arr.size:
        base_train = np.resize(base_train, y_tr_arr.size)
    base_train = np.where(np.isfinite(base_train), base_train, y_tr_arr)

    base_test = np.asarray(fitted.forecast(horizon), float) if horizon > 0 else np.zeros(0, dtype=float)
    if base_test.size != horizon:
        base_test = np.resize(base_test, horizon)
    last_val = base_train[-1] if base_train.size else (y_tr_arr[-1] if y_tr_arr.size else 0.0)
    base_test = np.where(np.isfinite(base_test), base_test, last_val)

    denom_tr = np.where(np.abs(base_train) < 1e-8, 1.0, base_train)
    ratio_train = y_tr_arr / denom_tr
    denom_te = np.where(np.abs(base_test) < 1e-8, 1.0, base_test)
    ratio_test = y_te_arr / denom_te
    ratio_train = np.where(np.isfinite(ratio_train), ratio_train, 1.0)
    ratio_test = np.where(np.isfinite(ratio_test), ratio_test, 1.0)

    level = np.asarray(getattr(fitted, "level", None), float) if hasattr(fitted, "level") else None
    trend = np.asarray(getattr(fitted, "trend", None), float) if hasattr(fitted, "trend") else None
    season = np.asarray(getattr(fitted, "season", None), float) if hasattr(fitted, "season") else None

    if level is not None and level.size != y_tr_arr.size:
        level = np.resize(level, y_tr_arr.size)
    if trend is not None and trend.size != y_tr_arr.size:
        trend = np.resize(trend, y_tr_arr.size)
    if season is not None and season.size != y_tr_arr.size:
        season = np.resize(season, y_tr_arr.size)

    return HelformerHWComponents(
        base_train=base_train,
        base_test=base_test,
        ratio_train=ratio_train,
        ratio_test=ratio_test,
        level=level,
        trend=trend,
        season=season,
        trend_mode=trend_mode,
        seasonal_mode=seasonal_mode,
        seasonal_periods=seasonal_periods,
    )


__all__ = [
    "create_helformer_model",
    "helformer_forecast",
    "helformer_hw_decompose",
    "helformer_hw_components",
    "load_helformer_model",
]
