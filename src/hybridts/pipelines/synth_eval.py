"""Evaluation helpers for synthetic time series experiments.

We generate large synthetic series under several regimes:
- with/without trend
- with/without seasonality
- with low / high noise

The same hybrid pipeline (TimesNet/N-BEATS + MODWT) and classical
baselines (ARIMA, auto-ARIMA, ETS, Prophet) are evaluated on all regimes.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

try:  # pragma: no cover - optional dependency
    import tensorflow as tf
except Exception:  # pragma: no cover - fallback when TF is missing
    tf = None

try:  # pragma: no cover - optional dependency
    import optuna
    from optuna.integration import TFKerasPruningCallback
except Exception:  # pragma: no cover - fallback when Optuna is missing
    optuna = None
    TFKerasPruningCallback = None

try:  # pragma: no cover - optional dependency
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:  # pragma: no cover - fallback when statsmodels is missing
    ExponentialSmoothing = None
try:  # pragma: no cover - optional dependency
    from statsmodels.tools.sm_exceptions import ConvergenceWarning
except Exception:  # pragma: no cover - fallback when statsmodels is missing
    ConvergenceWarning = None

from ..config.settings import settings
from ..data import best_L, mse, mape, plot_forecast, rmse, smape, seasonal_naive
from ..hybrids import HybridPlus, VWHybridMixed
from ..models import (
    arima_forecast,
    auto_arima_forecast,
    create_helformer_model,
    ets_forecast,
    make_model,
    prophet_forecast,
)
from ..training import TrainConfig
from ..viz import save_component_forecast_plot, save_series_viz_bundle, save_series_viz_bundle_basic

MODEL_LABELS = {
    "timesnet": "TimesNet+",
    "nbeats": "N-Beats+",
    "vw_timesnet_ets": "VW + TimesNet + ETS",
    "vw_timesnet_arima_auto": "VW + TimesNet + Arima_auto",
    "vw_nbeats_ets": "VW + N-Beats + ETS",
    "vw_nbeats_arima_auto": "VW + N-Beats + Arima_auto",
}


@dataclass(frozen=True)
class SynthProfile:
    name: str
    trend_slope: float
    season_period: int | None
    season_amp: float
    noise_std: float


PROFILES: Dict[str, SynthProfile] = {
    # No trend, no seasonality, low noise
    "flat_low_noise": SynthProfile(
        name="flat_low_noise",
        trend_slope=0.0,
        season_period=None,
        season_amp=0.0,
        noise_std=0.1,
    ),
    # Trend only, low noise
    "trend_only": SynthProfile(
        name="trend_only",
        trend_slope=0.02,
        season_period=None,
        season_amp=0.0,
        noise_std=0.1,
    ),
    # Seasonality only, low noise
    "season_only": SynthProfile(
        name="season_only",
        trend_slope=0.0,
        season_period=24,
        season_amp=1.0,
        noise_std=0.1,
    ),
    # Trend + seasonality, low noise
    "trend_season": SynthProfile(
        name="trend_season",
        trend_slope=0.02,
        season_period=24,
        season_amp=1.0,
        noise_std=0.1,
    ),
    # Trend + seasonality, high noise
    "trend_season_high_noise": SynthProfile(
        name="trend_season_high_noise",
        trend_slope=0.02,
        season_period=24,
        season_amp=1.0,
        noise_std=0.5,
    ),
    # Seasonality only, high noise
    "season_high_noise": SynthProfile(
        name="season_high_noise",
        trend_slope=0.0,
        season_period=24,
        season_amp=1.0,
        noise_std=0.5,
    ),
}


def _generate_series(
    length: int,
    profile: SynthProfile,
    rng: np.random.Generator,
    base_level: float = 10.0,
) -> np.ndarray:
    t = np.arange(length, dtype=float)
    trend = profile.trend_slope * t
    if profile.season_period and profile.season_period > 1 and profile.season_amp > 0:
        season = profile.season_amp * np.sin(2 * np.pi * t / profile.season_period)
    else:
        season = 0.0
    noise = rng.normal(loc=0.0, scale=profile.noise_std, size=length)
    y = base_level + trend + season + noise
    return y.astype(float)


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


def _run_helformer_optuna(
    X: np.ndarray,
    Y: np.ndarray,
    lookback: int,
    *,
    n_trials: int,
    timeout: float | None,
    validation_split: float,
) -> dict[str, Any]:
    if optuna is None:
        raise RuntimeError("Optuna is required for Helformer tuning")
    if tf is None:
        raise RuntimeError("TensorFlow is required for Helformer tuning")
    if X.size == 0 or Y.size == 0:
        raise ValueError("not enough data for Helformer tuning")
    if not (0.0 < validation_split < 1.0):
        raise ValueError("validation_split must be between 0 and 1")
    n_trials = int(n_trials)
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    def objective(trial: optuna.Trial) -> float:
        learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01, log=True)
        units = trial.suggest_int("units", 20, 50, step=5)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])
        epochs = trial.suggest_int("epochs", 50, 150, step=10)
        num_blocks = trial.suggest_int("num_blocks", 1, 4)
        num_heads = trial.suggest_int("num_heads", 2, 10, step=2)
        head_size = trial.suggest_int("head_size", 8, 64, step=8)

        tf.keras.backend.clear_session()
        try:
            model = create_helformer_model(
                lookback=lookback,
                num_blocks=num_blocks,
                num_heads=num_heads,
                head_size=head_size,
                dropout_rate=dropout_rate,
                units=units,
            )
            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=float(learning_rate)),
                loss="mean_squared_error",
            )
            callbacks = []
            if TFKerasPruningCallback is not None:
                callbacks = [TFKerasPruningCallback(trial, "val_loss")]
            history = model.fit(
                X,
                Y,
                batch_size=int(batch_size),
                epochs=int(epochs),
                validation_split=float(validation_split),
                verbose=0,
                callbacks=callbacks,
            )
            val_loss = history.history.get("val_loss")
            if not val_loss:
                return float("inf")
            return float(min(val_loss))
        finally:
            try:
                tf.keras.backend.clear_session()
            except Exception:
                pass

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, timeout=timeout)
    return dict(study.best_params)


def _base_factory(name: str, params: Mapping[str, Any] | None = None):
    def _fn(cfg: TrainConfig):
        return make_model(name, cfg, params=params)

    return _fn


def _effective_model_params(
    model_name: str,
    *,
    base_model_name: str | None = None,
    seasonal_period: int | None,
    model_params: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    base_name = (base_model_name or model_name).lower()
    primary = model_name.lower()
    params_raw = None
    if model_params:
        params_raw = model_params.get(primary)
        if params_raw is None and base_name != primary:
            params_raw = model_params.get(base_name)
    params = dict(params_raw) if params_raw else None
    if base_name == "nbeats" and (seasonal_period is None or seasonal_period <= 1):
        if params is None:
            params = {}
        params.setdefault("use_seasonality", False)
    return params


def evaluate_synth_hybrids(
    profiles: Iterable[str] | None = None,
    n_per_profile: int = 32,
    length: int = 400,
    horizon: int = 24,
    epochs: int = 8,
    base_models: Iterable[str] | None = None,
    seed: int = 42,
    out_prefix: Path | None = None,
    wavelet: str = "db4",
    level: int = 1,
    boundary: str = "wrap",
    plot: bool = True,
    visualize: bool = False,
    use_helformer: bool = False,
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
    helformer_optuna: bool = False,
    helformer_optuna_trials: int = 50,
    helformer_optuna_timeout: float | None = None,
    helformer_optuna_validation_split: float = 0.2,
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    """Run hybrid + baseline models on synthetic series."""
    if base_models is None:
        base_models = ("timesnet", "nbeats")
    base_models = tuple((m.lower() for m in base_models))
    label_map = {name: MODEL_LABELS.get(name, name.title() + "+") for name in base_models}
    hybrid_models = base_models

    use_profiles: Tuple[SynthProfile, ...]
    if profiles is None:
        use_profiles = tuple(PROFILES.values())
    else:
        use_profiles = tuple(PROFILES[p] for p in profiles if p in PROFILES)

    out_dir = Path(out_prefix or (settings.outputs_dir / "synth_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_helformer:
        if tf is None:
            raise RuntimeError("TensorFlow is required to train Helformer")
        tf.random.set_seed(seed)
        # Helformer is always used with HW decomposition in synthetic experiments.
        helformer_use_hw = True
        if helformer_optuna and optuna is None:
            raise RuntimeError("Optuna is required when helformer_optuna is enabled")

    rows: List[Dict] = []

    for profile in use_profiles:
        per = profile.season_period or 1
        for idx in range(n_per_profile):
            series_id = f"{profile.name}_{idx+1}"
            y = _generate_series(length=length, profile=profile, rng=rng)
            if y.size <= horizon + 8:
                continue
            y_tr = y[:-horizon]
            y_te = y[-horizon:]

            L = best_L(y_tr, horizon, per)
            cfg = TrainConfig(
                lookback=L,
                horizon=horizon,
                epochs=epochs,
                batch_size=64,
                lr=5e-4,
                weight_decay=1e-4,
                clip=1.0,
            )

            forecasts: Dict[str, np.ndarray] = {}
            component_forecasts: Dict[str, Dict[str, np.ndarray]] = {}
            # Hybrid neural models (TimesNet / N-BEATS)
            for model_name in hybrid_models:
                label = label_map[model_name]
                try:
                    per_eff = per if per and per > 1 else None
                    if model_name in {"timesnet", "nbeats"}:
                        params = _effective_model_params(
                            model_name,
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = HybridPlus(
                            base_model_fn=_base_factory(model_name, params=params),
                            cfg=cfg,
                            wavelet=wavelet,
                            level=level,
                            boundary=boundary,
                            seasonal_period=per_eff,
                        ).fit(y_tr)
                    elif model_name in {"vw_timesnet_ets", "vw_timesnet_arima_auto"}:
                        detail = "ets" if model_name.endswith("_ets") else "arima_auto"
                        aj_params = _effective_model_params(
                            model_name,
                            base_model_name="timesnet",
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = VWHybridMixed(
                            aj_model_fn=_base_factory("timesnet", params=aj_params),
                            detail_method=detail,
                            cfg=cfg,
                            wavelet=wavelet,
                            level=level,
                            seasonal_period=per_eff,
                        ).fit(y_tr)
                    elif model_name in {"vw_nbeats_ets", "vw_nbeats_arima_auto"}:
                        detail = "ets" if model_name.endswith("_ets") else "arima_auto"
                        aj_params = _effective_model_params(
                            model_name,
                            base_model_name="nbeats",
                            seasonal_period=per_eff,
                            model_params=model_params,
                        )
                        model = VWHybridMixed(
                            aj_model_fn=_base_factory("nbeats", params=aj_params),
                            detail_method=detail,
                            cfg=cfg,
                            wavelet=wavelet,
                            level=level,
                            seasonal_period=per_eff,
                        ).fit(y_tr)
                    else:
                        raise ValueError(f"Unknown hybrid model '{model_name}'")
                    forecasts[label] = model.forecast(y_tr)
                    if hasattr(model, "forecast_components"):
                        component_forecasts[label] = model.forecast_components(y_tr)  # type: ignore[assignment]
                except Exception as exc:
                    print(f"[{profile.name}:{series_id}] {label} failed: {exc}")

            if use_helformer:
                try:
                    tf.keras.backend.clear_session()
                    y_tr_arr = np.asarray(y_tr, float).ravel()
                    y_te_arr = np.asarray(y_te, float).ravel()
                    ratio_train = y_tr_arr.copy()
                    ratio_test = y_te_arr.copy()
                    base_train = None
                    base_test = None
                    if helformer_use_hw:
                        baseline = _hw_baseline(
                            y_tr_arr,
                            horizon,
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
                            print(
                                f"[{profile.name}:{series_id}] Helformer HW baseline failed; "
                                "using raw series."
                            )
                    ratio_train = _ensure_min_length(ratio_train, int(helformer_lookback) + 1)
                    vmin, scale = _minmax_fit(ratio_train)
                    scaled_train = _minmax_transform(ratio_train, vmin, scale)
                    X, Y = _build_xy(scaled_train, int(helformer_lookback))
                    if X.shape[0] == 0:
                        raise ValueError("not enough data for Helformer windows")
                    best_params = None
                    if helformer_optuna:
                        try:
                            best_params = _run_helformer_optuna(
                                X,
                                Y,
                                lookback=int(helformer_lookback),
                                n_trials=int(helformer_optuna_trials),
                                timeout=helformer_optuna_timeout,
                                validation_split=float(helformer_optuna_validation_split),
                            )
                            print(
                                f"[{profile.name}:{series_id}] Helformer Optuna best params: "
                                f"{best_params}"
                            )
                        except Exception as exc:
                            print(f"[{profile.name}:{series_id}] Helformer Optuna failed: {exc}")
                            best_params = None

                    if best_params:
                        lr = float(best_params["learning_rate"])
                        units = int(best_params["units"])
                        dropout_rate = float(best_params["dropout_rate"])
                        batch_size = int(best_params["batch_size"])
                        epochs_fit = int(best_params["epochs"])
                        num_blocks = int(best_params["num_blocks"])
                        num_heads = int(best_params["num_heads"])
                        head_size = int(best_params["head_size"])
                    else:
                        lr = float(helformer_lr)
                        units = int(helformer_units)
                        dropout_rate = float(helformer_dropout)
                        batch_size = int(helformer_batch_size)
                        epochs_fit = int(helformer_epochs)
                        num_blocks = int(helformer_num_blocks)
                        num_heads = int(helformer_num_heads)
                        head_size = int(helformer_head_size)

                    model = create_helformer_model(
                        lookback=int(helformer_lookback),
                        num_blocks=num_blocks,
                        num_heads=num_heads,
                        head_size=head_size,
                        dropout_rate=dropout_rate,
                        units=units,
                    )
                    model.compile(
                        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                        loss="mean_squared_error",
                    )
                    model.fit(
                        X,
                        Y,
                        batch_size=batch_size,
                        epochs=epochs_fit,
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
                    forecasts["Helformer"] = pred[:horizon]
                except Exception as exc:
                    print(f"[{profile.name}:{series_id}] Helformer failed: {exc}")
                finally:
                    try:
                        tf.keras.backend.clear_session()
                    except Exception:
                        pass

            # Classical baselines
            try:
                forecasts["ARIMA"] = arima_forecast(y_tr, horizon)
            except Exception as exc:
                print(f"[{profile.name}:{series_id}] ARIMA failed: {exc}")
            try:
                forecasts["ARIMA_auto"] = auto_arima_forecast(y_tr, horizon)
            except Exception as exc:
                print(f"[{profile.name}:{series_id}] ARIMA_auto failed: {exc}")
            try:
                forecasts["ETS"] = ets_forecast(y_tr, horizon, seasonal_periods=per)
            except Exception as exc:
                print(f"[{profile.name}:{series_id}] ETS failed: {exc}")
            try:
                # Use month-end ('ME') for synthetic seasonal series,
                # daily for non-seasonal as a neutral choice
                freq = "ME" if profile.season_period else "D"
                forecasts["Prophet"] = prophet_forecast(y_tr, horizon, freq=freq)
            except Exception as exc:
                    print(f"[{profile.name}:{series_id}] Prophet failed: {exc}")

            if not forecasts:
                # Fallback: seasonal naive or last-value persistence
                naive = seasonal_naive(y_tr, horizon, per)
                for model_name in hybrid_models:
                    label = label_map[model_name]
                    forecasts[label] = naive.copy()

            rec: Dict[str, float | str] = {
                "profile": profile.name,
                "series_id": series_id,
            }
            for name, pred in forecasts.items():
                key = name.replace(" ", "_")
                rec[f"{key}_sMAPE"] = smape(y_te, pred)
                rec[f"{key}_MAPE"] = mape(y_te, pred)
                rec[f"{key}_RMSE"] = rmse(y_te, pred)
                rec[f"{key}_MSE"] = mse(y_te, pred)
            rows.append(rec)

            title = f"{profile.name} {series_id} (H={horizon}, L={L})"
            if visualize:
                if hybrid_models:
                    save_series_viz_bundle(
                        out_dir=out_dir / "viz",
                        series_key=series_id,
                        title_prefix=title,
                        y_tr=y_tr,
                        y_te=y_te,
                        forecasts=forecasts,
                        wavelet=wavelet,
                        level=level,
                        boundary=boundary,
                        component_forecasts=component_forecasts if component_forecasts else None,
                    )
                else:
                    save_series_viz_bundle_basic(
                        out_dir=out_dir / "viz",
                        series_key=series_id,
                        title_prefix=title,
                        y_tr=y_tr,
                        y_te=y_te,
                        forecasts=forecasts,
                    )
            elif plot:
                save_png = out_dir / f"{series_id}.png"
                plot_forecast(title, y_tr, y_te, forecasts, save_path=save_png)
                if component_forecasts:
                    save_component_forecast_plot(
                        y_tr=y_tr,
                        y_te=y_te,
                        component_forecasts=component_forecasts,
                        wavelet=wavelet,
                        level=level,
                        boundary=boundary,
                        title=f"{title} component forecasts",
                        save_path=out_dir / f"{series_id}_components.png",
                    )

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[saved] synthetic metrics: {metrics_csv}")

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
            print(f"[{metric}] mean by profile")
            print(df.groupby("profile")[cols].mean(numeric_only=True).round(3))
            overall = df[cols].mean(numeric_only=True)
            print(f"[{metric} overall]")
            print(overall.round(3))
    else:
        print("No synthetic results generated; check settings.")

    return df


__all__ = ["evaluate_synth_hybrids", "SynthProfile", "PROFILES"]
