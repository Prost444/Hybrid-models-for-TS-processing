"""Evaluation helpers for M4 hybrid experiments."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence
import warnings

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore", message=".*np.object.*", category=FutureWarning)

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - best-effort fallback
    tqdm = None

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
from ..data import (
    M4_H,
    M4_P,
    best_L,
    ensure_m4_csv,
    load_m4_train_test,
    plot_forecast,
    plot_forecast_test_only,
    seasonal_naive,
    smape,
    mape,
    mse,
    rmse,
)
from ..hybrids import HybridComponent, HybridPlus, VWHybridMixed, build_global_hybrid_components
from ..hybrids.modwt_hybrid import modwt_decompose_with_boundary
from ..models import (
    arima_forecast,
    auto_arima_forecast,
    create_helformer_model,
    ets_forecast,
    make_model,
    prophet_forecast,
)
from ..training import TrainConfig
from ..viz import (
    save_component_forecast_plot,
    save_component_forecast_test_only_plot,
    save_series_viz_bundle,
    save_series_viz_bundle_basic,
    save_series_viz_bundle_helformer_hw,
    save_simulation_full_plot,
    save_simulation_train_plot,
)


def _fitted_one_step_series(
    component: HybridComponent,
    comp_tr: np.ndarray,
    *,
    device: str,
    total_len: int | None = None,
    mode: str = "rollout",
) -> np.ndarray:
    comp_tr = np.asarray(comp_tr, float).ravel()
    n = int(comp_tr.size)
    if n == 0:
        return comp_tr
    if total_len is None:
        total_len = n
    total_len = int(max(1, total_len))
    lookback = int(component.lookback or 0)
    if component.model is None or lookback <= 0 or n <= 2:
        out = np.empty(total_len, dtype=float)
        init_len = min(n, total_len)
        out[:init_len] = comp_tr[:init_len]
        for t in range(1, init_len):
            out[t] = out[t - 1]
        for t in range(init_len, total_len):
            out[t] = out[t - 1]
        return out

    if bool(getattr(component, "per_series_scaling", False)):
        mu = float(np.mean(comp_tr))
        sd = float(np.std(comp_tr) + 1e-8)
    else:
        mu = float(component.mu)
        sd = float(component.sd + 1e-8)

    lookback = min(lookback, max(1, n - 1))
    out = np.empty(total_len, dtype=float)
    init_len = min(lookback, n, total_len)
    out[:init_len] = comp_tr[:init_len]
    if init_len < lookback:
        # pad seed if series shorter than lookback
        for t in range(init_len, min(lookback, total_len)):
            out[t] = out[t - 1]
        init_len = min(lookback, total_len)
    model = component.model
    model.eval()
    mode = str(mode or "rollout").lower()
    for t in range(init_len, total_len):
        if mode == "fitted" and t < n:
            window = comp_tr[max(0, t - lookback) : t]
            if window.size < lookback:
                pad = np.repeat(window[0] if window.size else out[0], lookback - window.size)
                window = np.concatenate([pad, window])
        else:
            window = out[t - lookback : t]
        xb = ((window - mu) / sd).astype(np.float32).reshape(1, 1, -1)
        with torch.no_grad():
            pred = model(torch.from_numpy(xb).to(device)).detach().cpu().numpy().ravel()
        if pred.size <= 0:
            out[t] = out[t - 1]
        else:
            out[t] = float(pred[0]) * sd + mu
    return out


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
            warnings.filterwarnings("ignore", category=RuntimeWarning)
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


def _predict_rollout(
    model,
    history: list[float],
    lookback: int,
    horizon: int,
) -> np.ndarray:
    if horizon <= 0:
        return np.zeros(0, dtype=float)
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    hist = list(history)
    if not hist:
        hist = [0.0] * lookback
    preds: list[float] = []
    for _ in range(horizon):
        if len(hist) >= lookback:
            window = hist[-lookback:]
        else:
            pad_val = hist[0]
            window = [pad_val] * (lookback - len(hist)) + hist
        x_step = np.asarray(window, np.float32).reshape(1, lookback, 1)
        pred = _predict_test_windows(model, x_step)
        pred_val = float(pred[-1]) if pred.size else (hist[-1] if hist else 0.0)
        if not np.isfinite(pred_val):
            pred_val = hist[-1] if hist else 0.0
        preds.append(pred_val)
        hist.append(pred_val)
    return np.asarray(preds, float)


def _predict_rollout_full(
    model,
    series_scaled: np.ndarray,
    lookback: int,
    total_len: int,
) -> np.ndarray:
    if total_len <= 0:
        return np.zeros(0, dtype=float)
    if lookback <= 0:
        raise ValueError("lookback must be positive")
    series_scaled = np.asarray(series_scaled, float).ravel().tolist()
    if not series_scaled:
        seed = [0.0] * lookback
    elif len(series_scaled) >= lookback:
        seed = series_scaled[:lookback]
    else:
        pad_val = series_scaled[0]
        seed = series_scaled + [pad_val] * (lookback - len(series_scaled))
    if total_len <= lookback:
        return np.asarray(seed[:total_len], float)
    preds = _predict_rollout(model, seed, lookback, total_len - lookback)
    return np.asarray(seed + preds.tolist(), float)


def _ratio_std_score(
    y_tr: np.ndarray,
    horizon: int,
    seasonal_period: int | None,
    trend: str,
    seasonal: str,
) -> float:
    baseline = _hw_baseline(
        y_tr,
        horizon,
        seasonal_period=seasonal_period,
        trend=trend,
        seasonal=seasonal,
    )
    if baseline is None:
        return float("nan")
    base_train, _ = baseline
    denom = np.where(np.abs(base_train) < 1e-8, 1.0, base_train)
    ratio = np.asarray(y_tr, float) / denom
    if ratio.size == 0:
        return float("nan")
    return float(np.nanstd(ratio))


def _select_by_ratio_std(
    pairs: Sequence[tuple[str, np.ndarray, np.ndarray]],
    *,
    horizon: int,
    seasonal_period: int | None,
    trend: str,
    seasonal: str,
    min_std: float | None,
    top_k: int | None,
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    scored: list[tuple[float, tuple[str, np.ndarray, np.ndarray]]] = []
    for sid, y_tr, y_te in pairs:
        score = _ratio_std_score(
            y_tr,
            horizon,
            seasonal_period=seasonal_period,
            trend=trend,
            seasonal=seasonal,
        )
        if not np.isfinite(score):
            continue
        if min_std is not None and score < float(min_std):
            continue
        scored.append((score, (sid, y_tr, y_te)))
    if not scored:
        return []
    scored.sort(key=lambda item: item[0], reverse=True)
    if top_k is not None and top_k > 0:
        scored = scored[:top_k]
    picked = [item[1] for item in scored]
    sample = ", ".join(f"{sid} (std={score:.4f})" for score, (sid, _, _) in scored[:3])
    print(f"[m4] selected by ratio std: {sample}")
    return picked


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


MODEL_LABELS = {
    "timesnet": "TimesNet+",
    "nbeats": "N-BEATS Full",
    "vw_timesnet_ets": "VW + TimesNet + ETS",
    "vw_timesnet_arima_auto": "VW + TimesNet + Arima_auto",
    "vw_nbeats_ets": "VW + N-Beats + ETS",
    "vw_nbeats_arima_auto": "VW + N-Beats + Arima_auto",
}


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


def _cat_level(cat: str, base_level: int) -> int:
    base_level = int(base_level)
    suggested = {
        "yearly": base_level,
        "quarterly": max(base_level, 2),
        "monthly": max(base_level, 3),
        "weekly": max(base_level, 3),
        "daily": max(base_level, 4),
        "hourly": max(base_level, 5),
    }
    return int(suggested.get(str(cat).lower(), base_level))


def evaluate_m4_hybrids(
    categories: Iterable[str] = ("yearly", "quarterly", "monthly", "weekly", "daily", "hourly"),
    n_per_cat: int | None = None,
    pick: str = "random",
    seed: int = 42,
    epochs: int = 8,
    base_models: Iterable[str] | None = None,
    csv_dir: Path | None = None,
    raw_dir: Path | None = None,
    out_prefix: Path | None = None,
    wavelet: str = "db4",
    level: int = 1,
    boundary: str = "wrap",
    force_rebuild_csv: bool = False,
    force_rebuild_global_components: bool = False,
    series_override: Mapping[str, Sequence[str]] | None = None,
    visualize: bool = False,
    plot_test_only: bool = True,
    use_full_modwt_components: bool = True,
    simulate_full_series: bool = False,
    simulation_mode: str = "rollout",
    simulation_train_only_plot: bool = False,
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
    helformer_pick_by_ratio_std: bool = False,
    helformer_ratio_std_min: float | None = None,
    helformer_forecast_mode: str = "rollout",
    model_params: Mapping[str, Mapping[str, Any]] | None = None,
) -> pd.DataFrame:
    if base_models is None:
        base_models = ("timesnet", "nbeats")
    base_models = tuple((m.lower() for m in base_models))
    label_map = {name: MODEL_LABELS.get(name, f"{name.title()}+") for name in base_models}
    hybrid_models = base_models

    csv_dir = Path(csv_dir or settings.m4_csv_dir)
    out_dir = Path(out_prefix or (settings.outputs_dir / "m4_eval"))
    out_dir.mkdir(parents=True, exist_ok=True)

    ensure_m4_csv(
        csv_dir=csv_dir,
        raw_dir=raw_dir,
        categories=categories,
        force_rebuild=force_rebuild_csv,
    )

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_helformer:
        if tf is None:
            raise RuntimeError("TensorFlow is required to train Helformer")
        tf.random.set_seed(seed)
        # Helformer is always used with HW decomposition in M4 experiments.
        helformer_use_hw = True
        if helformer_optuna and optuna is None:
            raise RuntimeError("Optuna is required when helformer_optuna is enabled")

    rows: List[Dict] = []
    categories = tuple(categories)
    freq_map = {
        "yearly": "YE",
        "quarterly": "QE",
        "monthly": "ME",
        "weekly": "W",
        "daily": "D",
        "hourly": "H",
    }
    for cat in _progress(categories, desc="Categories"):
        cat = str(cat).lower()
        if cat not in M4_H:
            print(f"[m4:{cat}] unknown category; skipping")
            continue
        H = M4_H[cat]
        per = M4_P[cat]
        cat_level = _cat_level(cat, level)
        pairs = load_m4_train_test(cat, csv_dir=csv_dir)
        if not pairs:
            print(f"[m4:{cat}] no pairs found in CSV dir: {csv_dir}")
            continue

        selected_list: List[tuple[str, np.ndarray, np.ndarray]] | None = None
        if series_override and cat in series_override:
            wanted = set(series_override[cat])
            selected_list = [triple for triple in pairs if triple[0] in wanted]
            if n_per_cat and n_per_cat > 0:
                selected_list = selected_list[: min(len(selected_list), n_per_cat)]
        elif use_helformer and helformer_pick_by_ratio_std:
            selected_list = _select_by_ratio_std(
                pairs,
                horizon=H,
                seasonal_period=per,
                trend=helformer_hw_trend,
                seasonal=helformer_hw_seasonal,
                min_std=helformer_ratio_std_min,
                top_k=n_per_cat if n_per_cat and n_per_cat > 0 else None,
            )
            if not selected_list:
                print(f"[m4:{cat}] ratio-std selection empty; falling back to pick={pick}")
                selected_list = None

        if selected_list is None:
            if n_per_cat is None or n_per_cat <= 0:
                selected_list = list(pairs)
            elif pick == "first":
                selected_list = list(pairs[:n_per_cat])
            elif pick == "last":
                selected_list = list(pairs[-n_per_cat:])
            else:
                count = min(n_per_cat, len(pairs))
                idx = rng.choice(len(pairs), size=count, replace=False)
                selected_list = [pairs[int(i)] for i in idx]

        # Global hybrid components for neural base models (TimesNet / N-BEATS).
        # For M4 we optionally use MODWT components computed on the full series
        # (train+test) to avoid boundary mismatch; in that mode, global pretraining
        # is disabled to keep training consistent.
        global_hybrid_components: Dict[str, List[HybridComponent]] = {}
        if not use_full_modwt_components:
            for model_name in hybrid_models:
                if model_name not in {"timesnet", "nbeats"}:
                    continue
                params = _effective_model_params(
                    model_name,
                    seasonal_period=(per if per and per > 1 else None),
                    model_params=model_params,
                )
                hybrid_ckpt = out_dir / f"{cat}_{model_name}_hybrid_global.pt"
                if hybrid_ckpt.exists() and not force_rebuild_global_components:
                    try:
                        ckpt = torch.load(hybrid_ckpt, map_location="cpu")
                        if "horizon" in ckpt and int(ckpt.get("horizon", -1)) != int(H):
                            raise ValueError("checkpoint params mismatch")
                        if "wavelet" in ckpt and ckpt.get("wavelet") != wavelet:
                            raise ValueError("checkpoint params mismatch")
                        if "level" in ckpt and int(ckpt.get("level", -1)) != int(cat_level):
                            raise ValueError("checkpoint params mismatch")
                        if "boundary" in ckpt and str(ckpt.get("boundary", "wrap")).lower() != str(boundary).lower():
                            raise ValueError("checkpoint params mismatch")
                        comps_meta = list(ckpt.get("components", []))
                        if any(("per_series_scaling" not in item) for item in comps_meta):
                            raise ValueError("checkpoint too old (missing per_series_scaling)")
                        if any(not bool(item.get("per_series_scaling", False)) for item in comps_meta):
                            raise ValueError("checkpoint too old (global scaling)")
                        comps: List[HybridComponent] = []
                        for item in ckpt.get("components", []):
                            state_dict = item.get("state_dict")
                            lookback = item.get("lookback")
                            mu = float(item.get("mu", 0.0))
                            sd = float(item.get("sd", 1.0))
                            model = None
                            if state_dict is not None and lookback is not None:
                                cfg_global = TrainConfig(
                                    lookback=int(lookback),
                                    horizon=H,
                                    epochs=0,
                                    batch_size=128,
                                    lr=1e-3,
                                    weight_decay=1e-4,
                                    clip=1.0,
                                )
                                model = _base_factory(model_name, params=params)(cfg_global)
                                model.load_state_dict(state_dict)
                                model.to(cfg_global.device)
                                model.eval()
                            comps.append(
                                HybridComponent(
                                    model=model,
                                    mu=mu,
                                    sd=sd,
                                    lookback=lookback,
                                    per_series_scaling=bool(item.get("per_series_scaling", True)),
                                )
                            )
                        if comps:
                            global_hybrid_components[model_name] = comps
                            continue
                    except Exception as exc:
                        print(f"[m4:{cat}] failed to load hybrid components for {model_name}: {exc}")

                hybrid_cfg = TrainConfig(
                    lookback=max(16, min(256, max(32, 2 * H, 3 * per))),
                    horizon=H,
                    epochs=max(int(epochs), 2),
                    batch_size=128,
                    lr=3e-4,
                    weight_decay=1e-4,
                    clip=1.0,
                )
                comps = build_global_hybrid_components(
                    selected_list,
                    hybrid_cfg,
                    base_model_fn=_base_factory(model_name, params=params),
                    wavelet=wavelet,
                    level=cat_level,
                    boundary=boundary,
                )
                global_hybrid_components[model_name] = comps
                try:
                    payload = {
                        "category": cat,
                        "model_name": model_name,
                        "horizon": H,
                        "wavelet": wavelet,
                        "level": cat_level,
                        "boundary": boundary,
                        "components": [],
                    }
                    for comp in comps:
                        state = comp.model.state_dict() if comp.model is not None else None
                        payload["components"].append(
                            {
                                "state_dict": state,
                                "mu": comp.mu,
                                "sd": comp.sd,
                                "lookback": comp.lookback,
                                "per_series_scaling": bool(getattr(comp, "per_series_scaling", False)),
                            }
                        )
                    torch.save(payload, hybrid_ckpt)
                except Exception as exc:
                    print(f"[m4:{cat}] failed to save hybrid components for {model_name}: {exc}")

        for sid, y_tr, y_te in _progress(selected_list, desc=f"{cat} series", leave=False):
            L = best_L(y_tr, H, per)
            comps_override: list[np.ndarray] | None = None
            comps_full: list[np.ndarray] | None = None
            if use_full_modwt_components:
                y_full = np.concatenate([np.asarray(y_tr, float), np.asarray(y_te, float)], axis=0)
                A_full, D_full = modwt_decompose_with_boundary(
                    y_full, wavelet=wavelet, level=cat_level, boundary=boundary, check=True
                )
                comps_full = [A_full] + D_full if len(D_full) else [A_full]
                comps_override = [np.asarray(c[: len(y_tr)], float) for c in comps_full]
            total_len = int(len(y_tr) + H)
            cfg = TrainConfig(
                lookback=L,
                horizon=H,
                epochs=epochs,
                batch_size=64,
                lr=3e-4,
                weight_decay=2e-4,
                clip=0.5,
            )
            naive = seasonal_naive(y_tr, H, per)
            forecasts: Dict[str, np.ndarray] = {"Naive": naive.copy()}
            component_forecasts: Dict[str, Dict[str, np.ndarray]] = {}
            simulations: Dict[str, np.ndarray] = {}
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
                            level=cat_level,
                            boundary=boundary,
                            pretrained_components=None if use_full_modwt_components else global_hybrid_components.get(model_name),
                            seasonal_period=per_eff,
                        ).fit(y_tr, components_override=comps_override)
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                        component_forecasts[label] = model.forecast_components(y_tr, components_override=comps_override)
                        if simulate_full_series and comps_override is not None and comps_full is not None:
                            fitted_components = []
                            for comp_obj, comp_full_arr in zip(model.components, comps_full):
                                comp_tr_arr = np.asarray(comp_full_arr[: len(y_tr)], float)
                                fitted_components.append(
                                    _fitted_one_step_series(
                                        comp_obj,
                                        comp_tr_arr,
                                        device=cfg.device,
                                        total_len=total_len,
                                        mode=simulation_mode,
                                    )
                                )
                            y_sim = np.sum(np.stack(fitted_components, 0), axis=0)
                            # If we simulated beyond train+H (shouldn't), trim.
                            simulations[label] = np.asarray(y_sim[:total_len], float)
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
                            level=cat_level,
                            seasonal_period=per_eff,
                        ).fit(y_tr, components_override=comps_override)
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                        component_forecasts[label] = model.forecast_components(y_tr, components_override=comps_override)
                        if simulate_full_series and comps_full is not None and model.aj_component is not None:
                            A_tr_arr = np.asarray(comps_full[0][: len(y_tr)], float)
                            Aj_sim = _fitted_one_step_series(
                                model.aj_component,
                                A_tr_arr,
                                device=cfg.device,
                                total_len=total_len,
                                mode=simulation_mode,
                            )
                            details_sum = np.zeros(total_len, dtype=float)
                            # Use true details on train, predicted details on test (H).
                            comp_map = component_forecasts.get(label, {})
                            for j, dj_full in enumerate(comps_full[1:], start=1):
                                dj_tr = np.asarray(dj_full[: len(y_tr)], float)
                                dj_pred = np.asarray(comp_map.get(f"D_{j}", np.zeros(H)), float).ravel()
                                if dj_pred.size != H:
                                    dj_pred = np.pad(dj_pred, (0, max(0, H - dj_pred.size)), mode="edge")[:H]
                                dj_series = np.concatenate([dj_tr, dj_pred], axis=0)
                                if dj_series.size < total_len:
                                    dj_series = np.pad(dj_series, (0, total_len - dj_series.size), mode="edge")
                                details_sum += dj_series[:total_len]
                            simulations[label] = (Aj_sim + details_sum)[:total_len]
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
                            level=cat_level,
                            seasonal_period=per_eff,
                        ).fit(y_tr, components_override=comps_override)
                        forecasts[label] = model.forecast(y_tr, components_override=comps_override)
                        component_forecasts[label] = model.forecast_components(y_tr, components_override=comps_override)
                        if simulate_full_series and comps_full is not None and model.aj_component is not None:
                            A_tr_arr = np.asarray(comps_full[0][: len(y_tr)], float)
                            Aj_sim = _fitted_one_step_series(
                                model.aj_component,
                                A_tr_arr,
                                device=cfg.device,
                                total_len=total_len,
                                mode=simulation_mode,
                            )
                            details_sum = np.zeros(total_len, dtype=float)
                            comp_map = component_forecasts.get(label, {})
                            for j, dj_full in enumerate(comps_full[1:], start=1):
                                dj_tr = np.asarray(dj_full[: len(y_tr)], float)
                                dj_pred = np.asarray(comp_map.get(f"D_{j}", np.zeros(H)), float).ravel()
                                if dj_pred.size != H:
                                    dj_pred = np.pad(dj_pred, (0, max(0, H - dj_pred.size)), mode="edge")[:H]
                                dj_series = np.concatenate([dj_tr, dj_pred], axis=0)
                                if dj_series.size < total_len:
                                    dj_series = np.pad(dj_series, (0, total_len - dj_series.size), mode="edge")
                                details_sum += dj_series[:total_len]
                            simulations[label] = (Aj_sim + details_sum)[:total_len]
                    else:
                        raise ValueError(f"Unknown hybrid model '{model_name}'")
                except Exception as exc:
                    print(f"[m4:{cat}:{sid}] {label} failed: {exc}")

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
                            print(
                                f"[m4:{cat}:{sid}] Helformer HW baseline failed; "
                                "using raw series."
                            )
                    n_train = int(ratio_train.size)
                    min_windows = 8
                    max_lookback = max(1, n_train - min_windows)
                    effective_lookback = min(int(helformer_lookback), max_lookback)
                    if n_train <= effective_lookback:
                        effective_lookback = max(1, n_train - 1)
                    if effective_lookback != int(helformer_lookback):
                        print(
                            f"[m4:{cat}:{sid}] Helformer lookback adjusted "
                            f"{int(helformer_lookback)} -> {effective_lookback} "
                            f"(train_len={n_train})"
                        )
                    ratio_train = _ensure_min_length(ratio_train, effective_lookback + 1)
                    vmin, scale = _minmax_fit(ratio_train)
                    scaled_train = _minmax_transform(ratio_train, vmin, scale)
                    X, Y = _build_xy(scaled_train, effective_lookback)
                    if X.shape[0] == 0:
                        raise ValueError("not enough data for Helformer windows")
                    best_params = None
                    if helformer_optuna:
                        try:
                            best_params = _run_helformer_optuna(
                                X,
                                Y,
                                lookback=effective_lookback,
                                n_trials=int(helformer_optuna_trials),
                                timeout=helformer_optuna_timeout,
                                validation_split=float(helformer_optuna_validation_split),
                            )
                            print(
                                f"[m4:{cat}:{sid}] Helformer Optuna best params: "
                                f"{best_params}"
                            )
                        except Exception as exc:
                            print(f"[m4:{cat}:{sid}] Helformer Optuna failed: {exc}")
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
                        lookback=effective_lookback,
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
                    mode = str(helformer_forecast_mode or "rollout").lower()
                    pred_ratio_scaled_full = None
                    if mode in {"rollout_full", "full", "full_rollout"}:
                        pred_ratio_scaled_full = _predict_rollout_full(
                            model,
                            scaled_train,
                            effective_lookback,
                            total_len,
                        )
                        pred_ratio_scaled = pred_ratio_scaled_full[-H:]
                    elif mode in {"rollout", "iterative", "recursive"}:
                        scaled_history = scaled_train.tolist()
                        pred_ratio_scaled = _predict_rollout(
                            model,
                            scaled_history,
                            effective_lookback,
                            H,
                        )
                    elif mode in {"test_windows", "teacher", "direct"}:
                        ratio_test = _ensure_min_length(ratio_test, 1)
                        test_values = np.concatenate([ratio_train[-effective_lookback:], ratio_test])
                        scaled_test = _minmax_transform(test_values, vmin, scale)
                        x_test = _build_x(scaled_test, effective_lookback)
                        pred_ratio_scaled = _predict_test_windows(model, x_test)
                    else:
                        raise ValueError(f"Unknown Helformer forecast mode '{helformer_forecast_mode}'")
                    pred_ratio = _minmax_inverse(pred_ratio_scaled, vmin, scale)
                    if base_test is not None:
                        pred = pred_ratio * base_test
                    else:
                        pred = pred_ratio
                    forecasts["Helformer"] = pred[:H]
                    if simulate_full_series:
                        if pred_ratio_scaled_full is None:
                            pred_ratio_scaled_full = _predict_rollout_full(
                                model,
                                scaled_train,
                                effective_lookback,
                                total_len,
                            )
                        pred_ratio_full = _minmax_inverse(pred_ratio_scaled_full, vmin, scale)
                        if base_test is not None and base_train is not None:
                            base_full = np.concatenate([base_train, base_test])
                            base_full = np.resize(base_full, total_len)
                            pred_full = pred_ratio_full[:total_len] * base_full
                        else:
                            pred_full = pred_ratio_full[:total_len]
                        simulations["Helformer"] = np.asarray(pred_full, float)
                except Exception as exc:
                    print(f"[m4:{cat}:{sid}] Helformer failed: {exc}")
                finally:
                    try:
                        tf.keras.backend.clear_session()
                    except Exception:
                        pass

            # Classical baselines
            try:
                forecasts["ARIMA"] = arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[m4:{cat}:{sid}] ARIMA failed: {exc}")
            try:
                forecasts["ARIMA_auto"] = auto_arima_forecast(y_tr, H)
            except Exception as exc:
                print(f"[m4:{cat}:{sid}] ARIMA_auto failed: {exc}")
            try:
                forecasts["ETS"] = ets_forecast(y_tr, H, seasonal_periods=per)
            except Exception as exc:
                print(f"[m4:{cat}:{sid}] ETS failed: {exc}")
            try:
                freq = freq_map.get(cat, "D")
                forecasts["Prophet"] = prophet_forecast(y_tr, H, freq=freq)
            except Exception as exc:
                print(f"[m4:{cat}:{sid}] Prophet failed: {exc}")
            for model_name in base_models:
                label = label_map[model_name]
                forecasts.setdefault(label, naive.copy())

            rec = {"category": cat, "series_id": sid}
            for name, pred in forecasts.items():
                key = name.replace(" ", "_")
                rec[f"{key}_sMAPE"] = smape(y_te, pred)
                rec[f"{key}_MAPE"] = mape(y_te, pred)
                rec[f"{key}_RMSE"] = rmse(y_te, pred)
                rec[f"{key}_MSE"] = mse(y_te, pred)
            rows.append(rec)

            title = f"{cat.upper()} {sid} (H={H}, L={L})"
            if visualize:
                series_key = f"{cat}_{sid}"
                viz_dir = out_dir / "viz"
                if use_helformer and not base_models:
                    hw = None
                    try:
                        from ..models.helformer import helformer_hw_components

                        hw = helformer_hw_components(
                            y_tr,
                            y_te,
                            seasonal_period=per,
                            hw_trend=helformer_hw_trend,
                            hw_seasonal=helformer_hw_seasonal,
                        )
                    except Exception as exc:
                        print(f"[m4:{cat}:{sid}] HW decomposition failed: {exc}")
                        hw = None
                    save_series_viz_bundle_helformer_hw(
                        out_dir=viz_dir,
                        series_key=series_key,
                        title_prefix=title,
                        y_tr=np.asarray(y_tr, float),
                        y_te=np.asarray(y_te, float),
                        forecasts=forecasts,
                        base_train=None if hw is None else hw.base_train,
                        base_test=None if hw is None else hw.base_test,
                        season=None if hw is None else hw.season,
                        seasonal_mode="mul" if hw is None else hw.seasonal_mode,
                    )
                elif not base_models:
                    save_series_viz_bundle_basic(
                        out_dir=viz_dir,
                        series_key=series_key,
                        title_prefix=title,
                        y_tr=np.asarray(y_tr, float),
                        y_te=np.asarray(y_te, float),
                        forecasts=forecasts,
                    )
                else:
                    save_series_viz_bundle(
                        out_dir=viz_dir,
                        series_key=series_key,
                        title_prefix=title,
                        y_tr=y_tr,
                        y_te=y_te,
                        forecasts=forecasts,
                        wavelet=wavelet,
                        level=cat_level,
                        boundary=boundary,
                        component_forecasts=component_forecasts if component_forecasts else None,
                    )
                if simulate_full_series and simulations:
                    if simulation_train_only_plot:
                        save_simulation_train_plot(
                            y_tr=np.asarray(y_tr, float),
                            simulations=simulations,
                            title=f"{title} simulation",
                            save_path=(out_dir / "viz" / "09_simulation_train" / f"{series_key}.png"),
                        )
                    else:
                        save_simulation_full_plot(
                            y_tr=np.asarray(y_tr, float),
                            y_te=np.asarray(y_te, float),
                            simulations=simulations,
                            title=f"{title} simulation (full series)",
                            save_path=(out_dir / "viz" / "09_simulation_full" / f"{series_key}.png"),
                        )
            else:
                save_png = out_dir / f"{cat}_{sid}.png"
                if plot_test_only:
                    save_png = out_dir / f"{cat}_{sid}_test_only.png"
                    plot_forecast_test_only(title, y_te, forecasts, save_path=save_png)
                else:
                    plot_forecast(title, y_tr, y_te, forecasts, save_path=save_png)
                if component_forecasts:
                    save_component_forecast_plot(
                        y_tr=y_tr,
                        y_te=y_te,
                        component_forecasts=component_forecasts,
                        wavelet=wavelet,
                        level=cat_level,
                        boundary=boundary,
                        title=f"{title} component forecasts",
                        save_path=out_dir / f"{cat}_{sid}_components.png",
                    )
                    save_component_forecast_test_only_plot(
                        y_tr=y_tr,
                        y_te=y_te,
                        component_forecasts=component_forecasts,
                        wavelet=wavelet,
                        level=cat_level,
                        boundary=boundary,
                        title=f"{title} component forecasts (test only)",
                        save_path=out_dir / f"{cat}_{sid}_components_test_only.png",
                    )
                if simulate_full_series and simulations:
                    if simulation_train_only_plot:
                        save_simulation_train_plot(
                            y_tr=np.asarray(y_tr, float),
                            simulations=simulations,
                            title=f"{title} simulation (train only)",
                            save_path=out_dir / f"{cat}_{sid}_simulation_train.png",
                        )
                    else:
                        save_simulation_full_plot(
                            y_tr=np.asarray(y_tr, float),
                            y_te=np.asarray(y_te, float),
                            simulations=simulations,
                            title=f"{title} simulation (full series)",
                            save_path=out_dir / f"{cat}_{sid}_simulation_full.png",
                        )

    df = pd.DataFrame(rows)
    metrics_csv = out_dir / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    print(f"[saved] metrics: {metrics_csv}")
    if df.empty:
        print("No results generated — check CSV/logs.")
        return df

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
    return df


__all__ = ["evaluate_m4_hybrids"]
