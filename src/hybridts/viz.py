"""Visualization helpers for per-series analysis plots."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


def _clean_series(y: Iterable[float]) -> np.ndarray:
    data = np.asarray(list(y), dtype=float)
    if data.size == 0:
        return data
    mask = np.isfinite(data)
    if not np.all(mask):
        data = data[mask]
    return data.astype(float, copy=False)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_raw_series_plot(
    *,
    y_tr: np.ndarray,
    y_te: Optional[np.ndarray],
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_tr = _clean_series(y_tr)
    y_te = None if y_te is None else _clean_series(y_te)

    fig, ax = plt.subplots(figsize=(10.5, 3.8))
    xs_tr = np.arange(y_tr.size)
    ax.plot(xs_tr, y_tr, linewidth=1.9, label="train", color="C0")
    if y_te is not None and y_te.size > 0:
        xs_te = np.arange(y_tr.size, y_tr.size + y_te.size)
        ax.plot(xs_te, y_te, linewidth=2.0, label="test", color="C3")
        ax.axvline(x=y_tr.size - 0.5, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    fig.tight_layout()

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_helformer_hw_decomposition_pair_plot(
    *,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    base_train: np.ndarray,
    base_test: np.ndarray,
    title: str,
    save_path: Path,
) -> None:
    """Save a paired plot for the Holt-Winters ratio decomposition used by Helformer.

    Left: original series (train/test) with HW baseline (fitted/forecast).
    Right: multiplicative ratio series y/base (train/test).
    """
    import matplotlib.pyplot as plt

    y_tr_arr = np.asarray(y_tr, float).ravel()
    y_te_arr = np.asarray(y_te, float).ravel()
    base_train_arr = np.asarray(base_train, float).ravel()
    base_test_arr = np.asarray(base_test, float).ravel()

    n_tr = int(y_tr_arr.size)
    n_te = int(y_te_arr.size)
    if n_tr == 0 or n_te == 0:
        return

    if base_train_arr.size != n_tr:
        base_train_arr = np.resize(base_train_arr, n_tr)
    if base_test_arr.size != n_te:
        base_test_arr = np.resize(base_test_arr, n_te)

    denom_tr = np.where(np.abs(base_train_arr) < 1e-8, 1.0, base_train_arr)
    ratio_tr = y_tr_arr / denom_tr
    denom_te = np.where(np.abs(base_test_arr) < 1e-8, 1.0, base_test_arr)
    ratio_te = y_te_arr / denom_te

    xs_tr = np.arange(n_tr)
    xs_te = np.arange(n_tr, n_tr + n_te)

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.2))
    ax0, ax1 = axes

    # Series + baseline
    ax0.plot(xs_tr, y_tr_arr, linewidth=1.9, label="y (train)", color="C0")
    ax0.plot(xs_te, y_te_arr, linewidth=2.0, label="y (test)", color="C3")
    ax0.plot(xs_tr, base_train_arr, linewidth=1.7, label="HW baseline (fit)", color="C1")
    ax0.plot(xs_te, base_test_arr, linewidth=1.7, linestyle="--", label="HW baseline (fcst)", color="C1")
    ax0.axvline(x=n_tr - 0.5, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
    ax0.set_title("Series vs. Holt-Winters baseline", fontsize=10)
    ax0.set_xlabel("t")
    ax0.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax0.minorticks_on()
    ax0.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    ax0.legend(fontsize=8, frameon=False, ncol=2)

    # Ratio component
    ratio_tr = np.where(np.isfinite(ratio_tr), ratio_tr, np.nan)
    ratio_te = np.where(np.isfinite(ratio_te), ratio_te, np.nan)
    ax1.plot(xs_tr, ratio_tr, linewidth=1.7, label="y/base (train)", color="C0")
    ax1.plot(xs_te, ratio_te, linewidth=1.8, label="y/base (test)", color="C3")
    ax1.axhline(1.0, color="0.35", linewidth=1.0, alpha=0.65)
    ax1.axvline(x=n_tr - 0.5, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
    ax1.set_title("Multiplicative ratio component", fontsize=10)
    ax1.set_xlabel("t")
    ax1.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax1.minorticks_on()
    ax1.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    ax1.legend(fontsize=8, frameon=False, ncol=1)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_helformer_hw_components_plot(
    *,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    base_train: np.ndarray,
    base_test: np.ndarray,
    season: np.ndarray | None,
    title: str,
    save_path: Path,
    seasonal_mode: str = "mul",
) -> None:
    """Save Holt-Winters decomposition components (trend-like, seasonal, residual).

    - Trend-like component: baseline without seasonality (base/season for multiplicative, base-season for additive)
    - Seasonal component: seasonal factor/value from statsmodels (train only)
    - Residual component: y/base (multiplicative) or y-base (additive)
    """
    import matplotlib.pyplot as plt

    y_tr_arr = np.asarray(y_tr, float).ravel()
    y_te_arr = np.asarray(y_te, float).ravel()
    base_train_arr = np.asarray(base_train, float).ravel()
    base_test_arr = np.asarray(base_test, float).ravel()
    season_arr = None if season is None else np.asarray(season, float).ravel()

    n_tr = int(y_tr_arr.size)
    n_te = int(y_te_arr.size)
    if n_tr == 0 or n_te == 0:
        return

    if base_train_arr.size != n_tr:
        base_train_arr = np.resize(base_train_arr, n_tr)
    if base_test_arr.size != n_te:
        base_test_arr = np.resize(base_test_arr, n_te)
    if season_arr is not None and season_arr.size != n_tr:
        season_arr = np.resize(season_arr, n_tr)

    denom_tr = np.where(np.abs(base_train_arr) < 1e-8, 1.0, base_train_arr)
    denom_te = np.where(np.abs(base_test_arr) < 1e-8, 1.0, base_test_arr)

    seasonal_mode = (seasonal_mode or "none").lower()
    is_mul = seasonal_mode.startswith("mul")
    is_add = seasonal_mode.startswith("add")

    # Helformer uses a ratio residual regardless of additive/multiplicative HW.
    residual_tr = y_tr_arr / denom_tr
    residual_te = y_te_arr / denom_te
    residual_tr = np.where(np.isfinite(residual_tr), residual_tr, np.nan)
    residual_te = np.where(np.isfinite(residual_te), residual_te, np.nan)

    trend_like = None
    if season_arr is not None and np.any(np.isfinite(season_arr)):
        if is_mul:
            season_safe = np.where(np.abs(season_arr) < 1e-8, 1.0, season_arr)
            trend_like = base_train_arr / season_safe
        elif is_add:
            trend_like = base_train_arr - season_arr
        else:
            trend_like = base_train_arr
        trend_like = np.where(np.isfinite(trend_like), trend_like, np.nan)
    else:
        trend_like = base_train_arr

    xs_tr = np.arange(n_tr)
    xs_te = np.arange(n_tr, n_tr + n_te)

    fig, axes = plt.subplots(4, 1, figsize=(12.8, 9.6), sharex=False)

    # 1) Series + baseline
    ax = axes[0]
    ax.plot(xs_tr, y_tr_arr, linewidth=1.9, label="y (train)", color="C0")
    ax.plot(xs_te, y_te_arr, linewidth=2.0, label="y (test)", color="C3")
    ax.plot(xs_tr, base_train_arr, linewidth=1.7, label="HW baseline (fit)", color="C1")
    ax.plot(xs_te, base_test_arr, linewidth=1.7, linestyle="--", label="HW baseline (fcst)", color="C1")
    ax.axvline(x=n_tr - 0.5, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Series vs. Holt-Winters baseline", fontsize=10)
    ax.set_xlabel("t")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    # 2) Trend-like component (train only)
    ax = axes[1]
    if trend_like is not None:
        ax.plot(xs_tr, trend_like, linewidth=1.8, color="C0")
    ax.set_title("Trend-like component (baseline without seasonality)", fontsize=10)
    ax.set_xlabel("t (train)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    # 3) Seasonal component (train only)
    ax = axes[2]
    if season_arr is not None:
        ax.plot(xs_tr, season_arr, linewidth=1.6, color="C2")
        ax.axhline(1.0 if is_mul else 0.0, color="0.35", linewidth=1.0, alpha=0.65)
    else:
        ax.text(
            0.5,
            0.5,
            "No seasonal component",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=9,
            color="0.35",
        )
    ax.set_title("Seasonal component (state)", fontsize=10)
    ax.set_xlabel("t (train)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    # 4) Residual component (train/test)
    ax = axes[3]
    ax.plot(xs_tr, residual_tr, linewidth=1.6, label="residual (train)", color="C0")
    ax.plot(xs_te, residual_te, linewidth=1.7, label="residual (test)", color="C3")
    ax.axhline(1.0, color="0.35", linewidth=1.0, alpha=0.65)
    ax.axvline(x=n_tr - 0.5, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title("Residual component", fontsize=10)
    ax.set_xlabel("t")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_acf_pacf_plot(
    *,
    y: np.ndarray,
    title: str,
    save_path: Path,
    nlags: int = 48,
) -> None:
    import matplotlib.pyplot as plt

    y = _clean_series(y)
    if y.size < 8:
        return

    try:
        from statsmodels.tsa.stattools import acf as sm_acf
        from statsmodels.tsa.stattools import pacf as sm_pacf
    except Exception:
        return

    # statsmodels PACF requires nlags < 0.5 * nobs (strict inequality).
    max_lags = int(min(max(1, nlags), max(1, (y.size - 1) // 2)))
    acf_vals = sm_acf(y, nlags=max_lags, fft=False)
    pacf_vals = sm_pacf(y, nlags=max_lags, method="ywm")
    lags = np.arange(1, max_lags + 1)
    crit = 1.96 / np.sqrt(float(y.size))

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 5.8), sharex=True)
    for ax, vals, name in (
        (axes[0], acf_vals[1:], "ACF"),
        (axes[1], pacf_vals[1:], "PACF"),
    ):
        ax.axhline(0.0, color="0.35", linewidth=1.0)
        ax.vlines(lags, 0.0, vals, color="C0", linewidth=1.6)
        ax.plot(lags, vals, "o", color="C0", markersize=3.5)
        ax.axhline(crit, color="C3", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.axhline(-crit, color="C3", linestyle="--", linewidth=1.0, alpha=0.8)
        ax.set_ylabel(name)
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    axes[1].set_xlabel("lag")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_modwt_decomposition_plots(
    *,
    y: np.ndarray,
    wavelet: str,
    level: int,
    boundary: str = "wrap",
    title_prefix: str,
    out_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    from .hybrids.modwt_hybrid import modwt_decompose_with_boundary

    y = _clean_series(y)
    if y.size < 2:
        return

    out_dir = _ensure_dir(Path(out_dir))
    A, D = modwt_decompose_with_boundary(
        y, wavelet=wavelet, level=level, boundary=boundary, check=True
    )
    components = [("A_J", np.asarray(A, float))] + [
        (f"D_{j+1}", np.asarray(comp, float)) for j, comp in enumerate(D)
    ]

    # One combined figure for quick inspection.
    fig, axes = plt.subplots(len(components), 1, figsize=(10.5, 2.2 * len(components)), sharex=True)
    if len(components) == 1:
        axes = [axes]
    for ax, (name, comp) in zip(axes, components):
        ax.plot(np.arange(comp.size), comp, linewidth=1.4)
        ax.set_ylabel(name)
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
    axes[-1].set_xlabel("t")
    fig.suptitle(f"{title_prefix} MODWT ({wavelet}, level={level}, boundary={boundary})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "components.png", bbox_inches="tight", dpi=160)
    plt.close(fig)

    # Per-component figures.
    for name, comp in components:
        fig, ax = plt.subplots(figsize=(10.5, 3.2))
        ax.plot(np.arange(comp.size), comp, linewidth=1.6)
        ax.set_title(f"{title_prefix} {name}", fontsize=10)
        ax.set_xlabel("t")
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)
        fig.tight_layout()
        fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=160)
        plt.close(fig)


def save_forecast_zoom_plot(
    *,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    title: str,
    save_path: Path,
    tail: int = 120,
) -> None:
    import matplotlib.pyplot as plt

    y_tr = _clean_series(y_tr)
    y_te = _clean_series(y_te)
    H = int(y_te.size)
    if H <= 0 or y_tr.size == 0:
        return

    tail = int(max(H + 2, min(tail, y_tr.size)))
    start = y_tr.size - tail
    y_tr_tail = y_tr[start:]

    xs_tr = np.arange(start, start + y_tr_tail.size)
    xs_te = np.arange(y_tr.size - 1, y_tr.size + H)
    last_tr = np.array([y_tr[-1]])

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(xs_tr, y_tr_tail, label="train (tail)", linewidth=1.8, color="C0")
    ax.plot(xs_te, np.hstack([last_tr, y_te]), label="test", linewidth=2.0, color="C3")
    for k, v in forecasts.items():
        v = np.asarray(v, float).ravel()
        if v.size != H:
            continue
        ax.plot(xs_te, np.hstack([last_tr, v]), label=k, linewidth=1.7, alpha=0.95, linestyle="--")

    ax.axvline(x=y_tr.size - 1, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    all_series = [y_tr_tail, y_te] + [np.asarray(v, float) for v in forecasts.values()]
    ymin = float(np.nanmin([np.nanmin(s) for s in all_series if s.size > 0]))
    ymax = float(np.nanmax([np.nanmax(s) for s in all_series if s.size > 0]))
    pad = 0.06 * (ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_forecast_test_only_plot(
    *,
    y_te: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_te = _clean_series(y_te)
    H = int(y_te.size)
    if H <= 0:
        return

    xs = np.arange(H)
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(xs, y_te, label="test", linewidth=2.0, color="C3")
    for k, v in forecasts.items():
        v = np.asarray(v, float).ravel()
        if v.size != H:
            continue
        ax.plot(xs, v, label=k, linewidth=1.7, alpha=0.95, linestyle="--")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t (test)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    all_series = [y_te] + [np.asarray(v, float) for v in forecasts.values()]
    ymin = float(np.nanmin([np.nanmin(s) for s in all_series if s.size > 0]))
    ymax = float(np.nanmax([np.nanmax(s) for s in all_series if s.size > 0]))
    pad = 0.06 * (ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_simulation_full_plot(
    *,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    simulations: Dict[str, np.ndarray],
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_tr = _clean_series(y_tr)
    y_te = _clean_series(y_te)
    H = int(y_te.size)
    if y_tr.size == 0 or H <= 0:
        return

    y_full = np.concatenate([y_tr, y_te]).astype(float, copy=False)
    xs = np.arange(y_full.size)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(np.arange(y_tr.size), y_tr, label="train", linewidth=1.8, color="C0")
    ax.plot(np.arange(y_tr.size, y_tr.size + H), y_te, label="test", linewidth=2.0, color="C3")

    for label, sim in simulations.items():
        sim = np.asarray(sim, float).ravel()
        if sim.size != y_full.size:
            continue
        ax.plot(xs, sim, label=label, linewidth=1.6, alpha=0.95, linestyle="--")

    ax.axvline(x=y_tr.size - 0.5, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    all_series = [y_full] + [np.asarray(v, float) for v in simulations.values()]
    ymin = float(np.nanmin([np.nanmin(s) for s in all_series if s.size > 0]))
    ymax = float(np.nanmax([np.nanmax(s) for s in all_series if s.size > 0]))
    pad = 0.06 * (ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_simulation_train_plot(
    *,
    y_tr: np.ndarray,
    simulations: Dict[str, np.ndarray],
    title: str,
    save_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    y_tr = _clean_series(y_tr)
    if y_tr.size == 0:
        return

    n = int(y_tr.size)
    xs = np.arange(n)

    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(xs, y_tr, label="train", linewidth=1.9, color="C0")
    for label, sim in simulations.items():
        sim = np.asarray(sim, float).ravel()
        if sim.size < n:
            continue
        ax.plot(xs, sim[:n], label=label, linewidth=1.6, alpha=0.95, linestyle="--")

    ax.set_title(title, fontsize=10)
    ax.set_xlabel("t (train)")
    ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
    ax.minorticks_on()
    ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    all_series = [y_tr] + [np.asarray(v, float)[:n] for v in simulations.values() if np.asarray(v).size >= n]
    ymin = float(np.nanmin([np.nanmin(s) for s in all_series if s.size > 0]))
    ymax = float(np.nanmax([np.nanmax(s) for s in all_series if s.size > 0]))
    pad = 0.06 * (ymax - ymin) if np.isfinite(ymax - ymin) and (ymax - ymin) > 0 else 1.0
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()
    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_series_viz_bundle(
    *,
    out_dir: Path,
    series_key: str,
    title_prefix: str,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    wavelet: str,
    level: int,
    boundary: str = "wrap",
    acf_lags: int = 48,
    zoom_tail: int = 120,
    component_forecasts: Optional[Dict[str, Dict[str, np.ndarray]]] = None,
) -> None:
    """Save per-series plots into named subfolders under out_dir.

    The directory structure is:
    - 01_raw_series/<series_key>.png
    - 02_acf_pacf/<series_key>.png
    - 03_modwt/<series_key>/* (components)
    - 04_forecast_zoom/<series_key>.png
    - 05_forecast_full/<series_key>.png
    - 06_component_forecasts/<series_key>.png (optional)
    - 07_forecast_test_only/<series_key>.png
    - 08_component_forecasts_test_only/<series_key>.png (optional)
    """
    out_dir = Path(out_dir)
    save_raw_series_plot(
        y_tr=y_tr,
        y_te=y_te,
        title=f"{title_prefix} raw series",
        save_path=out_dir / "01_raw_series" / f"{series_key}.png",
    )
    save_acf_pacf_plot(
        y=y_tr,
        title=f"{title_prefix} ACF/PACF",
        save_path=out_dir / "02_acf_pacf" / f"{series_key}.png",
        nlags=acf_lags,
    )
    save_modwt_decomposition_plots(
        y=y_tr,
        wavelet=wavelet,
        level=level,
        boundary=boundary,
        title_prefix=title_prefix,
        out_dir=out_dir / "03_modwt" / series_key,
    )
    save_forecast_zoom_plot(
        y_tr=y_tr,
        y_te=y_te,
        forecasts=forecasts,
        title=f"{title_prefix} forecast (zoom)",
        save_path=out_dir / "04_forecast_zoom" / f"{series_key}.png",
        tail=zoom_tail,
    )

    from .data import plot_forecast

    plot_forecast(
        f"{title_prefix} forecast",
        np.asarray(y_tr, float),
        np.asarray(y_te, float),
        forecasts,
        save_path=out_dir / "05_forecast_full" / f"{series_key}.png",
    )

    save_forecast_test_only_plot(
        y_te=np.asarray(y_te, float),
        forecasts=forecasts,
        title=f"{title_prefix} forecast (test only)",
        save_path=out_dir / "07_forecast_test_only" / f"{series_key}.png",
    )

    if component_forecasts:
        save_component_forecast_plot(
            y_tr=np.asarray(y_tr, float),
            y_te=np.asarray(y_te, float),
            component_forecasts=component_forecasts,
            wavelet=wavelet,
            level=level,
            boundary=boundary,
            title=f"{title_prefix} component forecasts",
            save_path=out_dir / "06_component_forecasts" / f"{series_key}.png",
        )
        save_component_forecast_test_only_plot(
            y_tr=np.asarray(y_tr, float),
            y_te=np.asarray(y_te, float),
            component_forecasts=component_forecasts,
            wavelet=wavelet,
            level=level,
            boundary=boundary,
            title=f"{title_prefix} component forecasts (test only)",
            save_path=out_dir / "08_component_forecasts_test_only" / f"{series_key}.png",
        )


def save_series_viz_bundle_basic(
    *,
    out_dir: Path,
    series_key: str,
    title_prefix: str,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    acf_lags: int = 48,
    zoom_tail: int = 120,
) -> None:
    """Save per-series plots without MODWT-related visuals."""
    out_dir = Path(out_dir)
    save_raw_series_plot(
        y_tr=y_tr,
        y_te=y_te,
        title=f"{title_prefix} raw series",
        save_path=out_dir / "01_raw_series" / f"{series_key}.png",
    )
    save_acf_pacf_plot(
        y=y_tr,
        title=f"{title_prefix} ACF/PACF",
        save_path=out_dir / "02_acf_pacf" / f"{series_key}.png",
        nlags=acf_lags,
    )
    save_forecast_zoom_plot(
        y_tr=y_tr,
        y_te=y_te,
        forecasts=forecasts,
        title=f"{title_prefix} forecast (zoom)",
        save_path=out_dir / "03_forecast_zoom" / f"{series_key}.png",
        tail=zoom_tail,
    )

    from .data import plot_forecast

    plot_forecast(
        f"{title_prefix} forecast",
        np.asarray(y_tr, float),
        np.asarray(y_te, float),
        forecasts,
        save_path=out_dir / "04_forecast_full" / f"{series_key}.png",
    )
    save_forecast_test_only_plot(
        y_te=np.asarray(y_te, float),
        forecasts=forecasts,
        title=f"{title_prefix} forecast (test only)",
        save_path=out_dir / "05_forecast_test_only" / f"{series_key}.png",
    )


def save_series_viz_bundle_helformer_hw(
    *,
    out_dir: Path,
    series_key: str,
    title_prefix: str,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    forecasts: Dict[str, np.ndarray],
    base_train: np.ndarray | None,
    base_test: np.ndarray | None,
    season: np.ndarray | None,
    seasonal_mode: str = "mul",
    acf_lags: int = 48,
    zoom_tail: int = 120,
) -> None:
    """Save per-series plots with Holt-Winters (HW) decomposition (no MODWT visuals).

    The directory structure is:
    - 01_raw_series/<series_key>.png
    - 02_acf_pacf/<series_key>.png
    - 03_hw_pair/<series_key>.png (optional)
    - 04_hw_components/<series_key>.png (optional)
    - 05_forecast_zoom/<series_key>.png
    - 06_forecast_full/<series_key>.png
    - 07_forecast_test_only/<series_key>.png
    """
    out_dir = Path(out_dir)
    save_raw_series_plot(
        y_tr=y_tr,
        y_te=y_te,
        title=f"{title_prefix} raw series",
        save_path=out_dir / "01_raw_series" / f"{series_key}.png",
    )
    save_acf_pacf_plot(
        y=y_tr,
        title=f"{title_prefix} ACF/PACF",
        save_path=out_dir / "02_acf_pacf" / f"{series_key}.png",
        nlags=acf_lags,
    )
    if base_train is not None and base_test is not None:
        save_helformer_hw_decomposition_pair_plot(
            y_tr=np.asarray(y_tr, float),
            y_te=np.asarray(y_te, float),
            base_train=np.asarray(base_train, float),
            base_test=np.asarray(base_test, float),
            title=f"{title_prefix} HW ratio decomposition",
            save_path=out_dir / "03_hw_pair" / f"{series_key}.png",
        )
        save_helformer_hw_components_plot(
            y_tr=np.asarray(y_tr, float),
            y_te=np.asarray(y_te, float),
            base_train=np.asarray(base_train, float),
            base_test=np.asarray(base_test, float),
            season=None if season is None else np.asarray(season, float),
            seasonal_mode=str(seasonal_mode or "mul"),
            title=f"{title_prefix} HW decomposition components",
            save_path=out_dir / "04_hw_components" / f"{series_key}.png",
        )

    save_forecast_zoom_plot(
        y_tr=y_tr,
        y_te=y_te,
        forecasts=forecasts,
        title=f"{title_prefix} forecast (zoom)",
        save_path=out_dir / "05_forecast_zoom" / f"{series_key}.png",
        tail=zoom_tail,
    )

    from .data import plot_forecast

    plot_forecast(
        f"{title_prefix} forecast",
        np.asarray(y_tr, float),
        np.asarray(y_te, float),
        forecasts,
        save_path=out_dir / "06_forecast_full" / f"{series_key}.png",
    )
    save_forecast_test_only_plot(
        y_te=np.asarray(y_te, float),
        forecasts=forecasts,
        title=f"{title_prefix} forecast (test only)",
        save_path=out_dir / "07_forecast_test_only" / f"{series_key}.png",
    )


def save_component_forecast_plot(
    *,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    component_forecasts: Dict[str, Dict[str, np.ndarray]],
    wavelet: str,
    level: int,
    boundary: str = "wrap",
    title: str,
    save_path: Path,
) -> None:
    """Plot Aj/Dj component forecasts for hybrid models.

    component_forecasts maps model label -> {component_name: forecast (H,)}.
    Supported component names: "A_J", "D_1", ..., "D_J".
    """
    import matplotlib.pyplot as plt

    from .hybrids.modwt_hybrid import modwt_decompose_with_boundary

    y_tr = _clean_series(y_tr)
    y_te = _clean_series(y_te)
    H = int(y_te.size)
    if y_tr.size < 2 or H <= 0:
        return

    y_full = np.concatenate([y_tr, y_te]).astype(float, copy=False)
    A_full, D_full = modwt_decompose_with_boundary(
        y_full, wavelet=wavelet, level=level, boundary=boundary, check=True
    )
    comps = [("A_J", np.asarray(A_full, float))] + [
        (f"D_{j+1}", np.asarray(comp, float)) for j, comp in enumerate(D_full)
    ]

    n = len(comps)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    for ax, (cname, full) in zip(axes, comps):
        tr = full[: y_tr.size]
        te = full[y_tr.size :]
        xs_tr = np.arange(tr.size)
        xs_full = np.arange(y_tr.size + H)
        xs_te = np.arange(y_tr.size, y_tr.size + H)

        ax.plot(xs_tr, tr, linewidth=1.6, color="C0", label="train" if cname == "A_J" else None)
        ax.plot(xs_te, te, linewidth=1.8, color="C3", label="test" if cname == "A_J" else None)
        for model_label, comp_map in component_forecasts.items():
            pred = comp_map.get(cname)
            if pred is None:
                continue
            pred = np.asarray(pred, float).ravel()
            if pred.size != H:
                continue
            full_pred = np.concatenate([tr, pred]).astype(float, copy=False)
            ax.plot(
                xs_full,
                full_pred,
                linewidth=1.4,
                alpha=0.9,
                linestyle="--",
                label=model_label if cname == "A_J" else None,
            )

        ax.set_ylabel(cname)
        ax.axvline(x=y_tr.size - 0.5, color="0.35", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    axes[-1].set_xlabel("t")
    fig.suptitle(title, fontsize=10)
    # Put legend on the first subplot to avoid repetition.
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


def save_component_forecast_test_only_plot(
    *,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    component_forecasts: Dict[str, Dict[str, np.ndarray]],
    wavelet: str,
    level: int,
    boundary: str = "wrap",
    title: str,
    save_path: Path,
) -> None:
    """Plot Aj/Dj forecasts restricted to the test horizon (no train segment)."""
    import matplotlib.pyplot as plt

    from .hybrids.modwt_hybrid import modwt_decompose_with_boundary

    y_tr = _clean_series(y_tr)
    y_te = _clean_series(y_te)
    H = int(y_te.size)
    if y_tr.size < 2 or H <= 0:
        return

    y_full = np.concatenate([y_tr, y_te]).astype(float, copy=False)
    A_full, D_full = modwt_decompose_with_boundary(
        y_full, wavelet=wavelet, level=level, boundary=boundary, check=True
    )
    comps = [("A_J", np.asarray(A_full, float))] + [
        (f"D_{j+1}", np.asarray(comp, float)) for j, comp in enumerate(D_full)
    ]

    n = len(comps)
    fig, axes = plt.subplots(n, 1, figsize=(10.5, 2.2 * n), sharex=True)
    if n == 1:
        axes = [axes]

    xs = np.arange(H)
    for ax, (cname, full) in zip(axes, comps):
        te = full[y_tr.size :]
        ax.plot(xs, te, linewidth=1.8, color="C3", label="test" if cname == "A_J" else None)
        for model_label, comp_map in component_forecasts.items():
            pred = comp_map.get(cname)
            if pred is None:
                continue
            pred = np.asarray(pred, float).ravel()
            if pred.size != H:
                continue
            ax.plot(xs, pred, linewidth=1.4, alpha=0.9, linestyle="--", label=model_label if cname == "A_J" else None)

        ax.set_ylabel(cname)
        ax.grid(True, which="major", linestyle="-", linewidth=0.6, alpha=0.25)
        ax.minorticks_on()
        ax.grid(True, which="minor", linestyle=":", linewidth=0.5, alpha=0.18)

    axes[-1].set_xlabel("t (test)")
    fig.suptitle(title, fontsize=10)
    axes[0].legend(fontsize=8, ncol=2, frameon=False)
    fig.tight_layout()

    save_path = Path(save_path)
    _ensure_dir(save_path.parent)
    fig.savefig(save_path, bbox_inches="tight", dpi=160)
    plt.close(fig)


__all__ = [
    "save_series_viz_bundle",
    "save_series_viz_bundle_basic",
    "save_series_viz_bundle_helformer_hw",
    "save_helformer_hw_decomposition_pair_plot",
    "save_helformer_hw_components_plot",
    "save_component_forecast_plot",
    "save_component_forecast_test_only_plot",
    "save_forecast_test_only_plot",
    "save_simulation_full_plot",
    "save_simulation_train_plot",
]
