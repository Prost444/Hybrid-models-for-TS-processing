"""PyTorch Helformer — port of the TF/Keras Helformer architecture.

Architecture: N × (LayerNorm → MHA → Add → LayerNorm → Add) → LSTM → Dense(1)
Produces single-step predictions; use helformer_forecast_pt() for multi-step rollout.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except Exception:
    ExponentialSmoothing = None


class HelformerBlock(nn.Module):
    """One Helformer attention block: LN → MHA → Add → LN → Add."""

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model, eps=1e-6)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads,
            dropout=dropout, batch_first=True,
        )
        self.norm2 = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, d_model)
        h = self.norm1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.norm2(x)
        x = x + h
        return x


class HelformerPT(nn.Module):
    """PyTorch Helformer: Transformer blocks + LSTM → single-step forecast.

    Parameters match the original TF implementation defaults.
    """

    def __init__(
        self,
        lookback: int = 30,
        num_blocks: int = 4,
        num_heads: int = 4,
        head_size: int = 56,
        dropout: float = 0.11,
        units: int = 25,
    ):
        super().__init__()
        self.lookback = lookback
        d_model = num_heads * head_size  # 4 * 56 = 224

        self.input_proj = nn.Linear(1, d_model)
        self.blocks = nn.ModuleList([
            HelformerBlock(d_model, num_heads, dropout)
            for _ in range(num_blocks)
        ])
        self.lstm = nn.LSTM(
            input_size=d_model, hidden_size=units,
            batch_first=True, num_layers=1,
        )
        self.out = nn.Linear(units, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, L) from standard pipeline — squeeze channel dim
        if x.dim() == 3 and x.size(1) == 1:
            x = x.squeeze(1)  # (B, L)
        # x: (B, L) → (B, L, 1)
        x = x.unsqueeze(-1)
        # Project to d_model
        x = self.input_proj(x)  # (B, L, d_model)
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        # LSTM — take last hidden state
        lstm_out, _ = self.lstm(x)  # (B, L, units)
        h_last = lstm_out[:, -1, :]  # (B, units)
        # Output
        return self.out(h_last)  # (B, 1)


# ---- MinMax normalization (matching TF Helformer) ----

def _minmax_fit(series: np.ndarray) -> Tuple[float, float]:
    vmin = float(np.nanmin(series))
    vmax = float(np.nanmax(series))
    scale = vmax - vmin
    if not np.isfinite(scale) or abs(scale) < 1e-8:
        scale = 1.0
    return vmin, scale


def _minmax_transform(series: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return (series - vmin) / scale


def _minmax_inverse(series: np.ndarray, vmin: float, scale: float) -> np.ndarray:
    return series * scale + vmin


# ---- Holt-Winters baseline (reused from helformer.py) ----

def _holt_winters_baseline(
    y_tr: np.ndarray, horizon: int,
    seasonal_period: Optional[int], trend: str = "mul", seasonal: str = "mul",
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    if ExponentialSmoothing is None:
        return None
    data = np.asarray(y_tr, float).ravel()
    if data.size < 4:
        return None
    sp = seasonal_period if seasonal_period and seasonal_period > 1 else None
    s_mode = seasonal if sp else None
    t_mode = trend
    if (t_mode == "mul" or s_mode == "mul") and np.any(data <= 0):
        t_mode = "add" if t_mode == "mul" else t_mode
        s_mode = "add" if s_mode == "mul" else s_mode
    try:
        model = ExponentialSmoothing(data, trend=t_mode, seasonal=s_mode, seasonal_periods=sp)
        fitted = model.fit(optimized=True, use_brute=True)
        base_train = np.asarray(fitted.fittedvalues, float)
        if base_train.size != data.size:
            base_train = np.resize(base_train, data.size)
        base_train = np.where(np.isfinite(base_train), base_train, data)
        base_test = np.asarray(fitted.forecast(horizon), float)
        if base_test.size != horizon:
            base_test = np.resize(base_test, horizon)
        last_val = base_train[-1] if base_train.size else 0.0
        base_test = np.where(np.isfinite(base_test), base_test, last_val)
        return base_train, base_test
    except Exception:
        return None


# ---- Rollout forecast function ----

def helformer_forecast_pt(
    y_tr: np.ndarray,
    horizon: int,
    model: HelformerPT,
    lookback: int = 30,
    seasonal_period: Optional[int] = None,
    use_hw: bool = True,
    device: str = "cpu",
) -> np.ndarray:
    """Produce H-step forecast via iterative single-step rollout.

    Optionally applies Holt-Winters decomposition: the NN predicts ratios,
    which are then multiplied by the HW baseline forecast.
    """
    y_tr = np.asarray(y_tr, float).ravel()
    if horizon <= 0 or y_tr.size == 0:
        return np.zeros(max(horizon, 0), dtype=float)

    series = y_tr.copy()
    base_test = None

    if use_hw:
        bl = _holt_winters_baseline(y_tr, horizon, seasonal_period)
        if bl is not None:
            base_train, base_test = bl
            denom = np.where(np.abs(base_train) < 1e-8, 1.0, base_train)
            series = y_tr / denom
        else:
            use_hw = False

    series = np.where(np.isfinite(series), series, 0.0)
    vmin, scale = _minmax_fit(series)
    scaled = _minmax_transform(series, vmin, scale)
    history = scaled.tolist()

    model.eval()
    preds_scaled = []
    dev = torch.device(device)

    for _ in range(horizon):
        if len(history) >= lookback:
            window = history[-lookback:]
        else:
            pad = history[0] if history else 0.0
            window = [pad] * (lookback - len(history)) + history

        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(dev)
        with torch.no_grad():
            pred_val = model(x).item()

        if not np.isfinite(pred_val):
            pred_val = history[-1] if history else 0.0
        preds_scaled.append(pred_val)
        history.append(pred_val)

    preds = _minmax_inverse(np.array(preds_scaled, float), vmin, scale)

    if use_hw and base_test is not None:
        base_test = np.asarray(base_test, float).ravel()[:horizon]
        if base_test.size < horizon:
            pad = base_test[-1] if base_test.size else 1.0
            base_test = np.pad(base_test, (0, horizon - base_test.size), constant_values=pad)
        preds = preds * base_test

    return preds


__all__ = ["HelformerPT", "helformer_forecast_pt"]
