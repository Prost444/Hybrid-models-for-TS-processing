"""DLinear — learnable decomposition baseline (Zeng et al., AAAI 2023).

Decomposes the input via a moving-average kernel into trend and remainder,
then applies one linear layer per component.  Despite its simplicity it
matches or outperforms many Transformer-based forecasters.

Reference:
    Zeng A. et al. "Are Transformers Effective for Time Series Forecasting?"
    AAAI 2023.  https://arxiv.org/abs/2205.13504
"""
from __future__ import annotations

import torch
import torch.nn as nn


class MovingAvg(nn.Module):
    """Moving-average kernel for trend extraction."""

    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L) or (B, 1, L)."""
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, L)
        trend = self.avg(x)
        # Trim if kernel_size is even (AvgPool1d may produce L+1)
        if trend.size(-1) > x.size(-1):
            trend = trend[..., : x.size(-1)]
        return trend.squeeze(1)  # (B, L)


class DLinear(nn.Module):
    """DLinear forecaster.

    Parameters
    ----------
    lookback : int
        Input window length.
    horizon : int
        Forecast horizon.
    kernel_size : int
        Moving-average kernel for trend–remainder decomposition (default 25).
    """

    def __init__(self, lookback: int, horizon: int, *, kernel_size: int = 25):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.decomp = MovingAvg(kernel_size)
        self.linear_trend = nn.Linear(lookback, horizon)
        self.linear_remainder = nn.Linear(lookback, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, L) or (B, L)

        Returns
        -------
        forecast : (B, H)
        """
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, L)
        trend = self.decomp(x)             # (B, L)
        remainder = x - trend              # (B, L)
        return self.linear_trend(trend) + self.linear_remainder(remainder)  # (B, H)
