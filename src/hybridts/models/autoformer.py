"""Minimal Autoformer for univariate time series forecasting.

Implements the core ideas from Wu et al. (NeurIPS 2021):
  - Progressive series decomposition (trend + seasonal at each layer)
  - Auto-Correlation mechanism replacing canonical self-attention

This is a *simplified* univariate variant suitable for M3/M4 benchmarks.
For the full multivariate implementation see https://github.com/thuml/Autoformer.

Reference:
    Wu H. et al. "Autoformer: Decomposition Transformers with Auto-Correlation
    for Long-Term Series Forecasting", NeurIPS 2021.
    https://arxiv.org/abs/2106.13008
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Series Decomposition (moving-average trend extraction)
# ---------------------------------------------------------------------------

class SeriesDecomp(nn.Module):
    """Extracts trend via moving average; remainder = input - trend."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        self.kernel_size = kernel_size
        pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, x: torch.Tensor):
        """x: (B, L, D) -> trend (B, L, D), remainder (B, L, D)"""
        # AvgPool1d expects (B, C, L)
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        if trend.size(1) > x.size(1):
            trend = trend[:, :x.size(1), :]
        remainder = x - trend
        return remainder, trend


# ---------------------------------------------------------------------------
#  Auto-Correlation Mechanism
# ---------------------------------------------------------------------------

class AutoCorrelation(nn.Module):
    """Period-based dependency discovery via autocorrelation in freq. domain."""

    def __init__(self, d_model: int, n_heads: int, topk: int = 3, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.topk = topk
        self.d_head = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor):
        """All inputs: (B, L, D). Returns (B, L, D)."""
        B, L, _ = query.shape
        H, d = self.n_heads, self.d_head

        Q = self.Wq(query).view(B, L, H, d).permute(0, 2, 1, 3)  # (B,H,L,d)
        K = self.Wk(key).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.Wv(value).view(B, L, H, d).permute(0, 2, 1, 3)

        # Auto-correlation via FFT
        Q_fft = torch.fft.rfft(Q, dim=2)
        K_fft = torch.fft.rfft(K, dim=2)
        corr = torch.fft.irfft(Q_fft * K_fft.conj(), n=L, dim=2)  # (B,H,L,d)

        # Top-k delay aggregation — clamp k to sequence length to avoid
        # "selected index k out of range" on very short yearly series.
        k = min(self.topk, L)
        corr_mean = corr.mean(dim=-1)  # (B,H,L)
        topk_vals, topk_idx = torch.topk(corr_mean, k, dim=-1)  # (B,H,k)
        weights = F.softmax(topk_vals, dim=-1)  # (B,H,k)

        # Roll V by top-k delays and aggregate (fully vectorized)
        out = torch.zeros_like(V)
        # Create index tensor for gather-based roll (no Python loops)
        arange = torch.arange(L, device=V.device).view(1, 1, L, 1)  # (1,1,L,1)
        for i in range(k):
            delay = topk_idx[:, :, i].unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
            w = weights[:, :, i].unsqueeze(-1).unsqueeze(-1)  # (B,H,1,1)
            # Compute rolled indices: (idx - delay) % L
            rolled_idx = (arange - delay) % L  # (B,H,L,1)
            rolled_idx = rolled_idx.expand(-1, -1, -1, d)  # (B,H,L,d)
            rolled_V = torch.gather(V, 2, rolled_idx)  # (B,H,L,d)
            out = out + w * rolled_V

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.dropout(self.Wo(out))


# ---------------------------------------------------------------------------
#  Encoder Layer
# ---------------------------------------------------------------------------

class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 kernel_size: int = 25, topk: int = 3, dropout: float = 0.1):
        super().__init__()
        self.attn = AutoCorrelation(d_model, n_heads, topk=topk, dropout=dropout)
        self.decomp1 = SeriesDecomp(kernel_size)
        self.decomp2 = SeriesDecomp(kernel_size)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """x: (B, L, D) -> (B, L, D)"""
        # Auto-Correlation + decomposition
        attn_out = self.attn(x, x, x)
        x = x + attn_out
        x, _ = self.decomp1(self.norm1(x))

        # Feed-forward + decomposition
        ff_out = self.ff(x)
        x = x + ff_out
        x, _ = self.decomp2(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
#  Full Model
# ---------------------------------------------------------------------------

class Autoformer(nn.Module):
    """Simplified univariate Autoformer.

    Parameters
    ----------
    lookback : int
        Input sequence length.
    horizon : int
        Forecast horizon.
    d_model : int
        Hidden dimension (default 64).
    n_heads : int
        Number of attention heads (default 4).
    e_layers : int
        Number of encoder layers (default 2).
    d_ff : int
        Feed-forward hidden dim (default 128).
    kernel_size : int
        Moving-average kernel for decomposition (default 25).
    topk : int
        Top-k delays in Auto-Correlation (default 3).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, lookback: int, horizon: int, *,
                 d_model: int = 64, n_heads: int = 4, e_layers: int = 2,
                 d_ff: int = 128, kernel_size: int = 25, topk: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon

        # Input projection: 1 (univariate) -> d_model
        self.input_proj = nn.Linear(1, d_model)

        # Encoder
        self.encoder = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, kernel_size, topk, dropout)
            for _ in range(e_layers)
        ])

        # Output projection: pool encoded representation and project to horizon
        self.output_proj = nn.Linear(lookback * d_model, horizon)

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
        B, L = x.shape

        # (B, L) -> (B, L, 1) -> (B, L, d_model)
        h = self.input_proj(x.unsqueeze(-1))

        for layer in self.encoder:
            h = layer(h)

        # Flatten and project to horizon
        h = h.reshape(B, -1)  # (B, L * d_model)
        return self.output_proj(h)  # (B, H)
