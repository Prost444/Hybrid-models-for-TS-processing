"""Minimal FEDformer for univariate time series forecasting.

Implements core ideas from Zhou et al. (ICML 2022):
  - Seasonal-trend decomposition at each layer
  - Frequency-enhanced attention via random Fourier features

This is a simplified univariate variant for M3/M4 benchmarks.

Reference:
    Zhou T. et al. "FEDformer: Frequency Enhanced Decomposed Transformer
    for Long-term Series Forecasting", ICML 2022.
    https://arxiv.org/abs/2201.12740
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
#  Series Decomposition (same approach as Autoformer)
# ---------------------------------------------------------------------------

class SeriesDecomp(nn.Module):
    """Moving-average decomposition into trend + seasonal."""

    def __init__(self, kernel_size: int = 25):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=pad)

    def forward(self, x: torch.Tensor):
        """x: (B, L, D) -> seasonal (B, L, D), trend (B, L, D)"""
        trend = self.avg(x.transpose(1, 2)).transpose(1, 2)
        if trend.size(1) > x.size(1):
            trend = trend[:, :x.size(1), :]
        seasonal = x - trend
        return seasonal, trend


# ---------------------------------------------------------------------------
#  Frequency-Enhanced Attention (Fourier variant)
# ---------------------------------------------------------------------------

class FrequencyAttention(nn.Module):
    """Attention in the frequency domain via random Fourier feature selection.

    Keeps only a subset of Fourier modes to achieve linear complexity
    and frequency-enhanced representation.
    """

    def __init__(self, d_model: int, n_heads: int, n_modes: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.n_modes = n_modes
        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor,
                value: torch.Tensor) -> torch.Tensor:
        """All inputs: (B, L, D). Returns (B, L, D)."""
        B, L, _ = query.shape
        H, d = self.n_heads, self.d_head

        Q = self.Wq(query).view(B, L, H, d).permute(0, 2, 1, 3)  # (B,H,L,d)
        K = self.Wk(key).view(B, L, H, d).permute(0, 2, 1, 3)
        V = self.Wv(value).view(B, L, H, d).permute(0, 2, 1, 3)

        # Transform to frequency domain
        n_freq = L // 2 + 1
        modes = min(self.n_modes, n_freq)

        Q_fft = torch.fft.rfft(Q, dim=2)[:, :, :modes, :]   # (B,H,modes,d)
        K_fft = torch.fft.rfft(K, dim=2)[:, :, :modes, :]
        V_fft = torch.fft.rfft(V, dim=2)[:, :, :modes, :]

        # Frequency-domain attention: element-wise product (like spectral conv)
        # Weight = softmax(Q_fft * K_fft.conj) over modes
        attn_freq = Q_fft * K_fft.conj()
        attn_weight = F.softmax(attn_freq.real, dim=2)  # (B,H,modes,d)
        out_fft = attn_weight * V_fft  # (B,H,modes,d)

        # Pad back to full frequency domain and inverse FFT
        full_fft = torch.zeros(B, H, n_freq, d, dtype=out_fft.dtype,
                               device=out_fft.device)
        full_fft[:, :, :modes, :] = out_fft
        out = torch.fft.irfft(full_fft, n=L, dim=2)  # (B,H,L,d)

        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, -1)
        return self.dropout(self.Wo(out))


# ---------------------------------------------------------------------------
#  Encoder Layer
# ---------------------------------------------------------------------------

class FEDformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int,
                 kernel_size: int = 25, n_modes: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.attn = FrequencyAttention(d_model, n_heads, n_modes=n_modes,
                                       dropout=dropout)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, L, D) -> (B, L, D)"""
        attn_out = self.attn(x, x, x)
        x = x + attn_out
        x, _ = self.decomp1(self.norm1(x))

        ff_out = self.ff(x)
        x = x + ff_out
        x, _ = self.decomp2(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
#  Full Model
# ---------------------------------------------------------------------------

class FEDformer(nn.Module):
    """Simplified univariate FEDformer.

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
    n_modes : int
        Number of Fourier modes kept in frequency attention (default 32).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, lookback: int, horizon: int, *,
                 d_model: int = 64, n_heads: int = 4, e_layers: int = 2,
                 d_ff: int = 128, kernel_size: int = 25, n_modes: int = 32,
                 dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon

        self.input_proj = nn.Linear(1, d_model)

        self.encoder = nn.ModuleList([
            FEDformerEncoderLayer(d_model, n_heads, d_ff, kernel_size,
                                 n_modes, dropout)
            for _ in range(e_layers)
        ])

        self.output_proj = nn.Linear(lookback * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) or (B, L) -> (B, H)"""
        if x.dim() == 3:
            x = x.squeeze(1)
        B, L = x.shape
        h = self.input_proj(x.unsqueeze(-1))
        for layer in self.encoder:
            h = layer(h)
        return self.output_proj(h.reshape(B, -1))
