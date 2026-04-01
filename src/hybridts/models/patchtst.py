"""Minimal PatchTST for univariate time series forecasting.

Implements core ideas from Nie et al. (ICLR 2023):
  - Subseries-level patching (analogous to ViT image patches)
  - Standard Transformer encoder on patch tokens
  - Channel independence (trivial for univariate)

This is a simplified univariate variant for M3/M4 benchmarks.

Reference:
    Nie Y. et al. "A Time Series is Worth 64 Words: Long-term Forecasting
    with Transformers", ICLR 2023.
    https://arxiv.org/abs/2211.14730
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchTST(nn.Module):
    """Simplified univariate PatchTST.

    Parameters
    ----------
    lookback : int
        Input sequence length.
    horizon : int
        Forecast horizon.
    patch_len : int
        Length of each patch (default 16).
    stride : int
        Stride between consecutive patches (default 8).
    d_model : int
        Hidden dimension (default 64).
    n_heads : int
        Number of attention heads (default 4).
    e_layers : int
        Number of encoder layers (default 2).
    d_ff : int
        Feed-forward hidden dim (default 128).
    dropout : float
        Dropout rate (default 0.1).
    """

    def __init__(self, lookback: int, horizon: int, *,
                 patch_len: int = 16, stride: int = 8,
                 d_model: int = 64, n_heads: int = 4, e_layers: int = 2,
                 d_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.lookback = lookback
        self.horizon = horizon
        self.patch_len = patch_len
        self.stride = stride

        # Compute number of patches — pad input if needed
        self.n_patches = max(1, (lookback - patch_len) // stride + 1)
        # Effective input length after patching
        self.padded_len = (self.n_patches - 1) * stride + patch_len

        # Patch embedding: project each patch to d_model
        self.patch_embed = nn.Linear(patch_len, d_model)

        # Learnable positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, self.n_patches, d_model) * 0.02)

        # Standard Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        # Output head: flatten all patch tokens and project to horizon
        self.head = nn.Linear(self.n_patches * d_model, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) or (B, L) -> (B, H)"""
        if x.dim() == 3:
            x = x.squeeze(1)  # (B, L)
        B, L = x.shape

        # Pad if input shorter than expected
        if L < self.padded_len:
            pad_size = self.padded_len - L
            x = nn.functional.pad(x, (pad_size, 0), mode='replicate')
        elif L > self.padded_len:
            x = x[:, -self.padded_len:]

        # Extract patches: (B, n_patches, patch_len)
        patches = x.unfold(dimension=1, size=self.patch_len, step=self.stride)

        # Truncate to expected number of patches
        patches = patches[:, :self.n_patches, :]

        # Embed patches
        h = self.patch_embed(patches) + self.pos_embed  # (B, n_patches, d_model)

        # Transformer encoder
        h = self.encoder(h)  # (B, n_patches, d_model)

        # Flatten and project
        return self.head(h.reshape(B, -1))  # (B, H)
