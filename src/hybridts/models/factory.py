"""Model factory functions for experiment scripts.

The pipelines can optionally pass per-model parameter overrides (via JSON),
so experiments can scale model capacity without code changes.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch.nn as nn

from .autoformer import Autoformer
from .dlinear import DLinear
from .fedformer import FEDformer
from .helformer_pt import HelformerPT
from .nbeats import NBEATSV2
from .patchtst import PatchTST
from .timesnet import TimesNetV2
from ..training.engine import TrainConfig


def _safe_kernel(kernel_size: int, lookback: int) -> int:
    """Ensure the MA decomposition kernel fits within the lookback window."""
    # Kernel must be odd and ≤ lookback.  For very short windows just use 3.
    k = min(kernel_size, lookback)
    if k < 3:
        k = 3
    if k % 2 == 0:
        k -= 1
    return max(3, k)


def _pick_params(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    params = dict(defaults)
    if overrides:
        for key, value in overrides.items():
            if key not in defaults:
                raise ValueError(f"Unsupported model param '{key}'. Allowed: {sorted(defaults)}")
            params[key] = value
    return params


def make_model(name: str, cfg: TrainConfig, *, params: Mapping[str, Any] | None = None) -> nn.Module:
    name = name.lower()
    if name == "timesnet":
        is_monthly = cfg.horizon >= 18
        defaults = {
            "d_model": 128 if is_monthly else 96,
            "layers": 6 if is_monthly else 5,
            "topk": 5,
            "dropout": 0.12,
            "use_anchor": True,
        }
        p = _pick_params(defaults, params)
        return TimesNetV2(cfg.lookback, cfg.horizon, **p)
    if name == "nbeats":
        is_monthly = cfg.horizon >= 18
        defaults = {
            "width": 768 if is_monthly else 640,
            "depth": 8 if is_monthly else 6,
            "nblocks": 12 if is_monthly else 10,
            "dropout": 0.15,
            "use_anchor": False,
            "share_weights": True,
            "use_trend": True,
            "use_seasonality": True,
            "use_generic": True,
            "diff_loss_weight": 0.08,
        }
        p = _pick_params(defaults, params)
        return NBEATSV2(cfg.lookback, cfg.horizon, **p)
    if name == "dlinear":
        defaults = {
            "kernel_size": 25,
        }
        p = _pick_params(defaults, params)
        # Clamp kernel_size so it doesn't exceed the lookback window.
        # On yearly M3 series lookback can be as small as 1–8.
        p["kernel_size"] = _safe_kernel(p["kernel_size"], cfg.lookback)
        return DLinear(cfg.lookback, cfg.horizon, **p)
    if name == "autoformer":
        is_monthly = cfg.horizon >= 18
        defaults = {
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 128,
            "kernel_size": 25,
            "topk": 3,
            "dropout": 0.1,
        }
        p = _pick_params(defaults, params)
        p["kernel_size"] = _safe_kernel(p["kernel_size"], cfg.lookback)
        return Autoformer(cfg.lookback, cfg.horizon, **p)
    if name == "fedformer":
        defaults = {
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 128,
            "kernel_size": 25,
            "n_modes": 32,
            "dropout": 0.1,
        }
        p = _pick_params(defaults, params)
        p["kernel_size"] = _safe_kernel(p["kernel_size"], cfg.lookback)
        p["n_modes"] = min(p["n_modes"], cfg.lookback // 2 + 1)
        return FEDformer(cfg.lookback, cfg.horizon, **p)
    if name == "patchtst":
        defaults = {
            "patch_len": 16,
            "stride": 8,
            "d_model": 64,
            "n_heads": 4,
            "e_layers": 2,
            "d_ff": 128,
            "dropout": 0.1,
        }
        p = _pick_params(defaults, params)
        # Adapt patch_len and stride for short series
        p["patch_len"] = min(p["patch_len"], max(2, cfg.lookback))
        p["stride"] = min(p["stride"], max(1, p["patch_len"] // 2))
        return PatchTST(cfg.lookback, cfg.horizon, **p)
    if name == "helformer":
        defaults = {
            "num_blocks": 4,
            "num_heads": 4,
            "head_size": 56,
            "dropout": 0.11,
            "units": 25,
        }
        p = _pick_params(defaults, params)
        return HelformerPT(lookback=cfg.lookback, **p)
    raise ValueError(f"Unknown model '{name}'")


__all__ = ["make_model"]
