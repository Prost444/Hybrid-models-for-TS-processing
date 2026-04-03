"""Dataset helpers used during neural network training."""
from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


class WindowDatasetStd(Dataset):
    """Standard sliding window dataset with optional z-score scaling."""

    def __init__(self, y, lookback: int, horizon: int, stride: int = 1, scale: bool = True):
        y = np.asarray(y, np.float32)
        if scale:
            mu = float(y.mean())
            sd = float(y.std()) + 1e-8
            z = (y - mu) / sd
        else:
            mu, sd, z = 0.0, 1.0, y
        X, Y = [], []
        for t in range(0, len(z) - lookback - horizon + 1, stride):
            X.append(z[t : t + lookback])
            Y.append(z[t + lookback : t + lookback + horizon])
        self.X = np.asarray(X, np.float32)
        self.Y = np.asarray(Y, np.float32)
        self._scaler = (mu, sd)

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self.X)

    def __getitem__(self, idx: int):  # pragma: no cover - thin wrapper
        return (
            torch.from_numpy(self.X[idx]).unsqueeze(0),
            torch.from_numpy(self.Y[idx]),
        )

    @property
    def scaler(self) -> Tuple[float, float]:
        return self._scaler


class MultiSeriesWindowDataset(Dataset):
    """Sliding-window dataset built from *multiple* series (for pre-training).

    Each series is z-score normalised independently, then sliding windows from
    all series are concatenated into a single flat dataset so that a model can
    be pre-trained on a whole category at once.

    Parameters
    ----------
    series_list : list[np.ndarray]
        List of 1-D time-series arrays.
    lookback : int
        Input window length.
    horizon : int
        Target (forecast) window length.
    stride : int
        Step between consecutive windows (default 1).
    """

    def __init__(
        self,
        series_list: list,
        lookback: int,
        horizon: int,
        stride: int = 1,
    ):
        all_X, all_Y = [], []
        self._scalers: list[Tuple[float, float]] = []

        for y_raw in series_list:
            y = np.asarray(y_raw, np.float32)
            mu = float(y.mean())
            sd = float(y.std()) + 1e-8
            z = (y - mu) / sd
            self._scalers.append((mu, sd))
            for t in range(0, len(z) - lookback - horizon + 1, stride):
                all_X.append(z[t : t + lookback])
                all_Y.append(z[t + lookback : t + lookback + horizon])

        self.X = np.asarray(all_X, np.float32) if all_X else np.empty((0, lookback), np.float32)
        self.Y = np.asarray(all_Y, np.float32) if all_Y else np.empty((0, horizon), np.float32)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int):
        return (
            torch.from_numpy(self.X[idx]).unsqueeze(0),  # (1, L)
            torch.from_numpy(self.Y[idx]),                # (H,)
        )

    @property
    def scaler(self) -> Tuple[float, float]:
        """Return the first series' scaler for backward-compat with single-series code."""
        if self._scalers:
            return self._scalers[0]
        return (0.0, 1.0)

    @property
    def scalers(self) -> list:
        return list(self._scalers)


__all__ = ["WindowDatasetStd", "MultiSeriesWindowDataset"]
