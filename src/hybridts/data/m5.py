"""Utilities for loading M5 Competition dataset (Walmart daily sales).

M5 contains 30,490 univariate daily time series with 1941 days of history.
Horizon H=28 days, seasonal period P=7 (weekly).

Data format: wide CSV with columns item_id, dept_id, cat_id, store_id, state_id, d_1..d_1941
Test set: last 28 days (d_1914..d_1941 for evaluation split).
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config.settings import settings

M5_H: int = 28      # forecast horizon (28 days)
M5_P: int = 7       # seasonal period (weekly)
M5_TRAIN_DAYS: int = 1913  # d_1 through d_1913 (evaluation split)
M5_TEST_DAYS: int = 28     # d_1914 through d_1941


def _m5_data_dir() -> Path:
    """Locate M5 data directory."""
    candidates = [
        settings.data_dir / "m5",
        settings.project_root / "src" / "data" / "m5",
    ]
    for d in candidates:
        if d.exists() and (d / "sales_train_evaluation.csv").exists():
            return d
    raise FileNotFoundError(
        "M5 data not found. Expected sales_train_evaluation.csv in "
        + " or ".join(str(d) for d in candidates)
    )


def load_m5_train_test(
    max_series: int = 0,
    min_nonzero_frac: float = 0.1,
    seed: int = 42,
) -> List[Tuple[str, np.ndarray, np.ndarray]]:
    """Load M5 series with train/test split.

    Parameters
    ----------
    max_series : int
        Maximum number of series to load. 0 = all 30,490.
    min_nonzero_frac : float
        Minimum fraction of non-zero values in training data.
        Filters out very sparse series (many M5 series are mostly zeros).
    seed : int
        Random seed for subsampling.

    Returns
    -------
    List of (series_id, y_train, y_test) tuples.
    """
    data_dir = _m5_data_dir()
    csv_path = data_dir / "sales_train_evaluation.csv"

    print(f"[m5] loading {csv_path}...")
    df = pd.read_csv(csv_path)

    # Extract metadata and value columns
    meta_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in df.columns if c.startswith("d_")]
    day_cols_sorted = sorted(day_cols, key=lambda c: int(c.split("_")[1]))

    # Split into train (d_1..d_1913) and test (d_1914..d_1941)
    train_cols = [c for c in day_cols_sorted if int(c.split("_")[1]) <= M5_TRAIN_DAYS]
    test_cols = [c for c in day_cols_sorted if int(c.split("_")[1]) > M5_TRAIN_DAYS]

    if len(test_cols) < M5_H:
        raise ValueError(
            f"Expected {M5_H} test days, found {len(test_cols)}. "
            "Check that sales_train_evaluation.csv has d_1914..d_1941."
        )

    # Build series list
    series_ids = df["item_id"] + "_" + df["store_id"]
    train_vals = df[train_cols].values.astype(np.float32)
    test_vals = df[test_cols[:M5_H]].values.astype(np.float32)

    pairs = []
    for i in range(len(df)):
        y_tr = train_vals[i]
        y_te = test_vals[i]

        # Filter sparse series
        if min_nonzero_frac > 0:
            nonzero_frac = np.count_nonzero(y_tr) / len(y_tr)
            if nonzero_frac < min_nonzero_frac:
                continue

        # Skip series that are all zeros
        if y_tr.sum() == 0:
            continue

        pairs.append((str(series_ids.iloc[i]), y_tr, y_te))

    print(f"[m5] loaded {len(pairs)} series (after filtering, from {len(df)} total)")

    # Subsample if requested
    if max_series > 0 and len(pairs) > max_series:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_series, replace=False)
        pairs = [pairs[int(i)] for i in sorted(idx)]
        print(f"[m5] subsampled to {len(pairs)} series")

    return pairs


__all__ = ["M5_H", "M5_P", "load_m5_train_test"]
