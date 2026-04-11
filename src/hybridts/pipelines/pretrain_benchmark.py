"""Pre-train / fine-tune benchmark pipeline.

Pre-trains neural models on ALL series from a single category (e.g. all 1428
M3-monthly series), then fine-tunes and evaluates on a random subset.
This measures whether cross-series pre-training helps generalisation.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from ..config.settings import settings
from ..data import (
    M3_H, M3_P, M4_H, M4_P, best_L,
    ensure_m3_csv, load_train_tsts,
    ensure_m4_csv, load_m4_train_test,
    seasonal_naive,
    smape, mape, mse, rmse, mae, mase,
)
from ..models import make_model
from ..training import (
    TrainConfig, WindowDatasetStd, MultiSeriesWindowDataset, train_model,
)


def _progress(it, **kw):
    return tqdm(it, **kw) if tqdm is not None else it


# ---- Metric computation ---------------------------------------------------

def _compute_metrics(y_te: np.ndarray, pred: np.ndarray,
                     y_tr: np.ndarray, P: int) -> Dict[str, float]:
    return {
        "sMAPE": smape(y_te, pred),
        "MAPE": mape(y_te, pred),
        "RMSE": rmse(y_te, pred),
        "MSE": mse(y_te, pred),
        "MAE": mae(y_te, pred),
        "MASE": mase(y_te, pred, y_tr, P),
    }


# ---- Inference helper ------------------------------------------------------

def _forecast_from_model(
    model: torch.nn.Module, y_tr: np.ndarray, L: int, H: int,
    mu: float, sd: float, device: str,
) -> np.ndarray:
    """Direct H-step forecast using a trained model."""
    model.eval()
    # Pad or truncate y_tr to length L
    if len(y_tr) >= L:
        window = y_tr[-L:]
    else:
        pad_val = y_tr[0] if len(y_tr) > 0 else 0.0
        window = np.concatenate([np.full(L - len(y_tr), pad_val), y_tr])

    with torch.no_grad():
        x = torch.tensor(window, dtype=torch.float32)
        x_norm = (x - mu) / sd
        x_norm = x_norm.unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, L)
        pred_norm = model(x_norm).squeeze(0).cpu().numpy()

    return pred_norm * sd + mu


# ---- Main entry point -----------------------------------------------------

def run_pretrain_benchmark(
    pretrain_category: str = "monthly",
    pretrain_dataset: str = "m3",
    eval_n: int = 30,
    eval_pick: str = "random",
    seed: int = 42,
    pretrain_epochs: int = 50,
    finetune_epochs: int = 20,
    pretrain_models: Sequence[str] = ("dlinear", "nbeats", "timesnet", "autoformer", "fedformer", "patchtst", "helformer"),
    batch_size: int = 256,
    device: str = "cuda",
    out_dir: str | Path | None = None,
    max_pretrain_series: int = 0,
) -> pd.DataFrame:
    """Run a pre-train / fine-tune benchmark.

    1. Load ALL series from ``pretrain_category``.
    2. Compute an average lookback ``L_avg`` across all series.
    3. Create a :class:`MultiSeriesWindowDataset` from all training arrays.
    4. For each model, pre-train on the multi-series dataset.
    5. Select ``eval_n`` series (random sample) for evaluation.
    6. For each eval series, fine-tune from pretrained weights and compare
       with a from-scratch baseline.

    Parameters
    ----------
    pretrain_category : str
        The frequency category to use for pre-training (e.g. "monthly").
    pretrain_dataset : str
        "m3" or "m4".
    eval_n : int
        Number of series to evaluate on.
    eval_pick : str
        "random", "first", or "last".
    seed : int
        Random seed.
    pretrain_epochs : int
        Number of pre-training epochs.
    finetune_epochs : int
        Number of fine-tuning epochs per eval series.
    pretrain_models : sequence of str
        Model names to pre-train (must be supported by make_model).
    batch_size : int
        Batch size for pre-training.
    device : str
        PyTorch device.
    out_dir : Path
        Output directory.
    max_pretrain_series : int
        If >0, subsample the pre-training set to at most this many series
        (useful for large datasets like M4). 0 means use all.

    Returns
    -------
    pd.DataFrame with per-series, per-model metrics for pretrain+finetune
    and from-scratch.
    """
    out_path = Path(out_dir or (
        settings.outputs_dir / f"pretrain_{pretrain_dataset}_{pretrain_category}_benchmark"
    ))
    out_path.mkdir(parents=True, exist_ok=True)

    # Dataset-specific setup
    if pretrain_dataset == "m3":
        ensure_m3_csv()
        H_MAP, P_MAP = M3_H, M3_P
        load_fn = load_train_tsts
    elif pretrain_dataset == "m4":
        ensure_m4_csv(categories=(pretrain_category,))
        H_MAP, P_MAP = M4_H, M4_P
        load_fn = load_m4_train_test
    else:
        raise ValueError(f"Unknown dataset: {pretrain_dataset}")

    H = H_MAP[pretrain_category]
    P = P_MAP[pretrain_category]

    rng = np.random.default_rng(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # 1. Load ALL series from the category
    all_pairs = load_fn(pretrain_category)
    if not all_pairs:
        print(f"[{pretrain_category}] no data found")
        return pd.DataFrame()

    print(f"[pretrain] category={pretrain_category}, dataset={pretrain_dataset}")
    print(f"[pretrain] total series={len(all_pairs)}, H={H}, P={P}")

    # 2. Compute average lookback
    L_values = []
    all_y_tr = []
    for sid, y_tr, y_te in all_pairs:
        L_values.append(best_L(y_tr, H, P))
        all_y_tr.append(y_tr)
    L_avg = int(np.mean(L_values))
    L_avg = max(L_avg, H + 1)  # safety floor
    print(f"[pretrain] L_avg={L_avg} (min={min(L_values)}, max={max(L_values)})")

    # 2b. Optionally subsample pre-training series
    if max_pretrain_series > 0 and len(all_y_tr) > max_pretrain_series:
        rng_sub = np.random.default_rng(seed)
        sub_idx = rng_sub.choice(len(all_y_tr), size=max_pretrain_series, replace=False)
        all_y_tr = [all_y_tr[i] for i in sorted(sub_idx)]
        print(f"[pretrain] subsampled to {len(all_y_tr)} series for pretrain")

    # 3. Create MultiSeriesWindowDataset
    print(f"[pretrain] building multi-series dataset from {len(all_y_tr)} series...")
    multi_ds = MultiSeriesWindowDataset(all_y_tr, lookback=L_avg, horizon=H)
    print(f"[pretrain] total windows={len(multi_ds)}")

    if len(multi_ds) < 1:
        print("[pretrain] not enough windows for pre-training")
        return pd.DataFrame()

    # 4. Select eval series
    if eval_n <= 0:
        eval_pairs = all_pairs
    elif eval_pick == "first":
        eval_pairs = all_pairs[:eval_n]
    elif eval_pick == "last":
        eval_pairs = all_pairs[-eval_n:]
    else:
        count = min(eval_n, len(all_pairs))
        idx = rng.choice(len(all_pairs), size=count, replace=False)
        eval_pairs = [all_pairs[int(i)] for i in sorted(idx)]

    print(f"[pretrain] eval series={len(eval_pairs)}")

    rows: List[Dict] = []
    checkpoint_csv = out_path / "metrics_checkpoint.csv"
    t0 = time.time()

    # Resume from checkpoint
    if checkpoint_csv.exists():
        prev = pd.read_csv(checkpoint_csv)
        rows = prev.to_dict("records")
        done_keys = {(r["model"], r["series_id"]) for r in rows}
        print(f"[checkpoint] resuming -- {len(rows)} records already done")
    else:
        done_keys = set()

    # 5. For each model: pretrain, then evaluate
    for model_name in _progress(pretrain_models, desc="Models"):
        print(f"\n[pretrain] pre-training {model_name} for {pretrain_epochs} epochs...")

        # Special handling for Helformer (single-step model)
        is_helformer = model_name == "helformer"
        pretrain_horizon = 1 if is_helformer else H

        # Create model and pre-train on multi-series data
        pretrain_cfg = TrainConfig(
            lookback=L_avg, horizon=pretrain_horizon, epochs=pretrain_epochs,
            batch_size=batch_size, lr=1e-3, weight_decay=1e-4,
            clip=1.0, device=device,
        )

        try:
            pretrained_model = make_model(model_name, pretrain_cfg)
            if is_helformer:
                multi_ds_hf = MultiSeriesWindowDataset(all_y_tr, lookback=L_avg, horizon=1)
                train_model(pretrained_model, multi_ds_hf, pretrain_cfg)
            else:
                train_model(pretrained_model, multi_ds, pretrain_cfg)
        except Exception as exc:
            print(f"[pretrain] {model_name} pre-training failed: {exc}")
            continue

        # Save pretrained state dict
        pretrained_state = copy.deepcopy(pretrained_model.state_dict())
        weights_path = out_path / f"{model_name}_pretrained.pt"
        torch.save(pretrained_state, weights_path)
        print(f"[pretrain] saved pretrained weights: {weights_path}")

        # 6. Evaluate on each eval series
        for sid, y_tr, y_te in _progress(eval_pairs, desc=f"{model_name} eval", leave=False):
            # --- Pretrained + Fine-tuned ---
            if (model_name, sid) not in done_keys:
                rec: Dict[str, Any] = {
                    "model": model_name,
                    "series_id": sid,
                    "category": pretrain_category,
                    "strategy": "pretrain_finetune",
                }
                try:
                    # Create fresh model, load pretrained weights
                    ft_horizon = 1 if is_helformer else H
                    ft_cfg = TrainConfig(
                        lookback=L_avg, horizon=ft_horizon, epochs=finetune_epochs,
                        batch_size=32, lr=5e-4, weight_decay=1e-4,
                        clip=1.0, device=device,
                    )
                    ft_model = make_model(model_name, ft_cfg)
                    ft_model.load_state_dict(pretrained_state)

                    # Per-series dataset using L_avg for dimension compatibility
                    # Pad y_tr if shorter than L_avg
                    ft_ds_horizon = 1 if is_helformer else H
                    if len(y_tr) < L_avg + ft_ds_horizon:
                        pad_val = y_tr[0] if len(y_tr) > 0 else 0.0
                        y_tr_padded = np.concatenate([
                            np.full(L_avg + ft_ds_horizon - len(y_tr), pad_val), y_tr
                        ])
                    else:
                        y_tr_padded = y_tr

                    ft_ds = WindowDatasetStd(y_tr_padded, L_avg, ft_ds_horizon)

                    if len(ft_ds) >= 1:
                        train_model(ft_model, ft_ds, ft_cfg)
                        if is_helformer:
                            from ..models.helformer_pt import helformer_forecast_pt
                            pred = helformer_forecast_pt(
                                y_tr, H, ft_model, lookback=L_avg,
                                seasonal_period=P, use_hw=True, device=device,
                            )
                        else:
                            mu, sd = ft_ds.scaler
                            pred = _forecast_from_model(ft_model, y_tr, L_avg, H, mu, sd, device)
                        pred = np.asarray(pred, float).ravel()[:H]
                        metrics = _compute_metrics(y_te, pred, y_tr, P)
                        for mk, mv in metrics.items():
                            rec[mk] = mv
                    else:
                        print(f"[{model_name}:{sid}] not enough data for fine-tuning")

                except Exception as exc:
                    print(f"[{model_name}:{sid}] pretrain+finetune failed: {exc}")

                rows.append(rec)
                pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)

            # --- From-scratch baseline ---
            scratch_key = (f"{model_name}_scratch", sid)
            if scratch_key not in done_keys:
                rec_scratch: Dict[str, Any] = {
                    "model": model_name,
                    "series_id": sid,
                    "category": pretrain_category,
                    "strategy": "from_scratch",
                }
                try:
                    scratch_epochs = pretrain_epochs + finetune_epochs  # fair comparison
                    scratch_horizon = 1 if is_helformer else H
                    scratch_cfg = TrainConfig(
                        lookback=L_avg, horizon=scratch_horizon, epochs=scratch_epochs,
                        batch_size=32, lr=1e-3, weight_decay=1e-4,
                        clip=1.0, device=device,
                    )
                    scratch_model = make_model(model_name, scratch_cfg)

                    if len(y_tr) < L_avg + scratch_horizon:
                        pad_val = y_tr[0] if len(y_tr) > 0 else 0.0
                        y_tr_padded = np.concatenate([
                            np.full(L_avg + scratch_horizon - len(y_tr), pad_val), y_tr
                        ])
                    else:
                        y_tr_padded = y_tr

                    scratch_ds = WindowDatasetStd(y_tr_padded, L_avg, scratch_horizon)

                    if len(scratch_ds) >= 1:
                        train_model(scratch_model, scratch_ds, scratch_cfg)
                        if is_helformer:
                            from ..models.helformer_pt import helformer_forecast_pt
                            pred = helformer_forecast_pt(
                                y_tr, H, scratch_model, lookback=L_avg,
                                seasonal_period=P, use_hw=True, device=device,
                            )
                        else:
                            mu, sd = scratch_ds.scaler
                            pred = _forecast_from_model(
                                scratch_model, y_tr, L_avg, H, mu, sd, device,
                            )
                        pred = np.asarray(pred, float).ravel()[:H]
                        metrics = _compute_metrics(y_te, pred, y_tr, P)
                        for mk, mv in metrics.items():
                            rec_scratch[mk] = mv
                    else:
                        print(f"[{model_name}:{sid}] not enough data from scratch")

                except Exception as exc:
                    print(f"[{model_name}:{sid}] from-scratch failed: {exc}")

                rows.append(rec_scratch)
                pd.DataFrame(rows).to_csv(checkpoint_csv, index=False)

    df = pd.DataFrame(rows)
    metrics_csv = out_path / "metrics.csv"
    df.to_csv(metrics_csv, index=False)
    elapsed = time.time() - t0
    print(f"\n[saved] {metrics_csv}  ({len(df)} records, {elapsed:.0f}s)")

    # Print summary: pretrain+finetune vs from_scratch
    if not df.empty:
        metric_names = ["sMAPE", "MAPE", "RMSE", "MSE", "MAE", "MASE"]
        for mname in metric_names:
            if mname not in df.columns:
                continue
            print(f"\n--- {mname} by model and strategy ---")
            pivot = df.groupby(["model", "strategy"])[mname].mean()
            print(pivot.round(3).to_string())

        print(f"\n--- Overall comparison ---")
        for strategy in ("pretrain_finetune", "from_scratch"):
            sub = df[df["strategy"] == strategy]
            if sub.empty:
                continue
            avail = [m for m in metric_names if m in sub.columns]
            if avail:
                means = sub[avail].mean(numeric_only=True).round(3)
                print(f"  {strategy}:")
                print(f"    {means.to_string()}")

    return df


__all__ = ["run_pretrain_benchmark"]
