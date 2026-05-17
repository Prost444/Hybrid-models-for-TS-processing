#!/usr/bin/env python3
"""Fill missing TimesNet/Helformer cells in parallel_m4_1000 checkpoint.

Supports sharded parallel execution on multiple GPUs:

    # Single GPU (original mode):
    CUDA_VISIBLE_DEVICES=1 python fill_missing_daily.py --model timesnet --epochs 50

    # Sharded across 4 GPUs:
    CUDA_VISIBLE_DEVICES=0 python fill_missing_daily.py --model helformer --epochs 50 --shard 0 --num-shards 4
    CUDA_VISIBLE_DEVICES=3 python fill_missing_daily.py --model helformer --epochs 50 --shard 1 --num-shards 4
    CUDA_VISIBLE_DEVICES=4 python fill_missing_daily.py --model helformer --epochs 50 --shard 2 --num-shards 4
    CUDA_VISIBLE_DEVICES=6 python fill_missing_daily.py --model helformer --epochs 50 --shard 3 --num-shards 4
"""
import argparse
import os
import sys
import time
import filelock
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hybridts.data import (
    M4_H, M4_P, best_L,
    ensure_m4_csv, load_m4_train_test,
    smape, mape, mse, rmse, mae, mase,
)
from hybridts.models import make_model
from hybridts.models.helformer_pt import helformer_forecast_pt
from hybridts.training import TrainConfig, WindowDatasetStd, train_model

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None


CHECKPOINT = Path("outputs/parallel_m4_1000/metrics_checkpoint.csv")
METRICS = ["sMAPE", "MAPE", "RMSE", "MSE", "MAE", "MASE"]


def compute_metrics(y_te, pred, y_tr, P):
    return {
        "sMAPE": smape(y_te, pred),
        "MAPE": mape(y_te, pred),
        "RMSE": rmse(y_te, pred),
        "MSE": mse(y_te, pred),
        "MAE": mae(y_te, pred),
        "MASE": mase(y_te, pred, y_tr, P),
    }


def train_and_forecast(model_name, y_tr, H, P, epochs, device):
    L = best_L(y_tr, H, P)
    n_windows = len(y_tr) - L - H + 1
    bs = 256 if n_windows > 1024 else (128 if n_windows > 256 else 32)
    cfg = TrainConfig(lookback=L, horizon=H, epochs=epochs, batch_size=bs,
                      lr=1e-3, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model(model_name, cfg)
    ds = WindowDatasetStd(y_tr, L, H)
    if len(ds) < 1:
        return None
    train_model(model, ds, cfg)
    mu, sd = ds.scaler
    model.eval()
    with torch.no_grad():
        x = torch.tensor(y_tr[-L:], dtype=torch.float32)
        x_norm = (x - mu) / sd
        x_in = x_norm.unsqueeze(0).unsqueeze(0).to(cfg.device)
        pred_norm = model(x_in).squeeze(0).cpu().numpy()
    return np.asarray(pred_norm * sd + mu, float).ravel()[:H]


def predict_helformer(y_tr, H, P, epochs, device):
    L = min(30, len(y_tr) - 2)
    if L < 4:
        L = max(4, len(y_tr) // 2)
    cfg = TrainConfig(lookback=L, horizon=1, epochs=epochs, batch_size=32,
                      lr=8e-4, weight_decay=1e-4, clip=1.0, device=device)
    model = make_model("helformer", cfg)
    vmin = float(np.nanmin(y_tr))
    scale = float(np.nanmax(y_tr)) - vmin
    if scale < 1e-8:
        scale = 1.0
    scaled = (y_tr - vmin) / scale
    ds = WindowDatasetStd(scaled, L, 1, scale=False)
    if len(ds) < 1:
        return None
    train_model(model, ds, cfg)
    return helformer_forecast_pt(y_tr, H, model, lookback=L,
                                 seasonal_period=P, use_hw=True, device=device)


def get_missing_series(df, model_name):
    """Get list of daily series_id where model metrics are NaN."""
    daily = df[df["category"] == "daily"]
    col = f"{model_name}_sMAPE"
    missing = daily[daily[col].isna()]["series_id"].tolist()
    return missing


def merge_save(results, model_name, lock_path):
    """Thread-safe merge-and-save: read fresh CSV, update only our cells, write back."""
    if not results:
        return
    lock = filelock.FileLock(str(lock_path))
    with lock:
        df = pd.read_csv(CHECKPOINT)
        for sid, metrics in results.items():
            idx = df.index[(df["category"] == "daily") & (df["series_id"] == sid)]
            if len(idx) > 0:
                for mk, mv in metrics.items():
                    df.loc[idx[0], f"{model_name}_{mk}"] = mv
        df.to_csv(CHECKPOINT, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=["timesnet", "helformer"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--save-every", type=int, default=20,
                        help="Save checkpoint every N series")
    parser.add_argument("--shard", type=int, default=0,
                        help="Shard index (0-based)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="Total number of shards (1 = no sharding)")
    parser.add_argument("--start-idx", type=int, default=0,
                        help="Start index in missing series list (for GPU/CPU split)")
    parser.add_argument("--end-idx", type=int, default=None,
                        help="End index in missing series list (exclusive)")
    args = parser.parse_args()

    model_name = args.model
    device = args.device
    epochs = args.epochs
    shard = args.shard
    num_shards = args.num_shards

    tag = f"[shard {shard}/{num_shards}]" if num_shards > 1 else "[fill_missing]"

    print(f"{tag} model={model_name}, device={device}, epochs={epochs}")
    print(f"{tag} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")

    if device == "cuda" and torch.cuda.is_available():
        print(f"{tag} GPU: {torch.cuda.get_device_name(0)}")
    elif device == "cuda":
        print(f"{tag} WARNING: CUDA not available, falling back to CPU")
        device = "cpu"

    # Load M4 daily data
    ensure_m4_csv(categories=("daily",))
    all_pairs = load_m4_train_test("daily")
    pair_dict = {sid: (y_tr, y_te) for sid, y_tr, y_te in all_pairs}
    print(f"{tag} loaded {len(pair_dict)} daily series")

    H = M4_H["daily"]
    P = M4_P["daily"]

    # Load checkpoint and find missing
    lock_path = CHECKPOINT.parent / ".checkpoint.lock"
    df = pd.read_csv(CHECKPOINT)
    missing_ids = get_missing_series(df, model_name)
    print(f"{tag} {len(missing_ids)} series missing for {model_name}")

    if not missing_ids:
        print(f"{tag} nothing to do!")
        return

    # Filter to only series we have data for
    available = [sid for sid in missing_ids if sid in pair_dict]

    # Slice by start/end index (for GPU/CPU split)
    end_idx = args.end_idx if args.end_idx is not None else len(available)
    available = available[args.start_idx:end_idx]

    # Apply sharding: each shard gets every num_shards-th series
    if num_shards > 1:
        available = [sid for i, sid in enumerate(available) if i % num_shards == shard]

    print(f"{tag} {len(available)} series to process (this shard, idx {args.start_idx}:{end_idx})")

    if not available:
        print(f"{tag} nothing to do for this shard!")
        return

    filled = 0
    errors = 0
    t0 = time.time()
    # Buffer for merge-save: {series_id: {metric: value}}
    pending_results = {}

    desc = f"fill {model_name}" if num_shards == 1 else f"fill {model_name} s{shard}"
    iterator = tqdm(available, desc=desc) if tqdm else available

    for i, sid in enumerate(iterator):
        y_tr, y_te = pair_dict[sid]

        try:
            if model_name == "helformer":
                pred = predict_helformer(y_tr, H, P, epochs, device)
            else:
                pred = train_and_forecast(model_name, y_tr, H, P, epochs, device)

            if pred is not None:
                pred = np.asarray(pred, float).ravel()[:H]
                metrics = compute_metrics(y_te, pred, y_tr, P)
                pending_results[sid] = metrics
                filled += 1
            else:
                errors += 1

        except Exception as e:
            errors += 1
            if tqdm:
                pass  # tqdm handles display
            else:
                print(f"  ERROR on {sid}: {e}")

        # Periodic checkpoint: merge-save accumulated results
        if (i + 1) % args.save_every == 0:
            merge_save(pending_results, model_name, lock_path)
            pending_results = {}
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            remaining = (len(available) - i - 1) / rate if rate > 0 else 0
            print(f"  {tag} [checkpoint] {filled}/{i+1} filled, {errors} errors, "
                  f"{elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining")

        # Clear GPU cache periodically
        if device == "cuda" and (i + 1) % 5 == 0:
            torch.cuda.empty_cache()

    # Final save
    merge_save(pending_results, model_name, lock_path)

    elapsed = time.time() - t0
    print(f"\n{tag} DONE: {filled}/{len(available)} filled, "
          f"{errors} errors, {elapsed:.0f}s total")


if __name__ == "__main__":
    main()
