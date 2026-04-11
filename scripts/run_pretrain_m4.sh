#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate hybridts

echo "=== M4 QUARTERLY ===" && python eval_pretrain.py --config configs/pretrain_m4_quarterly_gpu.json 2>&1 | tee pretrain_m4_quarterly.log
echo "=== M4 MONTHLY ===" && python eval_pretrain.py --config configs/pretrain_m4_monthly_gpu.json 2>&1 | tee pretrain_m4_monthly.log
echo "=== M4 DAILY ===" && python eval_pretrain.py --config configs/pretrain_m4_daily_gpu.json 2>&1 | tee pretrain_m4_daily.log
echo "=== ALL M4 DONE ==="
