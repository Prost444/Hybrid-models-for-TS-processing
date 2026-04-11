#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
eval "$(~/miniconda3/bin/conda shell.bash hook)" && conda activate hybridts

echo "=== M3 YEARLY ===" && python eval_pretrain.py --config configs/pretrain_m3_yearly_gpu.json 2>&1 | tee pretrain_m3_yearly.log
echo "=== M3 QUARTERLY ===" && python eval_pretrain.py --config configs/pretrain_m3_quarterly_gpu.json 2>&1 | tee pretrain_m3_quarterly.log
echo "=== M3 MONTHLY ===" && python eval_pretrain.py --config configs/pretrain_m3_monthly_gpu.json 2>&1 | tee pretrain_m3_monthly.log
echo "=== ALL M3 DONE ==="
