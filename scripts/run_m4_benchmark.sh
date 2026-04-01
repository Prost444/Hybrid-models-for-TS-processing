#!/usr/bin/env bash
# Run M4 benchmark on beleriand (or locally).
# Usage: bash scripts/run_m4_benchmark.sh [config_name]
# Default config: m4_full_gpu.json
set -euo pipefail

CONFIG="${1:-m4_full_gpu.json}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

cd "$PROJECT_DIR"

# Detect if we're on beleriand (has nvidia-smi) or local
if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
    echo "GPU detected — using CUDA"
    DEVICE="cuda"
else
    echo "No GPU — using CPU"
    DEVICE="cpu"
fi

# Activate environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
elif [ -d "$HOME/miniconda3/envs/hybridts" ]; then
    eval "$(${HOME}/miniconda3/bin/conda shell.bash hook)"
    conda activate hybridts
fi

echo "Running: python eval_m4.py --config configs/$CONFIG"
echo "Device: $DEVICE"
echo "Start: $(date)"

python eval_m4.py --config "configs/$CONFIG"

echo "Done: $(date)"
