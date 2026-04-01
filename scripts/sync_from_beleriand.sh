#!/usr/bin/env bash
# Pull results from beleriand back to local.
# Usage: bash scripts/sync_from_beleriand.sh
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="beleriand"
REMOTE_DIR="~/projects/Hybrid-models-for-TS-processing"

echo "=== Syncing results from beleriand ==="
rsync -avz --progress \
    "$REMOTE:$REMOTE_DIR/src/outputs/" \
    "$LOCAL_DIR/src/outputs/"

echo "=== Done ==="
