#!/usr/bin/env bash
# Sync local project to beleriand server via git.
# Usage: bash scripts/sync_to_beleriand.sh
set -euo pipefail

LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE="beleriand"
REMOTE_DIR="~/projects/Hybrid-models-for-TS-processing"

echo "=== Syncing to beleriand ==="

# Push latest changes to GitHub
cd "$LOCAL_DIR"
echo "[1/3] Pushing to GitHub..."
git add -A
git diff --cached --quiet && echo "  (nothing to commit)" || \
    git commit -m "sync: pre-beleriand checkpoint $(date +%Y%m%d_%H%M%S)"
git push origin HEAD 2>/dev/null || echo "  (push skipped or failed — check remote)"

# Pull on beleriand
echo "[2/3] Pulling on beleriand..."
ssh "$REMOTE" "cd $REMOTE_DIR && git pull origin HEAD"

# Sync data that may not be in git (large files)
echo "[3/3] Syncing data files..."
rsync -avz --progress \
    --exclude '.venv' \
    --exclude '__pycache__' \
    --exclude '.git' \
    --exclude 'archive' \
    --exclude '*.pyc' \
    "$LOCAL_DIR/src/data/" \
    "$REMOTE:$REMOTE_DIR/src/data/"

echo "=== Done ==="
