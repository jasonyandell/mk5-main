#!/usr/bin/env bash
set -euo pipefail

# Crystal Forge Cloud Training (train-only)
# Usage: ./forge/scripts/cloud-train.sh
#
# Assumes data already tokenized. Just runs training.

WORK_DIR="$HOME/crystal-forge"
GROUP_ID="cloud-run-$(date +%Y%m%d-%H%M%S)"

cd "$WORK_DIR"

echo "=== Crystal Forge Training ==="
echo "Wandb group: $GROUP_ID"
echo ""

echo "Training baseline (73K params): 2L, 4H, d64, ff128..."
python -m forge.cli.train \
    --epochs 20 \
    --precision bf16-mixed \
    --no-compile \
    --wandb \
    --wandb-group "$GROUP_ID"

echo ""
echo "Training medium (275K params): 3L, 6H, d96, ff256..."
python -m forge.cli.train \
    --epochs 20 \
    --embed-dim 96 \
    --n-heads 6 \
    --n-layers 3 \
    --ff-dim 256 \
    --precision bf16-mixed \
    --no-compile \
    --wandb \
    --wandb-group "$GROUP_ID"

echo ""
echo "=== Training Complete ==="
echo ""

# Package checkpoints
TARBALL="$HOME/crystal-forge-checkpoints-$(date +%Y%m%d-%H%M%S).tar.gz"
echo "Creating tarball: $TARBALL"
tar -czf "$TARBALL" runs/

echo ""
echo "Download with:"
echo "  scp ubuntu@<IP>:$TARBALL ."
