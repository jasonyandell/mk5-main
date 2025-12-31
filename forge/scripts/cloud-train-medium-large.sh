#!/usr/bin/env bash
set -euo pipefail

# Crystal Forge: Train Medium + Large
# Usage: ./forge/scripts/cloud-train-medium-large.sh

WORK_DIR="$HOME/crystal-forge"
GROUP_ID="cloud-run-$(date +%Y%m%d-%H%M%S)"

cd "$WORK_DIR"

echo "=== Crystal Forge Training (Medium + Large) ==="
echo "Wandb group: $GROUP_ID"
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
echo "Training large (817K params): 4L, 8H, d128, ff512..."
python -m forge.cli.train \
    --epochs 20 \
    --embed-dim 128 \
    --n-heads 8 \
    --n-layers 4 \
    --ff-dim 512 \
    --precision bf16-mixed \
    --no-compile \
    --wandb \
    --wandb-group "$GROUP_ID"

echo ""
echo "=== Training Complete ==="

# Package checkpoints
TARBALL="$HOME/crystal-forge-checkpoints-$(date +%Y%m%d-%H%M%S).tar.gz"
echo "Creating tarball: $TARBALL"
tar -czf "$TARBALL" runs/

echo ""
echo "Download with:"
echo "  scp ubuntu@<IP>:$TARBALL ."
