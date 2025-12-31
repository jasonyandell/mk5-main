#!/usr/bin/env bash
set -euo pipefail

# Crystal Forge: Value Head Training (v2)
# Usage: ./forge/scripts/cloud-train-v2.sh
#
# Trains a model with value head for MC bidding inference.
# - 200 train seeds (2x previous), decl = seed % 10
# - Val/Test: 10 seeds × 10 decls each (200 golden shards)
# - Large config (817K + value head)

WORK_DIR="$HOME/crystal-forge"
GROUP_ID="v2-valuehead-$(date +%Y%m%d-%H%M%S)"

cd "$WORK_DIR"

echo "=== Crystal Forge v2: Value Head Training ==="
echo "Wandb group: $GROUP_ID"
echo ""

# Phase 1a: Golden seeds for val/test (parallel)
echo "Phase 1a: Generating golden seeds (val + test)..."

echo "  Generating val seeds 900-909 (all decls)..."
python -m forge.oracle.generate --seed-range 900:910 --decl all --out data/shards &
PID1=$!

echo "  Generating test seeds 950-959 (all decls)..."
python -m forge.oracle.generate --seed-range 950:960 --decl all --out data/shards &
PID2=$!

wait $PID1 $PID2
echo "Golden seeds complete!"
echo ""

# Phase 1b: Training seeds (200 seeds, diversity strategy: decl = seed % 10)
echo "Phase 1b: Generating training seeds (0-199, 1 decl per seed)..."

for batch in 0 1 2 3 4 5 6 7 8 9; do
    start=$((batch * 20))
    end=$((start + 20))
    echo "  Batch $((batch + 1))/10: seeds $start-$((end - 1))..."

    for seed in $(seq "$start" "$((end - 1))"); do
        decl=$((seed % 10))
        python -m forge.oracle.generate --seed "$seed" --decl "$decl" --out data/shards &
    done
    wait
done

echo "Training seeds complete!"
echo ""

# Verify shard count
SHARD_COUNT=$(ls data/shards/*.parquet 2>/dev/null | wc -l)
echo "Total shards: $SHARD_COUNT"
echo ""

# Phase 2: Tokenize (now includes V for value head)
echo "Phase 2: Tokenizing..."
python -m forge.cli.tokenize \
    --input data/shards \
    --output data/tokenized \
    --wandb \
    --wandb-group "$GROUP_ID"

echo ""

# Phase 3: Train Large model with value head
echo "Phase 3: Training Large model with value head..."
python -m forge.cli.train \
    --epochs 20 \
    --embed-dim 128 \
    --n-heads 8 \
    --n-layers 4 \
    --ff-dim 512 \
    --value-weight 0.5 \
    --precision bf16-mixed \
    --no-compile \
    --wandb \
    --wandb-group "$GROUP_ID"

echo ""
echo "=== Training Complete ==="

# Package checkpoints
TARBALL="$HOME/crystal-forge-v2-checkpoints-$(date +%Y%m%d-%H%M%S).tar.gz"
echo "Creating tarball: $TARBALL"
tar -czf "$TARBALL" runs/

echo ""
echo "Download with:"
echo "  scp ubuntu@<IP>:$TARBALL ."
