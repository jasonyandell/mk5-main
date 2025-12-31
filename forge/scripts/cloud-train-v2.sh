#!/usr/bin/env bash
set -euo pipefail

# Crystal Forge: Value Head Training (v2)
# Usage: ./forge/scripts/cloud-train-v2.sh
#
# Trains a model with value head for MC bidding inference.
# - 200 train seeds (2x previous), decl = seed % 10
# - Val/Test: 20 seeds × 10 decls each
# - Large config (817K + value head)

WORK_DIR="$HOME/crystal-forge"
GROUP_ID="v2-valuehead-$(date +%Y%m%d-%H%M%S)"

cd "$WORK_DIR"

echo "=== Crystal Forge v2: Value Head Training ==="
echo "Wandb group: $GROUP_ID"
echo ""

# Phase 1a: Golden seeds for val/test (parallel batches)
echo "Phase 1a: Generating golden seeds (val + test)..."

echo "  Generating val seeds 900-909 (all decls)..."
python -m forge.oracle.generate --seed-range 900:910 --decl all --out data/shards &
PID1=$!

echo "  Generating val seeds 910-919 (all decls)..."
python -m forge.oracle.generate --seed-range 910:920 --decl all --out data/shards &
PID2=$!

echo "  Generating test seeds 950-959 (all decls)..."
python -m forge.oracle.generate --seed-range 950:960 --decl all --out data/shards &
PID3=$!

echo "  Generating test seeds 960-969 (all decls)..."
python -m forge.oracle.generate --seed-range 960:970 --decl all --out data/shards &
PID4=$!

wait $PID1 $PID2 $PID3 $PID4
echo "Golden seeds complete!"
echo ""

# Phase 1b: Training seeds (200 seeds, diversity strategy)
echo "Phase 1b: Generating training seeds (0-199)..."

# Batch 1: seeds 0-39 (4 parallel jobs of 10 seeds)
echo "  Batch 1: seeds 0-39..."
for offset in 0 10 20 30; do
    end=$((offset + 10))
    python -m forge.oracle.generate --seed-range ${offset}:${end} --out data/shards &
done
wait

# Batch 2: seeds 40-79
echo "  Batch 2: seeds 40-79..."
for offset in 40 50 60 70; do
    end=$((offset + 10))
    python -m forge.oracle.generate --seed-range ${offset}:${end} --out data/shards &
done
wait

# Batch 3: seeds 80-119
echo "  Batch 3: seeds 80-119..."
for offset in 80 90 100 110; do
    end=$((offset + 10))
    python -m forge.oracle.generate --seed-range ${offset}:${end} --out data/shards &
done
wait

# Batch 4: seeds 120-159
echo "  Batch 4: seeds 120-159..."
for offset in 120 130 140 150; do
    end=$((offset + 10))
    python -m forge.oracle.generate --seed-range ${offset}:${end} --out data/shards &
done
wait

# Batch 5: seeds 160-199
echo "  Batch 5: seeds 160-199..."
for offset in 160 170 180 190; do
    end=$((offset + 10))
    python -m forge.oracle.generate --seed-range ${offset}:${end} --out data/shards &
done
wait

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
