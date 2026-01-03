#!/usr/bin/env bash
set -euo pipefail

# Crystal Forge: Marginalized Q-Value Training (v3)
# Usage: ./forge/scripts/cloud-train-v3-marginalized.sh
#
# Trains a model on marginalized Q-values for robust imperfect-info play.
# - 600 train shards: 200 base seeds × 3 opponent seeds × 1 decl
# - 50 val shards: seeds 900-904 × 10 decls (golden, not marginalized)
# - 50 test shards: seeds 950-954 × 10 decls (golden, not marginalized)
# - Large config (817K + value head)
#
# Target: A100 40GB ($1.29/hr)
# Batch size: 5 (20 OOMs even on 80GB)

WORK_DIR="${WORK_DIR:-$HOME/crystal-forge}"
GROUP_ID="v3-marginalized-$(date +%Y%m%d-%H%M%S)"
BATCH_SIZE=5

cd "$WORK_DIR"

echo "=== Crystal Forge v3: Marginalized Q-Value Training ==="
echo "Wandb group: $GROUP_ID"
echo "Batch size: $BATCH_SIZE"
echo ""

# Phase 1a: Golden seeds for val/test (parallel batches of 5)
echo "Phase 1a: Generating golden seeds (val + test)..."
echo ""

echo "  Generating val seeds 900-904 (all 10 decls each)..."
for seed in 900 901 902 903 904; do
    # Run 2 decls at a time to stay under batch limit
    python -m forge.oracle.generate --seed "$seed" --decl 0,1,2,3,4 --out data/shards \
        --wandb --wandb-group "$GROUP_ID/golden" &
    python -m forge.oracle.generate --seed "$seed" --decl 5,6,7,8,9 --out data/shards \
        --wandb --wandb-group "$GROUP_ID/golden" &
    wait
done
echo "  Val seeds complete!"

echo "  Generating test seeds 950-954 (all 10 decls each)..."
for seed in 950 951 952 953 954; do
    python -m forge.oracle.generate --seed "$seed" --decl 0,1,2,3,4 --out data/shards \
        --wandb --wandb-group "$GROUP_ID/golden" &
    python -m forge.oracle.generate --seed "$seed" --decl 5,6,7,8,9 --out data/shards \
        --wandb --wandb-group "$GROUP_ID/golden" &
    wait
done
echo "  Test seeds complete!"
echo ""

# Phase 1b: Marginalized training shards
echo "Phase 1b: Generating marginalized training shards..."
python -m forge.scripts.campaign_marginalized \
    --base-seed-range 0:200 \
    --n-opp-seeds 3 \
    --batch-size "$BATCH_SIZE" \
    --out data/shards \
    --wandb-group "$GROUP_ID/marginalized"
echo ""

# Phase 1c: Verify shard counts
echo "Phase 1c: Verifying shard counts..."

# Count golden val shards: seeds 900-904 × 10 decls = 50 shards
# Pattern: seed_00000900_decl_*.parquet through seed_00000904_decl_*.parquet
VAL_COUNT=$(ls data/shards/seed_0000090[0-4]_decl_?.parquet 2>/dev/null | wc -l)
# Count golden test shards: seeds 950-954 × 10 decls = 50 shards
TEST_COUNT=$(ls data/shards/seed_0000095[0-4]_decl_?.parquet 2>/dev/null | wc -l)
# Count marginalized train shards: 200 base seeds × 3 opp seeds = 600 shards
TRAIN_COUNT=$(ls data/shards/seed_*_opp*_decl_*.parquet 2>/dev/null | wc -l)

echo "  Val shards:   $VAL_COUNT (expected 50)"
echo "  Test shards:  $TEST_COUNT (expected 50)"
echo "  Train shards: $TRAIN_COUNT (expected 600)"
echo ""

EXPECTED_VAL=50
EXPECTED_TEST=50
EXPECTED_TRAIN=600

if [ "$VAL_COUNT" -ne "$EXPECTED_VAL" ]; then
    echo "ERROR: Expected $EXPECTED_VAL val shards, got $VAL_COUNT"
    exit 1
fi
if [ "$TEST_COUNT" -ne "$EXPECTED_TEST" ]; then
    echo "ERROR: Expected $EXPECTED_TEST test shards, got $TEST_COUNT"
    exit 1
fi
if [ "$TRAIN_COUNT" -ne "$EXPECTED_TRAIN" ]; then
    echo "ERROR: Expected $EXPECTED_TRAIN train shards, got $TRAIN_COUNT"
    exit 1
fi

echo "All shard counts verified!"
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
TARBALL="$HOME/crystal-forge-v3-checkpoints-$(date +%Y%m%d-%H%M%S).tar.gz"
echo "Creating tarball: $TARBALL"
tar -czf "$TARBALL" runs/

echo ""
echo "Download with:"
echo "  scp ubuntu@<IP>:$TARBALL ."
