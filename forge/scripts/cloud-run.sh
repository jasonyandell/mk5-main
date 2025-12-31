#!/usr/bin/env bash
set -euo pipefail

# Crystal Forge Cloud Training Script
# Usage: ./forge/scripts/cloud-run.sh
#
# Prerequisites: Run cloud-setup.sh first!
# Runs on Lambda Labs A100 80GB (or similar)

WORK_DIR="$HOME/crystal-forge"

# Generate unique group ID for this run (used for wandb organization)
GROUP_ID="cloud-run-$(date +%Y%m%d-%H%M%S)"

echo "=== Crystal Forge Cloud Run ==="
echo "Work dir: $WORK_DIR"
echo "Wandb group: $GROUP_ID"
echo ""

# Verify we're set up
if [ ! -d "$WORK_DIR" ]; then
    echo "ERROR: Work dir not found. Run cloud-setup.sh first!"
    exit 1
fi

cd "$WORK_DIR"

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1a: Generate golden seeds (val + test, all decls)
# ─────────────────────────────────────────────────────────────────────────────
phase_generate_golden() {
    echo "=== Phase 1a: Generate Golden Seeds ==="
    echo "Seeds 900-909 (val) + 950-959 (test), all 10 decls each"
    echo "Running in 2 batches of 10..."

    cd "$WORK_DIR"
    mkdir -p data/shards

    # Batch 1: val seeds (900-909)
    echo "  Batch 1/2: val seeds 900-909..."
    for seed in $(seq 900 909); do
        python -m forge.oracle.generate --seed "$seed" --decl all --out data/shards \
            --wandb --wandb-group "$GROUP_ID/generate" &
    done
    wait
    echo "  Batch 1 complete."

    # Batch 2: test seeds (950-959)
    echo "  Batch 2/2: test seeds 950-959..."
    for seed in $(seq 950 959); do
        python -m forge.oracle.generate --seed "$seed" --decl all --out data/shards \
            --wandb --wandb-group "$GROUP_ID/generate" &
    done
    wait
    echo "  Batch 2 complete."

    echo "Golden seeds complete. Shards:"
    ls data/shards/seed_0009*.parquet 2>/dev/null | wc -l
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 1b: Generate train seeds (1 random decl per seed)
# ─────────────────────────────────────────────────────────────────────────────
phase_generate_train() {
    echo "=== Phase 1b: Generate Train Seeds ==="
    echo "Seeds 0-99, decl = seed % 10 (10 seeds per decl)"
    echo "Running in 5 batches of 20..."

    cd "$WORK_DIR"

    for batch in 0 1 2 3 4; do
        start=$((batch * 20))
        end=$((start + 20))
        echo "  Batch $((batch + 1))/5: seeds $start-$((end - 1))..."

        for seed in $(seq "$start" "$((end - 1))"); do
            decl=$((seed % 10))
            python -m forge.oracle.generate --seed "$seed" --decl "$decl" --out data/shards \
                --wandb --wandb-group "$GROUP_ID/generate" &
        done

        wait
        echo "  Batch $((batch + 1)) complete."
    done

    echo "Train seeds complete. Total shards:"
    ls data/shards/*.parquet | wc -l
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 2: Tokenize
# ─────────────────────────────────────────────────────────────────────────────
phase_tokenize() {
    echo "=== Phase 2: Tokenize ==="

    cd "$WORK_DIR"

    python -m forge.cli.tokenize --input data/shards --output data/tokenized \
        --wandb --wandb-group "$GROUP_ID/main"

    echo "Tokenization complete."
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 3: Train
# ─────────────────────────────────────────────────────────────────────────────
phase_train() {
    echo "=== Phase 3: Train ==="

    cd "$WORK_DIR"

    echo "Training baseline (73K params): 2L, 4H, d64, ff128..."
    python -m forge.cli.train \
        --epochs 20 \
        --precision bf16-mixed \
        --wandb \
        --wandb-group "$GROUP_ID/main"

    echo ""
    echo "Training medium (275K params): 3L, 6H, d96, ff256..."
    python -m forge.cli.train \
        --epochs 20 \
        --embed-dim 96 \
        --n-heads 6 \
        --n-layers 3 \
        --ff-dim 256 \
        --precision bf16-mixed \
        --wandb \
        --wandb-group "$GROUP_ID/main"

    echo "Training complete."
    echo ""
}

# ─────────────────────────────────────────────────────────────────────────────
# Phase 4: Package results
# ─────────────────────────────────────────────────────────────────────────────
phase_package() {
    echo "=== Phase 4: Package Results ==="

    cd "$WORK_DIR"

    # Find all checkpoints
    echo "Checkpoints:"
    find runs -name "*.ckpt" -type f 2>/dev/null | head -20

    # Create tarball of checkpoints
    TARBALL="$HOME/crystal-forge-checkpoints-$(date +%Y%m%d-%H%M%S).tar.gz"
    echo ""
    echo "Creating tarball: $TARBALL"
    tar -czvf "$TARBALL" runs/

    echo ""
    echo "=========================================="
    echo "DOWNLOAD YOUR MODELS:"
    echo "=========================================="
    echo ""
    echo "From your local machine, run:"
    echo ""
    echo "  scp ubuntu@\$(hostname -I | awk '{print \$1}'):$TARBALL ."
    echo ""
    echo "Or if you know the public IP:"
    echo ""
    echo "  scp ubuntu@<PUBLIC_IP>:$TARBALL ."
    echo ""
    echo "=========================================="
}

# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
main() {
    phase_generate_golden
    phase_generate_train
    phase_tokenize
    phase_train
    phase_package

    echo "=== All phases complete ==="
    echo "Check wandb for training metrics."
}

main "$@"
