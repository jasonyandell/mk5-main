#!/bin/bash
# Chain STaR iterations as separate Modal runs to avoid container staleness.
# Each iteration: fresh container, load latest adapter, run, push new adapter.
#
# Usage:
#   bash lem/gemma_star/iterate.sh 5       # 5 iterations starting from 0
#   bash lem/gemma_star/iterate.sh 5 3     # 5 iterations starting from 3

set -e

N_ITERS=${1:-5}
START=${2:-0}
SUBSET=${3:-200}
DATA="lem/data/narrations_enriched.jsonl"

# Adapter chain: stage0 → star-iter0 → star-iter1 → ...
if [ "$START" -eq 0 ]; then
    ADAPTER="jasonyandell/gemma-4-e2b-texas42-stage0-v3"
else
    PREV=$((START - 1))
    ADAPTER="jasonyandell/gemma-4-e2b-texas42-star-iter${PREV}"
fi

echo "=== STaR Chain: $N_ITERS iterations, start=$START, subset=$SUBSET ==="
echo "=== Starting adapter: $ADAPTER ==="
echo "=== Data: $DATA ==="
echo ""

for i in $(seq $START $((START + N_ITERS - 1))); do
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  Launching iteration $i (adapter: $ADAPTER)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    LOG="scratch/star_iter${i}.log"

    modal run lem/gemma_star/star_loop.py \
        --iterations 1 \
        --start-iteration "$i" \
        --adapter "$ADAPTER" \
        --subset "$SUBSET" \
        --narrations "$DATA" \
        2>&1 | tee "$LOG"

    # Check if it succeeded
    if grep -q "ALL ITERATIONS COMPLETE" "$LOG"; then
        echo ""
        echo "  ✓ Iteration $i complete"
        ADAPTER="jasonyandell/gemma-4-e2b-texas42-star-iter${i}"
        echo "  Next adapter: $ADAPTER"
        echo ""
    else
        echo ""
        echo "  ✗ Iteration $i FAILED — check $LOG"
        echo "  Stopping chain."
        exit 1
    fi
done

echo ""
echo "=== ALL $N_ITERS ITERATIONS COMPLETE ==="
echo "=== Final adapter: $ADAPTER ==="
