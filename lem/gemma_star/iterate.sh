#!/bin/bash
# Run one STaR iteration: inference → train → push adapter
#
# Usage:
#   ./lem/gemma_star/iterate.sh 0                    # Iteration 0 (uses stage0 adapter)
#   ./lem/gemma_star/iterate.sh 1                    # Iteration 1 (uses iter0 adapter)
#   ./lem/gemma_star/iterate.sh 0 --limit 20         # Quick test with 20 examples
#
set -euo pipefail

ITER=${1:?Usage: iterate.sh <iteration_number> [--limit N]}
shift
EXTRA_ARGS="$@"

ADAPTER_BASE="jasonyandell/gemma-4-e2b-texas42"
NARRATIONS="lem/data/narrations_train.jsonl"
DATA_DIR="lem/data"

# Determine input adapter
if [ "$ITER" -eq 0 ]; then
    INPUT_ADAPTER="${ADAPTER_BASE}-stage0"
else
    PREV=$((ITER - 1))
    INPUT_ADAPTER="${ADAPTER_BASE}-star-iter${PREV}"
fi

OUTPUT_TRACES="${DATA_DIR}/star_iter${ITER}.jsonl"
OUTPUT_ADAPTER="${ADAPTER_BASE}-star-iter${ITER}"

echo "=== STaR Iteration ${ITER} ==="
echo "  Input adapter:  ${INPUT_ADAPTER}"
echo "  Output traces:  ${OUTPUT_TRACES}"
echo "  Output adapter: ${OUTPUT_ADAPTER}"
echo ""

# Phase 1: Inference + grading + rationalization
echo "[phase 1] Running inference on Modal..."
modal run lem/gemma_star/star_harness.py \
    --narrations "${NARRATIONS}" \
    --adapter "${INPUT_ADAPTER}" \
    --output "${OUTPUT_TRACES}" \
    ${EXTRA_ARGS}

N_TRACES=$(wc -l < "${OUTPUT_TRACES}")
echo "[phase 1] Got ${N_TRACES} training traces"

# Phase 2: Train LoRA on traces
echo ""
echo "[phase 2] Training LoRA on Modal A100..."

# The training script expects a Q&A corpus format, but STaR traces are
# chat messages. We need to use the traces directly as SFT data.
# For now, we'll use the existing train_stage0.py infrastructure but
# point it at the STaR traces.
#
# TODO: Build a dedicated STaR training function that reads the traces
# format directly. For iteration 0, we can at least validate the loop
# manually.

echo "[phase 2] NOTE: Manual training step needed — see lem/gemma_star/iterate.sh"
echo "  Traces at: ${OUTPUT_TRACES}"
echo "  Train with: modal run lem/gemma_star/train_stage0.py --corpus ${OUTPUT_TRACES}"
echo ""
echo "[done] Iteration ${ITER} inference complete. Train manually, then run iteration $((ITER + 1))."
