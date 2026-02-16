#!/usr/bin/env bash
# vast_up.sh — Launch Vast.ai training fleet
#
# Usage:
#   ./vast_up.sh                    # 4 workers (default)
#   ./vast_up.sh 8                  # 8 workers
#   ./vast_up.sh 4 RTX_3090         # 4 workers on 3090s
#
# Multi-model (env var overrides):
#   ZEB_WEIGHTS_NAME=large ZEB_FLEET=zeb-large ./vast_up.sh 4
#
# Environment variables:
#   ZEB_REPO_ID         HF weights repo (default: jasonyandell/zeb-42)
#   ZEB_EXAMPLES_REPO_ID  HF examples repo (default: jasonyandell/zeb-42-examples)
#   ZEB_WEIGHTS_NAME    Weights filename stem (default: zeb-557k-1m)
#   ZEB_FLEET           Instance label prefix (default: zeb)
#   ZEB_WORKER_MODE     selfplay|eval_aux (default: selfplay)
#   ZEB_UPLOAD_INTERVAL Seconds between HF uploads (default: 240)
#   ZEB_ORACLE_CHECKPOINT  Eval-aux oracle checkpoint path
#   ZEB_EQ_N_SAMPLES    Eval-aux world samples per decision
#   ZEB_EQ_GAMES_PER_BATCH Eval-aux games per batch
#   ZEB_EQ_BATCH_SIZE   Eval-aux --eval-batch-size
#   ZEB_EQ_TEMPERATURE  Eval-aux Zeb action temperature
#   ZEB_EQ_POLICY_TARGETS true|false (default false, keep policy off)
#
# Prerequisites:
#   pip install vastai
#   vastai set api-key YOUR_KEY
#   Set HF_TOKEN env var (fine-grained, write access to zeb-42 + zeb-42-examples)
#
# The learner runs locally on your 3050 Ti (free, reliable, no preemption).
# Only workers go on Vast.ai.

set -euo pipefail

N_WORKERS="${1:-4}"
GPU_FILTER="${2:-}"
REPO_ID="${ZEB_REPO_ID:-jasonyandell/zeb-42}"
EXAMPLES_REPO_ID="${ZEB_EXAMPLES_REPO_ID:-jasonyandell/zeb-42-examples}"
WEIGHTS_NAME="${ZEB_WEIGHTS_NAME:-zeb-557k-1m}"
FLEET="${ZEB_FLEET:-zeb}"
WORKER_MODE="${ZEB_WORKER_MODE:-selfplay}"
UPLOAD_INTERVAL="${ZEB_UPLOAD_INTERVAL:-240}"
ORACLE_CHECKPOINT="${ZEB_ORACLE_CHECKPOINT:-forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt}"
EQ_N_SAMPLES="${ZEB_EQ_N_SAMPLES:-100}"
EQ_GAMES_PER_BATCH="${ZEB_EQ_GAMES_PER_BATCH:-128}"
EQ_BATCH_SIZE="${ZEB_EQ_BATCH_SIZE:-0}"
EQ_TEMPERATURE="${ZEB_EQ_TEMPERATURE:-0.1}"
EQ_POLICY_TARGETS="${ZEB_EQ_POLICY_TARGETS:-false}"
IMAGE="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime"
DISK_GB=15

# Git repo URL — change if using a different repo/branch
GIT_REPO="https://github.com/jasonyandell/mk5-main.git"
GIT_BRANCH="forge"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export a fine-grained HuggingFace token."
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

case "$WORKER_MODE" in
    selfplay|eval_aux) ;;
    *)
        echo "ERROR: ZEB_WORKER_MODE must be selfplay or eval_aux (got: $WORKER_MODE)"
        exit 1
        ;;
esac

# Build GPU filter: accept any 3000-series by default
if [ -z "$GPU_FILTER" ]; then
    QUERY='gpu_name in [RTX_3060,RTX_3060_Ti,RTX_3070,RTX_3070_Ti,RTX_3080,RTX_3080_Ti,RTX_3090] num_gpus=1'
else
    QUERY="gpu_name=${GPU_FILTER} num_gpus=1"
fi

echo "Searching for ${N_WORKERS} cheapest offers..."
echo "  Filter: ${QUERY}"
echo "  Worker mode: ${WORKER_MODE}"
if [ "$WORKER_MODE" = "eval_aux" ]; then
    echo "  Eval-aux cfg: n_samples=${EQ_N_SAMPLES}, games_per_batch=${EQ_GAMES_PER_BATCH}, eval_batch_size=${EQ_BATCH_SIZE}, temp=${EQ_TEMPERATURE}, policy_targets=${EQ_POLICY_TARGETS}"
fi

# Get cheapest offers as JSON
OFFERS=$(vastai search offers "$QUERY" -o 'dph_total' --raw --limit "$N_WORKERS")
N_FOUND=$(echo "$OFFERS" | python3 -c "import sys,json; print(len(json.load(sys.stdin)))")

if [ "$N_FOUND" -lt "$N_WORKERS" ]; then
    echo "WARNING: Only found ${N_FOUND} offers (wanted ${N_WORKERS})"
    N_WORKERS="$N_FOUND"
fi

if [ "$N_WORKERS" -eq 0 ]; then
    echo "ERROR: No matching offers found."
    exit 1
fi

# Onstart script: install deps, clone repo, start worker
# All output goes to stdout (visible via `vastai logs`) AND to log files.
ONSTART='#!/bin/bash
set -e

echo "=== Setup started at $(date) ==="

# Install git and clone
apt-get update -qq && apt-get install -y -qq git > /dev/null
echo "git installed"
git clone --depth 1 --branch '"$GIT_BRANCH"' '"$GIT_REPO"' /root/code
echo "repo cloned"

# Minimal deps (torch already in image)
pip install -q huggingface_hub
echo "deps installed"

# Auth
huggingface-cli login --token $HF_TOKEN --add-to-git-credential 2>/dev/null || true

# GPU smoke test
python -c "import torch; assert torch.cuda.is_available(), \"No CUDA\"; print(f\"GPU OK: {torch.cuda.get_device_name()}\")"

# Start worker — exec replaces this shell so worker output IS container output
echo "=== Starting worker $WORKER_ID (${ZEB_WORKER_MODE:-selfplay}) at $(date) ==="
cd /root/code
if [ "${ZEB_WORKER_MODE:-selfplay}" = "eval_aux" ]; then
    EQ_POLICY_FLAG="--no-eq-policy-targets"
    if [ "${ZEB_EQ_POLICY_TARGETS:-false}" = "true" ]; then
        EQ_POLICY_FLAG="--eq-policy-targets"
    fi
    # Fallback to script-path execution if module mode is unavailable in cloned code.
    python -u -m forge.zeb.worker.run_eval_aux \
        --repo-id '"$REPO_ID"' \
        --examples-repo-id '"$EXAMPLES_REPO_ID"' \
        --worker-id "$WORKER_ID" \
        --device cuda \
        --checkpoint "${ZEB_ORACLE_CHECKPOINT:-forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt}" \
        --n-samples "${ZEB_EQ_N_SAMPLES:-100}" \
        --games-per-batch "${ZEB_EQ_GAMES_PER_BATCH:-128}" \
        --eval-batch-size "${ZEB_EQ_BATCH_SIZE:-0}" \
        --zeb-temperature "${ZEB_EQ_TEMPERATURE:-0.1}" \
        "$EQ_POLICY_FLAG" \
        --weights-name '"$WEIGHTS_NAME"' \
        --weight-sync-interval 2 \
        --upload-interval "${ZEB_UPLOAD_INTERVAL:-240}" \
    || exec python -u forge/zeb/worker/run_eval_aux.py \
        --repo-id '"$REPO_ID"' \
        --examples-repo-id '"$EXAMPLES_REPO_ID"' \
        --worker-id "$WORKER_ID" \
        --device cuda \
        --checkpoint "${ZEB_ORACLE_CHECKPOINT:-forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt}" \
        --n-samples "${ZEB_EQ_N_SAMPLES:-100}" \
        --games-per-batch "${ZEB_EQ_GAMES_PER_BATCH:-128}" \
        --eval-batch-size "${ZEB_EQ_BATCH_SIZE:-0}" \
        --zeb-temperature "${ZEB_EQ_TEMPERATURE:-0.1}" \
        "$EQ_POLICY_FLAG" \
        --weights-name '"$WEIGHTS_NAME"' \
        --weight-sync-interval 2 \
        --upload-interval "${ZEB_UPLOAD_INTERVAL:-240}"
else
    exec python -u -m forge.zeb.worker.run \
        --repo-id '"$REPO_ID"' \
        --examples-repo-id '"$EXAMPLES_REPO_ID"' \
        --worker-id "$WORKER_ID" \
        --device cuda \
        --n-parallel-games 128 \
        --n-simulations 200 \
        --max-mcts-nodes 512 \
        --games-per-batch 256 \
        --weights-name '"$WEIGHTS_NAME"' \
        --weight-sync-interval 2 \
        --upload-interval "${ZEB_UPLOAD_INTERVAL:-240}"
fi
'

echo ""
echo "Launching ${N_WORKERS} workers..."
echo "---"

for i in $(seq 0 $((N_WORKERS - 1))); do
    OFFER_ID=$(echo "$OFFERS" | python3 -c "import sys,json; print(json.load(sys.stdin)[$i]['id'])")
    GPU_NAME=$(echo "$OFFERS" | python3 -c "import sys,json; print(json.load(sys.stdin)[$i]['gpu_name'])")
    PRICE=$(echo "$OFFERS" | python3 -c "import sys,json; print(f\"{json.load(sys.stdin)[$i]['dph_total']:.3f}\")")
    WORKER_ID="worker-vast-${i}"

    echo "  Worker ${i}: ${GPU_NAME} @ \$${PRICE}/hr (offer ${OFFER_ID})"

    ENV_FLAGS="-e HF_TOKEN=${HF_TOKEN} -e WORKER_ID=${WORKER_ID} -e ZEB_WORKER_MODE=${WORKER_MODE} -e ZEB_UPLOAD_INTERVAL=${UPLOAD_INTERVAL} -e ZEB_ORACLE_CHECKPOINT=${ORACLE_CHECKPOINT} -e ZEB_EQ_N_SAMPLES=${EQ_N_SAMPLES} -e ZEB_EQ_GAMES_PER_BATCH=${EQ_GAMES_PER_BATCH} -e ZEB_EQ_BATCH_SIZE=${EQ_BATCH_SIZE} -e ZEB_EQ_TEMPERATURE=${EQ_TEMPERATURE} -e ZEB_EQ_POLICY_TARGETS=${EQ_POLICY_TARGETS}"

    vastai create instance "$OFFER_ID" \
        --image "$IMAGE" \
        --disk "$DISK_GB" \
        --label "${FLEET}-${WORKER_ID}" \
        --env "$ENV_FLAGS" \
        --onstart-cmd "$ONSTART" \
        2>&1 | sed 's/^/    /'
done

echo ""
echo "Fleet launched. Workers will take 3-5 min to start generating."
echo ""
echo "Monitor:"
echo "  vastai show instances                   # instance status"
echo "  ./vast_status.sh                        # filtered view"
echo "  vastai logs INSTANCE_ID --tail 50       # worker output"
echo "  ssh -p PORT root@HOST                   # full shell access"
