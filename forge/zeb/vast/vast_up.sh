#!/usr/bin/env bash
# vast_up.sh — Launch Vast.ai training fleet
#
# Usage:
#   ./vast_up.sh                    # 4 workers (default)
#   ./vast_up.sh 8                  # 8 workers
#   ./vast_up.sh 4 RTX_3090         # 4 workers on 3090s
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
REPO_ID="jasonyandell/zeb-42"
EXAMPLES_REPO_ID="jasonyandell/zeb-42-examples"
IMAGE="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime"
DISK_GB=15

# Git repo URL — change if using a different repo/branch
GIT_REPO="https://github.com/jasonyandell/v2.git"
GIT_BRANCH="forge"

if [ -z "${HF_TOKEN:-}" ]; then
    echo "ERROR: HF_TOKEN not set. Export a fine-grained HuggingFace token."
    echo "  export HF_TOKEN=hf_..."
    exit 1
fi

# Build GPU filter: accept any 3000-series by default
if [ -z "$GPU_FILTER" ]; then
    QUERY='gpu_name in [RTX_3060,RTX_3060_Ti,RTX_3070,RTX_3070_Ti,RTX_3080,RTX_3080_Ti,RTX_3090] reliability>0.95 rentable=true num_gpus=1 gpu_ram>=8000'
else
    QUERY="gpu_name=${GPU_FILTER} reliability>0.95 rentable=true num_gpus=1"
fi

echo "Searching for ${N_WORKERS} cheapest offers..."
echo "  Filter: ${QUERY}"

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
echo "=== Starting worker $WORKER_ID at $(date) ==="
cd /root/code
exec python -u -m forge.zeb.worker.run \
    --repo-id '"$REPO_ID"' \
    --examples-repo-id '"$EXAMPLES_REPO_ID"' \
    --worker-id $WORKER_ID \
    --device cuda \
    --n-parallel-games 128 \
    --n-simulations 200 \
    --max-mcts-nodes 512 \
    --games-per-batch 256 \
    --weight-sync-interval 2
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

    vastai create instance "$OFFER_ID" \
        --image "$IMAGE" \
        --disk "$DISK_GB" \
        --label "zeb-${WORKER_ID}" \
        --env "-e HF_TOKEN=${HF_TOKEN} -e WORKER_ID=${WORKER_ID}" \
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
