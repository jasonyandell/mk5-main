#!/usr/bin/env bash
# vast_monitor.sh — Autonomous self-healing fleet manager
#
# Long-running process that maintains a target number of healthy workers.
# Replaces the manual workflow: vast_up → vast_status → vast_replenish → vast_down
#
# Usage:
#   ./vast_monitor.sh                # 2 workers (default)
#   ./vast_monitor.sh 4              # 4 workers
#   ./vast_monitor.sh 4 --dry-run    # show config, don't create/destroy
#
# Environment variables (same as vast_up.sh):
#   ZEB_FLEET               Instance label prefix (default: zeb)
#   ZEB_WEIGHTS_NAME        Weights filename stem (default: zeb-557k-1m)
#   ZEB_REPO_ID             HF weights repo (default: jasonyandell/zeb-42)
#   ZEB_EXAMPLES_REPO_ID    HF examples repo (default: jasonyandell/zeb-42-examples)
#   ZEB_MAX_DPH             Max cost per worker $/hr (default: 0.09)
#   ZEB_GPUS                Space-separated GPU list (default: RTX_3060 RTX_3060_Ti ... RTX_3090)
#
# Ctrl-C to stop — runs vast_down.sh to tear down the fleet on exit.

set -uo pipefail

# ─── Parse CLI args ───
TARGET_WORKERS=2
DRY_RUN=""
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=1 ;;
        [0-9]*)    TARGET_WORKERS="$arg" ;;
        *)         echo "Usage: vast_monitor.sh [N_WORKERS] [--dry-run]"; exit 1 ;;
    esac
done

# ─── Config from env vars ───
FLEET="${ZEB_FLEET:-zeb}"
WEIGHTS_NAME="${ZEB_WEIGHTS_NAME:-zeb-557k-1m}"
REPO_ID="${ZEB_REPO_ID:-jasonyandell/zeb-42}"
EXAMPLES_REPO_ID="${ZEB_EXAMPLES_REPO_ID:-jasonyandell/zeb-42-examples}"
MAX_DPH="${ZEB_MAX_DPH:-0.09}"
GPUS="${ZEB_GPUS:-RTX_3060 RTX_3060_Ti RTX_3070 RTX_3070_Ti RTX_3080 RTX_3080_Ti RTX_3090}"

IMAGE="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime"
GIT_REPO="https://github.com/jasonyandell/mk5-main.git"
GIT_BRANCH="forge"

PHASE1_INTERVAL=60
PHASE2_INTERVAL=600
PHASE1_DURATION=1800  # 30 min

# ─── Colors ───
if [ -t 1 ]; then
    GREEN=$'\033[32m'
    YELLOW=$'\033[33m'
    RED=$'\033[31m'
    BOLD=$'\033[1m'
    RESET=$'\033[0m'
else
    GREEN="" YELLOW="" RED="" BOLD="" RESET=""
fi

# ─── Helpers ───
log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

color_tag() {
    local tag="$1" text="$2"
    case "$tag" in
        OK)    printf '%s' "${GREEN}${text}${RESET}" ;;
        SETUP|BOOT|WAIT) printf '%s' "${YELLOW}${text}${RESET}" ;;
        *)     printf '%s' "${RED}${text}${RESET}" ;;
    esac
}

fmt_duration() {
    local s="$1"
    if [ "$s" -ge 3600 ]; then
        printf '%dh%02dm' $((s / 3600)) $(((s % 3600) / 60))
    elif [ "$s" -ge 60 ]; then
        printf '%dm%02ds' $((s / 60)) $((s % 60))
    else
        printf '%ds' "$s"
    fi
}

# ─── HF_TOKEN check ───
if [ -z "${HF_TOKEN:-}" ]; then
    # Try to read from cache
    if [ -f ~/.cache/huggingface/token ]; then
        HF_TOKEN=$(cat ~/.cache/huggingface/token)
        export HF_TOKEN
    else
        echo "${RED}ERROR: HF_TOKEN not set and ~/.cache/huggingface/token not found.${RESET}"
        echo "  export HF_TOKEN=hf_..."
        exit 1
    fi
fi

# ─── Bootstrap check: verify weights exist on HF ───
bootstrap_check() {
    log "Checking HF repo for ${WEIGHTS_NAME}.pt..."
    if python3 -c "
from huggingface_hub import hf_hub_url, get_hf_file_metadata
import os
os.environ.setdefault('HF_TOKEN', '$HF_TOKEN')
try:
    url = hf_hub_url('$REPO_ID', '${WEIGHTS_NAME}.pt')
    get_hf_file_metadata(url, token='$HF_TOKEN')
    print('OK')
except Exception as e:
    print(f'MISSING: {e}')
    exit(1)
" 2>/dev/null; then
        return 0
    else
        echo ""
        echo "${RED}${BOLD}ERROR: ${WEIGHTS_NAME}.pt not found in ${REPO_ID}${RESET}"
        echo ""
        echo "The learner must run first to seed the HF repo with initial weights."
        echo "Start the learner locally:"
        echo ""
        echo "  python -u -m forge.zeb.learner.run \\"
        echo "      --repo-id ${REPO_ID} \\"
        echo "      --examples-repo-id ${EXAMPLES_REPO_ID} \\"
        echo "      --weights-name ${WEIGHTS_NAME} \\"
        echo "      --checkpoint YOUR_CHECKPOINT.pt \\"
        echo "      --lr 1e-4 --batch-size 64 --wandb"
        echo ""
        return 1
    fi
}

# ─── Fleet query ───
get_fleet() {
    vastai show instances --raw 2>/dev/null | python3 -c "
import json, sys, time
now = time.time()
fleet = '$FLEET'
for i in json.load(sys.stdin):
    label = i.get('label') or ''
    if not label.startswith(fleet + '-'): continue
    iid = i.get('id', '?')
    gpu = i.get('gpu_name', '?')
    st = i.get('actual_status', '?')
    dph = i.get('dph_total', 0)
    age = int(now - i['start_date']) if i.get('start_date') else 0
    print(f'{iid}\t{label}\t{gpu}\t{st}\t{dph:.3f}\t{age}')
"
}

# ─── Worker health check ───
check_worker() {
    local iid="$1"
    local raw
    raw=$(vastai logs "$iid" --tail 20 2>/dev/null) || { echo "ERR:no_logs"; return; }

    # Fatal errors (skip "No such container" during boot — that's normal)
    local fatal
    fatal=$(echo "$raw" | grep -iE '(Exception|Traceback|killed|OOM|size mismatch)' | tail -1 | cut -c1-80)
    if [ -n "$fatal" ]; then
        echo "ERR:$fatal"
        return
    fi

    # Batch progress: "batch N: XXXX examples, X.X games/s"
    local batch_line
    batch_line=$(echo "$raw" | grep -oE 'batch [0-9]+:.*games/s' | tail -1)
    if [ -n "$batch_line" ]; then
        echo "OK:$batch_line"
        return
    fi

    # Still setting up?
    if echo "$raw" | grep -qE '(cloning|installed|Setup started|Pulling initial|repo cloned|deps installed|GPU OK)' 2>/dev/null; then
        echo "SETUP"
        return
    fi

    # "No such container" = Docker image pulling, not an error
    if echo "$raw" | grep -qE 'No such container' 2>/dev/null; then
        echo "BOOT"
        return
    fi

    echo "WAIT"
}

# ─── Find cheapest offer across GPU types ───
# Queries each GPU individually to avoid vastai 'in' operator bugs
find_cheapest_offer() {
    python3 -c "
import subprocess, json, sys

gpus = '$GPUS'.split()
max_dph = $MAX_DPH
best = None

for gpu in gpus:
    try:
        r = subprocess.run(
            ['vastai', 'search', 'offers', f'gpu_name={gpu} num_gpus=1',
             '-o', 'dph_total', '--raw', '--limit', '1'],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode != 0: continue
        offers = json.loads(r.stdout)
        if not offers: continue
        o = offers[0]
        if o['dph_total'] <= max_dph:
            if best is None or o['dph_total'] < best['dph_total']:
                best = o
    except Exception:
        pass

if best:
    print(f\"{best['id']}\t{best['gpu_name']}\t{best['dph_total']:.3f}\")
    sys.exit(0)
else:
    sys.exit(1)
" 2>/dev/null
}

# ─── Launch a worker ───
spawn_worker() {
    local wid="$1"

    if [ -n "$DRY_RUN" ]; then
        log "${YELLOW}DRY-RUN: would spawn $wid${RESET}"
        return 0
    fi

    local offer
    offer=$(find_cheapest_offer) || { log "${RED}WARN: no offers under \$$MAX_DPH/hr${RESET}"; return 1; }

    local offer_id gpu price
    offer_id=$(echo "$offer" | cut -f1)
    gpu=$(echo "$offer" | cut -f2)
    price=$(echo "$offer" | cut -f3)

    log "${GREEN}LAUNCH: $gpu @ \$$price/hr -> $wid (offer $offer_id)${RESET}"

    local ONSTART='#!/bin/bash
set -e
echo "=== Setup started at $(date) ==="
apt-get update -qq && apt-get install -y -qq git > /dev/null
echo "git installed"
git clone --depth 1 --branch '"$GIT_BRANCH"' '"$GIT_REPO"' /root/code
echo "repo cloned"
pip install -q huggingface_hub
echo "deps installed"
huggingface-cli login --token $HF_TOKEN --add-to-git-credential 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(), \"No CUDA\"; print(f\"GPU OK: {torch.cuda.get_device_name()}\")"
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
    --weights-name '"$WEIGHTS_NAME"' \
    --weight-sync-interval 2'

    vastai create instance "$offer_id" \
        --image "$IMAGE" --disk 15 \
        --label "${FLEET}-${wid}" \
        --env "-e HF_TOKEN=${HF_TOKEN} -e WORKER_ID=${wid}" \
        --onstart-cmd "$ONSTART" 2>&1 | head -1
}

# ─── Replace a dead/stuck worker ───
replace_worker() {
    local dead_id="$1"
    local dead_label="$2"

    if [ -n "$DRY_RUN" ]; then
        log "${YELLOW}DRY-RUN: would replace $dead_label ($dead_id)${RESET}"
        return 0
    fi

    log "${RED}REPLACE: destroying $dead_id ($dead_label)${RESET}"
    vastai destroy instance "$dead_id" 2>/dev/null
    local wid
    wid=$(echo "$dead_label" | grep -oE 'worker-vast-[0-9]+')
    spawn_worker "$wid"
}

# ─── Graceful shutdown ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shutdown() {
    echo ""
    log "${BOLD}Shutting down fleet...${RESET}"
    if [ -n "$DRY_RUN" ]; then
        log "${YELLOW}DRY-RUN: would run vast_down.sh $FLEET --force${RESET}"
    else
        "${SCRIPT_DIR}/vast_down.sh" "$FLEET" --force
    fi
    log "Monitor stopped."
    exit 0
}

trap shutdown SIGINT SIGTERM

# ─── Summary stats ───
TOTAL_GAMES_SEEN=0
TOTAL_CHECKS=0
TOTAL_REPLACEMENTS=0

print_summary() {
    local elapsed="$1"
    echo ""
    log "${BOLD}── Summary ($(fmt_duration "$elapsed")) ──${RESET}"
    log "  Checks: $TOTAL_CHECKS | Replacements: $TOTAL_REPLACEMENTS"
    echo ""
}

# ─── Startup ───
echo "${BOLD}vast_monitor.sh — Autonomous fleet manager${RESET}"
echo ""
echo "  Fleet:    $FLEET"
echo "  Workers:  $TARGET_WORKERS"
echo "  Weights:  $WEIGHTS_NAME"
echo "  Repo:     $REPO_ID"
echo "  Examples: $EXAMPLES_REPO_ID"
echo "  Max $/hr: $MAX_DPH"
echo "  GPUs:     $GPUS"
if [ -n "$DRY_RUN" ]; then
    echo "  Mode:     ${YELLOW}DRY-RUN (no instances created/destroyed)${RESET}"
fi
echo ""

# Bootstrap: verify weights exist
bootstrap_check || exit 1

if [ -n "$DRY_RUN" ]; then
    log "${YELLOW}DRY-RUN: config OK, would start monitoring ${TARGET_WORKERS} workers${RESET}"
    exit 0
fi

log "${GREEN}=== Monitor started. Fleet=$FLEET, target=${TARGET_WORKERS}w, max=\$$MAX_DPH/hr ===${RESET}"

# ─── Main loop ───
START_TIME=$(date +%s)

while true; do
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    elapsed=$(( $(date +%s) - START_TIME ))

    fleet_data=$(get_fleet)
    n_instances=$(echo "$fleet_data" | grep -c . 2>/dev/null || echo 0)

    if [ "$n_instances" -eq 0 ]; then
        log "CHECK #$TOTAL_CHECKS: ${YELLOW}NO INSTANCES — spawning fleet${RESET}"
    else
        total_dph=0
        perfline=""
        while IFS=$'\t' read -r iid label gpu status dph age; do
            total_dph=$(python3 -c "print(f'{$total_dph + $dph:.3f}')")

            # Cost guard
            if python3 -c "exit(0 if $dph > $MAX_DPH else 1)"; then
                log "$(color_tag ERR "COST: $label ($gpu) @ \$$dph > \$$MAX_DPH — replacing")"
                replace_worker "$iid" "$label"
                TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                perfline+=" ${label##*-}=$(color_tag ERR "REPLACED(\$$dph)")"
                continue
            fi

            wstate=$(check_worker "$iid")
            tag="${wstate%%:*}"
            detail="${wstate#*:}"

            case "$tag" in
                OK)
                    gps=$(echo "$detail" | grep -oE '[0-9]+\.[0-9]+ games/s' | head -1)
                    batch=$(echo "$detail" | grep -oE 'batch [0-9]+' | head -1)
                    perfline+=" ${label##*-}=$(color_tag OK "${gpu}/\$$dph/${gps}/${batch}/$(fmt_duration "$age")")"
                    ;;
                ERR)
                    log "$(color_tag ERR "ERROR: $label ($iid): $detail")"
                    if [ "$age" -gt 300 ]; then
                        replace_worker "$iid" "$label"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                        perfline+=" ${label##*-}=$(color_tag ERR "REPLACED(err)")"
                    else
                        perfline+=" ${label##*-}=$(color_tag ERR "ERR($(fmt_duration "$age"))")"
                    fi
                    ;;
                SETUP)
                    perfline+=" ${label##*-}=$(color_tag SETUP "SETUP/$(fmt_duration "$age")")"
                    if [ "$age" -gt 600 ]; then
                        log "$(color_tag ERR "STALE: $label stuck in setup $(fmt_duration "$age") — replacing")"
                        replace_worker "$iid" "$label"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                    fi
                    ;;
                BOOT)
                    perfline+=" ${label##*-}=$(color_tag BOOT "BOOT/$(fmt_duration "$age")")"
                    if [ "$age" -gt 600 ]; then
                        log "$(color_tag ERR "STALE: $label no container after $(fmt_duration "$age") — replacing")"
                        replace_worker "$iid" "$label"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                    fi
                    ;;
                WAIT)
                    perfline+=" ${label##*-}=$(color_tag WAIT "WAIT/$(fmt_duration "$age")")"
                    if [ "$age" -gt 900 ]; then
                        log "$(color_tag ERR "STALE: $label waiting $(fmt_duration "$age") with no progress — replacing")"
                        replace_worker "$iid" "$label"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                    fi
                    ;;
            esac
        done <<< "$fleet_data"

        log "CHECK #$TOTAL_CHECKS [$(fmt_duration "$elapsed")] \$$total_dph/hr |$perfline"
    fi

    # Replenish: spawn missing workers up to target
    live_count=$(get_fleet | grep -c . 2>/dev/null || echo 0)
    if [ "$live_count" -lt "$TARGET_WORKERS" ]; then
        live_labels=$(get_fleet | cut -f2)
        for idx in $(seq 0 $((TARGET_WORKERS - 1))); do
            wid="worker-vast-${idx}"
            if ! echo "$live_labels" | grep -q "$wid"; then
                log "${GREEN}REPLENISH: $wid missing ($live_count/$TARGET_WORKERS alive)${RESET}"
                spawn_worker "$wid"
                live_count=$((live_count + 1))
            fi
        done
    fi

    # Print summary every 10 checks
    if [ $((TOTAL_CHECKS % 10)) -eq 0 ]; then
        print_summary "$elapsed"
    fi

    # Adaptive polling: fast ramp-up, slow steady-state
    if [ "$elapsed" -lt "$PHASE1_DURATION" ]; then
        sleep "$PHASE1_INTERVAL"
    else
        sleep "$PHASE2_INTERVAL"
    fi
done
