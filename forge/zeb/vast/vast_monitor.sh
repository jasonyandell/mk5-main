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
GPUS="${ZEB_GPUS:-RTX_3060 RTX_3060_Ti RTX_3070 RTX_3070_Ti RTX_3080 RTX_3080_Ti RTX_3090 RTX_4060 RTX_4060_Ti RTX_4070 RTX_4070_Ti}"
# HF free tier: 128 commits/hr. Formula: N_WORKERS * (3600 / interval) < 128
# Default auto-calculates from TARGET_WORKERS to stay under 100 commits/hr
UPLOAD_INTERVAL="${ZEB_UPLOAD_INTERVAL:-auto}"

# Bad host blacklist — machine_ids that consistently fail/stall
BLACKLIST_FILE="${ZEB_BLACKLIST:-$HOME/.config/zeb/bad-hosts.txt}"

IMAGE="pytorch/pytorch:2.6.0-cuda12.6-cudnn9-runtime"
GIT_REPO="https://github.com/jasonyandell/mk5-main.git"
GIT_BRANCH="forge"

PHASE1_INTERVAL=60
PHASE2_INTERVAL=600
PHASE1_DURATION=1800  # 30 min
CREDIT_WAIT_INTERVAL=300  # 5 min between credit probes

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
# Uses a temp file instead of pipe to avoid vastai broken-pipe bug
get_fleet() {
    timeout 15 vastai show instances --raw > /tmp/_vast_fleet.json 2>/dev/null
    python3 -c "
import json, time, sys
now = time.time()
fleet = '$FLEET'
try:
    data = json.load(open('/tmp/_vast_fleet.json'))
except (json.JSONDecodeError, FileNotFoundError):
    sys.exit(0)
for i in data:
    label = i.get('label') or ''
    if not label.startswith(fleet + '-worker-vast-'): continue
    iid = i.get('id', '?')
    gpu = i.get('gpu_name', '?')
    st = i.get('actual_status', '?')
    dph = i.get('dph_total', 0)
    age = int(now - i['start_date']) if i.get('start_date') else 0
    mid = i.get('machine_id', '?')
    print(f'{iid}\t{label}\t{gpu}\t{st}\t{dph:.3f}\t{age}\t{mid}')
"
}

# ─── Worker health check ───
check_worker() {
    local iid="$1"
    local raw
    raw=$(timeout 5 vastai logs "$iid" --tail 50 2>/dev/null) || { echo "BOOT"; return; }

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

    # Uploading to HF? Worker is alive, just busy uploading
    if echo "$raw" | grep -qE '(Upload complete|Uploading .* batches|New Data Upload)' 2>/dev/null; then
        echo "OK:uploading"
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

# ─── Find best-value offer across GPU types ───
# Picks by lowest $/game (dph / expected_gps) rather than lowest $/hr.
# Expected games/s from 13.9hrs of observations (Feb 2026):
#   RTX 3070 Ti: 2.6, RTX 3070: 2.4, RTX 3060 Ti: 2.1, RTX 3060: 1.8
# GPUs without data default to 1.8 (conservative).
find_cheapest_offer() {
    local blacklist
    blacklist=$(load_blacklist)
    python3 -c "
import subprocess, json, sys

gpus = '$GPUS'.split()
max_dph = $MAX_DPH
blacklist = set('$blacklist'.split(',')) - {''}

# Expected games/s per GPU — used to rank by value, not raw price
expected_gps = {
    'RTX 3060': 1.8, 'RTX 3060 Ti': 2.1,
    'RTX 3070': 2.4, 'RTX 3070 Ti': 2.6,
    'RTX 3080': 2.6, 'RTX 3080 Ti': 2.6, 'RTX 3090': 2.6,
    'RTX 4060': 1.8, 'RTX 4060 Ti': 2.1,
    'RTX 4070': 2.6, 'RTX 4070 Ti': 2.8,
}

best = None
best_value = float('inf')  # lower = better ($/game/s)

for gpu in gpus:
    try:
        r = subprocess.run(
            ['vastai', 'search', 'offers', f'gpu_name={gpu} num_gpus=1',
             '-o', 'dph_total', '--raw', '--limit', '5'],
            capture_output=True, text=True, timeout=15
        )
        if r.returncode != 0: continue
        offers = json.loads(r.stdout)
        for o in offers:
            mid = str(o.get('machine_id', ''))
            if mid in blacklist: continue
            dph = o['dph_total']
            if dph > max_dph: break
            gps = expected_gps.get(o['gpu_name'], 1.8)
            value = dph / gps  # $/hr per game/s — lower is better
            if value < best_value:
                best = o
                best_value = value
            break  # only check cheapest offer per GPU type
    except Exception:
        pass

if best:
    gps = expected_gps.get(best['gpu_name'], 1.8)
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
    --weight-sync-interval 2 \
    --upload-interval '"$UPLOAD_INTERVAL"''

    local result
    result=$(vastai create instance "$offer_id" \
        --image "$IMAGE" --disk 15 \
        --label "${FLEET}-${wid}" \
        --env "-e HF_TOKEN=${HF_TOKEN} -e WORKER_ID=${wid}" \
        --onstart-cmd "$ONSTART" 2>&1 | head -1)
    echo "$result"

    # Detect credit exhaustion
    if echo "$result" | grep -qi 'lacks credit'; then
        return 2
    fi
}

# ─── Blacklist helpers ───
# File stores one machine_id per strike. 3 strikes = blocked from offers.
BLACKLIST_STRIKES=3

blacklist_host() {
    local machine_id="$1" reason="$2"
    [ "$machine_id" = "?" ] && return
    mkdir -p "$(dirname "$BLACKLIST_FILE")"
    echo "$machine_id" >> "$BLACKLIST_FILE"
    local count
    count=$(grep -cx "$machine_id" "$BLACKLIST_FILE" 2>/dev/null || echo 0)
    if [ "$count" -ge "$BLACKLIST_STRIKES" ]; then
        log "${RED}BLACKLIST: machine $machine_id — strike $count/$BLACKLIST_STRIKES, now blocked ($reason)${RESET}"
    else
        log "${YELLOW}STRIKE: machine $machine_id — $count/$BLACKLIST_STRIKES ($reason)${RESET}"
    fi
}

# Returns comma-separated machine_ids with >= BLACKLIST_STRIKES strikes
load_blacklist() {
    if [ -f "$BLACKLIST_FILE" ]; then
        python3 -c "
from collections import Counter
ids = open('$BLACKLIST_FILE').read().split()
blocked = [mid for mid, n in Counter(ids).items() if n >= $BLACKLIST_STRIKES]
print(','.join(blocked))
" 2>/dev/null
    fi
}

# ─── Replace a dead/stuck worker ───
replace_worker() {
    local dead_id="$1"
    local dead_label="$2"
    local machine_id="${3:-?}"

    if [ -n "$DRY_RUN" ]; then
        log "${YELLOW}DRY-RUN: would replace $dead_label ($dead_id, machine $machine_id)${RESET}"
        return 0
    fi

    log "${RED}REPLACE: destroying $dead_id ($dead_label, machine $machine_id)${RESET}"
    blacklist_host "$machine_id" "$dead_label"
    vastai destroy instance "$dead_id" 2>/dev/null

    # Skip spawning replacement if we know credit is exhausted
    if [ -n "$NO_CREDIT" ]; then
        log "${YELLOW}Skipping replacement spawn (no credit)${RESET}"
        return 1
    fi

    local wid
    wid=$(echo "$dead_label" | grep -oE 'worker-vast-[0-9]+')
    spawn_worker "$wid"
    local rc=$?
    if [ "$rc" -eq 2 ]; then
        NO_CREDIT=1
        log "${RED}${BOLD}NO CREDIT — pausing fleet operations. Will probe every $(( CREDIT_WAIT_INTERVAL / 60 ))min.${RESET}"
    fi
    return "$rc"
}

# ─── Graceful shutdown ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

shutdown() {
    echo ""
    log "${BOLD}Monitor stopped.${RESET} Cleaning up checkers..."
    for pid in "${CHECKER_PIDS[@]}"; do
        kill "$pid" 2>/dev/null
    done
    wait 2>/dev/null
    rm -rf "$STATUS_DIR"
    log "  Fleet is still running. Tear down: ${SCRIPT_DIR}/vast_down.sh $FLEET --force"
    exit 0
}

trap shutdown SIGINT SIGTERM

# ─── Summary stats ───
TOTAL_GAMES_SEEN=0
TOTAL_CHECKS=0
TOTAL_REPLACEMENTS=0
NO_CREDIT=""  # set to 1 when Vast.ai reports credit exhaustion

# ─── CQRS: per-worker background checkers write status files ───
STATUS_DIR="/tmp/zeb-monitor-${FLEET}"
mkdir -p "$STATUS_DIR"
declare -A CHECKER_PIDS  # iid -> PID of background checker

# Background loop: polls one worker's logs every 45s, writes to status file
# File format: EPOCH TAG:DETAIL
worker_checker() {
    local iid="$1"
    local sfile="$STATUS_DIR/$iid"
    # Initialize status file
    echo "$(date +%s) UNKNOWN" > "$sfile" 2>/dev/null
    while true; do
        result=$(check_worker "$iid" 2>/dev/null) || result="BOOT"
        echo "$(date +%s) $result" > "$sfile" 2>/dev/null
        sleep 45
    done
}

# Start a checker for an instance (idempotent — kills existing first)
start_checker() {
    local iid="$1"
    stop_checker "$iid"
    worker_checker "$iid" &
    CHECKER_PIDS["$iid"]=$!
}

# Stop a checker for an instance
stop_checker() {
    local iid="$1"
    local pid="${CHECKER_PIDS[$iid]:-}"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null
        wait "$pid" 2>/dev/null
    fi
    unset "CHECKER_PIDS[$iid]"
    rm -f "$STATUS_DIR/$iid"
}

# Read cached status for an instance (returns "TAG:DETAIL" or "UNKNOWN" if no data yet)
read_status() {
    local iid="$1"
    local sfile="$STATUS_DIR/$iid"
    if [ -f "$sfile" ]; then
        local epoch tag_detail
        read -r epoch tag_detail < "$sfile"
        local age_of_check=$(( $(date +%s) - epoch ))
        # If status file is older than 2 minutes, checker may have died
        if [ "$age_of_check" -gt 120 ]; then
            echo "STALE_CHECK"
        else
            echo "$tag_detail"
        fi
    else
        echo "UNKNOWN"
    fi
}

print_summary() {
    local elapsed="$1"
    echo ""
    log "${BOLD}── Summary ($(fmt_duration "$elapsed")) ──${RESET}"
    log "  Checks: $TOTAL_CHECKS | Replacements: $TOTAL_REPLACEMENTS"
    echo ""
}

# ─── Compute upload interval ───
# HF free tier: 128 commits/hr. Target <100 commits/hr for safety.
if [ "$UPLOAD_INTERVAL" = "auto" ]; then
    # ceil(N_WORKERS * 3600 / 100) — ensures N * (3600/interval) < 100
    UPLOAD_INTERVAL=$(python3 -c "import math; print(max(240, math.ceil($TARGET_WORKERS * 3600 / 100)))")
fi
COMMITS_PER_HR=$(python3 -c "print(f'{$TARGET_WORKERS * 3600 / $UPLOAD_INTERVAL:.0f}')")

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
echo "  Upload:   every ${UPLOAD_INTERVAL}s (~${COMMITS_PER_HR} commits/hr, limit 128)"
if [ -f "$BLACKLIST_FILE" ]; then
    N_STRIKES=$(wc -l < "$BLACKLIST_FILE")
    _bl=$(load_blacklist)
    N_BLOCKED=$([ -n "$_bl" ] && echo "$_bl" | tr ',' '\n' | wc -l || echo 0)
    if [ "$N_STRIKES" -gt 0 ]; then
        echo "  Blacklist: $N_BLOCKED blocked, $N_STRIKES strikes total ($BLACKLIST_STRIKES to block)"
    fi
fi
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

    # Dedup: if two instances share a worker label, kill the older one
    if [ -n "$fleet_data" ]; then
        dupes=$(echo "$fleet_data" | python3 -c "
import sys
from collections import defaultdict
by_label = defaultdict(list)
for line in sys.stdin:
    parts = line.strip().split('\t')
    if len(parts) >= 6:
        iid, label, gpu, st, dph, age = parts[0], parts[1], parts[2], parts[3], parts[4], int(parts[5])
        by_label[label].append((age, iid))
for label, instances in by_label.items():
    if len(instances) > 1:
        instances.sort(reverse=True)  # oldest first
        for age, iid in instances[1:]:  # keep newest, kill rest
            print(f'{iid}\t{label}')
")
        if [ -n "$dupes" ]; then
            while IFS=$'\t' read -r dup_id dup_label; do
                log "${YELLOW}DEDUP: killing extra $dup_label ($dup_id)${RESET}"
                [ -z "$DRY_RUN" ] && vastai destroy instance "$dup_id" 2>/dev/null
            done <<< "$dupes"
            fleet_data=$(get_fleet)
        fi
    fi

    if [ -z "$fleet_data" ]; then
        log "CHECK #$TOTAL_CHECKS: ${YELLOW}NO INSTANCES — spawning fleet${RESET}"
    else
        # Ensure checkers are running for all live instances
        new_checkers=0
        live_iids=()
        while IFS=$'\t' read -r iid _ _ _ _ _ _; do
            live_iids+=("$iid")
            if [ -z "${CHECKER_PIDS[$iid]:-}" ] || ! kill -0 "${CHECKER_PIDS[$iid]}" 2>/dev/null; then
                start_checker "$iid"
                new_checkers=$((new_checkers + 1))
            fi
        done <<< "$fleet_data"
        # Stop checkers for instances that no longer exist
        for old_iid in "${!CHECKER_PIDS[@]}"; do
            found=0
            for live_iid in "${live_iids[@]}"; do
                [ "$old_iid" = "$live_iid" ] && found=1 && break
            done
            [ "$found" -eq 0 ] && stop_checker "$old_iid"
        done
        # On first check or when many new checkers started, wait for them to get data
        if [ "$new_checkers" -gt 3 ]; then
            log "${YELLOW}Waiting 10s for $new_checkers checkers to poll...${RESET}"
            sleep 10
        fi

        total_dph=0
        total_gps=0
        n_ok=0
        n_up=$(echo "$fleet_data" | wc -l)
        perfline=""
        while IFS=$'\t' read -r iid label gpu status dph age mid; do
            total_dph=$(python3 -c "print(f'{$total_dph + $dph:.3f}')")

            # Cost guard
            if python3 -c "exit(0 if $dph > $MAX_DPH else 1)"; then
                log "$(color_tag ERR "COST: $label ($gpu) @ \$$dph > \$$MAX_DPH — replacing")"
                stop_checker "$iid"
                replace_worker "$iid" "$label" "$mid"
                TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                perfline+=" ${label##*-}=$(color_tag ERR "REPLACED(\$$dph)")"
                continue
            fi

            wstate=$(read_status "$iid")
            tag="${wstate%%:*}"
            detail="${wstate#*:}"

            case "$tag" in
                OK)
                    gps=$(echo "$detail" | grep -oE '[0-9]+\.[0-9]+ games/s' | head -1)
                    gps_num=$(echo "$gps" | grep -oE '[0-9]+\.[0-9]+')
                    total_gps=$(python3 -c "print(f'{$total_gps + ${gps_num:-0}:.1f}')")
                    n_ok=$((n_ok + 1))
                    batch=$(echo "$detail" | grep -oE 'batch [0-9]+' | head -1)
                    perfline+=" ${label##*-}=$(color_tag OK "${gpu}/\$$dph/${gps}/${batch}/$(fmt_duration "$age")")"
                    ;;
                ERR)
                    log "$(color_tag ERR "ERROR: $label ($iid): $detail")"
                    if [ "$age" -gt 300 ]; then
                        stop_checker "$iid"
                        replace_worker "$iid" "$label" "$mid"
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
                        stop_checker "$iid"
                        replace_worker "$iid" "$label" "$mid"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                    fi
                    ;;
                BOOT)
                    perfline+=" ${label##*-}=$(color_tag BOOT "BOOT/$(fmt_duration "$age")")"
                    if [ "$age" -gt 600 ]; then
                        log "$(color_tag ERR "STALE: $label no container after $(fmt_duration "$age") — replacing")"
                        stop_checker "$iid"
                        replace_worker "$iid" "$label" "$mid"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                    fi
                    ;;
                WAIT)
                    perfline+=" ${label##*-}=$(color_tag WAIT "WAIT/$(fmt_duration "$age")")"
                    if [ "$age" -gt 900 ]; then
                        log "$(color_tag ERR "STALE: $label waiting $(fmt_duration "$age") with no progress — replacing")"
                        stop_checker "$iid"
                        replace_worker "$iid" "$label" "$mid"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                    fi
                    ;;
                UNKNOWN)
                    # Checker just started, no data yet — show as booting
                    perfline+=" ${label##*-}=$(color_tag BOOT "POLL/$(fmt_duration "$age")")"
                    ;;
                STALE_CHECK)
                    # Checker died or hung — restart it
                    log "${YELLOW}CHECKER STALE for $label — restarting${RESET}"
                    start_checker "$iid"
                    perfline+=" ${label##*-}=$(color_tag BOOT "REPOLL/$(fmt_duration "$age")")"
                    ;;
            esac
        done <<< "$fleet_data"

        log "CHECK #$TOTAL_CHECKS [$(fmt_duration "$elapsed")] ${n_ok}/${n_up}up ${total_gps} games/s \$$total_dph/hr |$perfline"
    fi

    # Downscale: if more workers than target, trim worst-value first
    if [ -z "$NO_CREDIT" ]; then
        live_data=$(get_fleet)
        live_count=$([ -n "$live_data" ] && echo "$live_data" | wc -l || echo 0)
        if [ "$live_count" -gt "$TARGET_WORKERS" ]; then
            excess=$((live_count - TARGET_WORKERS))
            log "${YELLOW}DOWNSCALE: $live_count alive > $TARGET_WORKERS target — trimming $excess${RESET}"
            # Rank workers by kill priority: non-running > non-producing > worst $/game/s
            kill_order=$(echo "$live_data" | python3 -c "
import sys
expected_gps = {
    'RTX 3060': 1.8, 'RTX 3060 Ti': 2.1,
    'RTX 3070': 2.4, 'RTX 3070 Ti': 2.6,
    'RTX 3080': 2.6, 'RTX 3080 Ti': 2.6, 'RTX 3090': 2.6,
    'RTX 4060': 1.8, 'RTX 4060 Ti': 2.1,
    'RTX 4070': 2.6, 'RTX 4070 Ti': 2.8,
}
workers = []
for line in sys.stdin:
    parts = line.strip().split('\t')
    if len(parts) < 7: continue
    iid, label, gpu, status, dph, age, mid = parts[0], parts[1], parts[2], parts[3], float(parts[4]), int(parts[5]), parts[6]
    gps = expected_gps.get(gpu, 1.8)
    value = dph / gps  # lower = better
    # Kill priority: 0=non-running, 1=young (<120s), 2=producing — then by worst value
    if status not in ('running',):
        tier = 0
    elif age < 120:
        tier = 1
    else:
        tier = 2
    workers.append((tier, -value, iid, label, gpu, dph))
# Sort: lowest tier first (kill first), then most negative value first (worst value)
workers.sort()
for tier, neg_val, iid, label, gpu, dph in workers:
    print(f'{iid}\t{label}\t{gpu}\t{dph}')
")
            echo "$kill_order" | head -n "$excess" | while IFS=$'\t' read -r kill_id kill_label kill_gpu kill_dph; do
                log "${RED}TRIM: $kill_label ($kill_gpu @ \$$kill_dph/hr) — destroying${RESET}"
                vastai destroy instance "$kill_id" 2>/dev/null
            done
        fi
    fi

    # Replenish: spawn missing workers up to target (parallel, batches of 4)
    if [ -z "$NO_CREDIT" ]; then
        live_data=$(get_fleet)
        live_count=$([ -n "$live_data" ] && echo "$live_data" | wc -l || echo 0)
        if [ "$live_count" -lt "$TARGET_WORKERS" ]; then
            live_labels=$(echo "$live_data" | cut -f2)
            missing=()
            for idx in $(seq 0 $((TARGET_WORKERS - 1))); do
                wid="worker-vast-${idx}"
                if ! echo "$live_labels" | grep -q "$wid"; then
                    missing+=("$wid")
                fi
            done

            BATCH_SIZE=4
            for ((i=0; i<${#missing[@]}; i+=BATCH_SIZE)); do
                batch=("${missing[@]:i:BATCH_SIZE}")
                pids=()
                for wid in "${batch[@]}"; do
                    log "${GREEN}REPLENISH: $wid missing ($live_count/$TARGET_WORKERS alive)${RESET}"
                    spawn_worker "$wid" &
                    pids+=($!)
                    live_count=$((live_count + 1))
                done
                # Wait for batch and check for credit issues
                for pid in "${pids[@]}"; do
                    wait "$pid"
                    rc=$?
                    if [ "$rc" -eq 2 ]; then
                        NO_CREDIT=1
                        log "${RED}${BOLD}NO CREDIT — pausing fleet operations. Will probe every $(( CREDIT_WAIT_INTERVAL / 60 ))min.${RESET}"
                        break 2
                    fi
                done
            done
        fi
    fi

    # Print summary every 10 checks
    if [ $((TOTAL_CHECKS % 10)) -eq 0 ]; then
        print_summary "$elapsed"
    fi

    # Credit-wait mode: probe with one launch, resume if it works
    if [ -n "$NO_CREDIT" ]; then
        sleep "$CREDIT_WAIT_INTERVAL"
        # Find first missing worker to use as probe
        live_data=$(get_fleet)
        live_labels=$([ -n "$live_data" ] && echo "$live_data" | cut -f2 || echo "")
        probe_wid=""
        for idx in $(seq 0 $((TARGET_WORKERS - 1))); do
            wid="worker-vast-${idx}"
            if ! echo "$live_labels" | grep -q "$wid"; then
                probe_wid="$wid"
                break
            fi
        done
        if [ -n "$probe_wid" ]; then
            log "${YELLOW}CREDIT PROBE: trying $probe_wid...${RESET}"
            spawn_worker "$probe_wid"
            if [ $? -ne 2 ]; then
                NO_CREDIT=""
                log "${GREEN}${BOLD}CREDIT RESTORED — resuming fleet operations${RESET}"
            else
                log "${YELLOW}Still no credit. Next probe in $(( CREDIT_WAIT_INTERVAL / 60 ))min.${RESET}"
            fi
        else
            # All workers alive — credit issue resolved itself
            NO_CREDIT=""
            log "${GREEN}${BOLD}Fleet is full — resuming normal monitoring${RESET}"
        fi
        continue  # skip normal sleep, we already waited
    fi

    # Adaptive polling: fast ramp-up, slow steady-state
    if [ "$elapsed" -lt "$PHASE1_DURATION" ]; then
        sleep "$PHASE1_INTERVAL"
    else
        sleep "$PHASE2_INTERVAL"
    fi
done
