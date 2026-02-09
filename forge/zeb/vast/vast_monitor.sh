#!/usr/bin/env bash
# vast_monitor.sh — Autonomous self-healing fleet manager
#
# Event-sourced architecture: observe → decide → act each cycle.
# Single-pass parallel polling replaces background checker processes.
# Append-only event log enables stall detection and poll resilience.
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
#   ZEB_GPUS                Space-separated GPU list (default: RTX_3060 ... RTX_4070_Ti)

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
UPLOAD_INTERVAL="${ZEB_UPLOAD_INTERVAL:-auto}"

BLACKLIST_FILE="${ZEB_BLACKLIST:-$HOME/.config/zeb/bad-hosts.txt}"
PREFERRED_FILE="${ZEB_PREFERRED:-$HOME/.config/zeb/preferred-hosts.txt}"

IMAGE="jasonyandell/zeb-worker:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PHASE1_INTERVAL=60
PHASE2_INTERVAL=600
PHASE1_DURATION=1800  # 30 min ramp-up
CREDIT_WAIT_INTERVAL=300  # 5 min between credit probes
BLACKLIST_STRIKES=3

# ─── Colors ───
if [ -t 1 ]; then
    GREEN=$'\033[32m'; YELLOW=$'\033[33m'; RED=$'\033[31m'
    BOLD=$'\033[1m'; RESET=$'\033[0m'
else
    GREEN="" YELLOW="" RED="" BOLD="" RESET=""
fi

# ─── Helpers ───

log() { echo "$(date '+%Y-%m-%d %H:%M:%S') $*"; }

color_tag() {
    local status="$1" text="$2"
    case "$status" in
        OK)         printf '%s' "${GREEN}${text}${RESET}" ;;
        SETUP|BOOT|WAIT) printf '%s' "${YELLOW}${text}${RESET}" ;;
        *)          printf '%s' "${RED}${text}${RESET}" ;;
    esac
}

fmt_duration() {
    local s=$1
    if [ "$s" -ge 3600 ]; then
        printf '%dh%02dm' $((s/3600)) $(( (s%3600)/60 ))
    elif [ "$s" -ge 60 ]; then
        printf '%dm%02ds' $((s/60)) $((s%60))
    else
        printf '%ds' "$s"
    fi
}

# ─── HF bootstrap check ───

bootstrap_check() {
    if [ -z "${HF_TOKEN:-}" ]; then
        if [ -f "$HOME/.cache/huggingface/token" ]; then
            HF_TOKEN=$(cat "$HOME/.cache/huggingface/token")
            export HF_TOKEN
        else
            echo "${RED}ERROR: HF_TOKEN not set and ~/.cache/huggingface/token not found.${RESET}"
            return 1
        fi
    fi
    log "Checking HF repo for ${WEIGHTS_NAME}.pt..."
    if python3 -c "
from huggingface_hub import hf_hub_url, get_hf_file_metadata
url = hf_hub_url('$REPO_ID', '${WEIGHTS_NAME}.pt')
get_hf_file_metadata(url)
" 2>/dev/null; then
        echo "OK"
    else
        echo "${RED}${BOLD}ERROR: ${WEIGHTS_NAME}.pt not found in ${REPO_ID}${RESET}"
        echo "Start the learner first to seed the HF repo with initial weights."
        return 1
    fi
}

# ─── Fleet observation ───

get_fleet() {
    local raw
    raw=$(timeout 15 vastai show instances --raw 2>/dev/null) || return
    echo "$raw" > /tmp/_vast_fleet.json
    python3 -c "
import json, time
data = json.load(open('/tmp/_vast_fleet.json'))
now = time.time()
for i in data:
    label = i.get('label') or ''
    if not label.startswith('${FLEET}-'):
        continue
    iid = str(i['id'])
    gpu = i.get('gpu_name', '?')
    status = i.get('actual_status', '?')
    dph = i.get('dph_total', 0)
    start = i.get('start_date', 0)
    age = int(now - start) if start else 0
    mid = str(i.get('machine_id', '?'))
    print(f'{iid}\t{label}\t{gpu}\t{status}\t{dph:.3f}\t{age}\t{mid}')
" 2>/dev/null
}

check_worker() {
    local iid="$1"
    local raw
    raw=$(timeout 5 vastai logs "$iid" --tail 50 2>/dev/null) || { echo "BOOT"; return; }

    # Fatal errors
    local fatal
    fatal=$(echo "$raw" | grep -iE '(Exception|Traceback|killed|OOM|size mismatch)' | tail -1 | cut -c1-80)
    if [ -n "$fatal" ]; then
        echo "ERR:$fatal"
        return
    fi

    # Broken launch detection (e.g. "ssh: command not found" in a loop)
    local err_count
    err_count=$(echo "$raw" | grep -cE '(command not found|No such file or directory|Permission denied|cannot execute)' 2>/dev/null || true)
    if [ "$err_count" -ge 10 ]; then
        local has_progress
        has_progress=$(echo "$raw" | grep -cE '(batch [0-9]+|Setup started|cloning|installed|GPU OK|games/s)' 2>/dev/null || true)
        if [ "$has_progress" -eq 0 ]; then
            local err_sample
            err_sample=$(echo "$raw" | grep -E '(command not found|No such file)' | tail -1 | cut -c1-60)
            echo "ERR:launch-broken($err_sample)"
            return
        fi
    fi

    # Batch progress
    local batch_line
    batch_line=$(echo "$raw" | grep -oE 'batch [0-9]+:.*games/s' | tail -1)
    if [ -n "$batch_line" ]; then
        echo "OK:$batch_line"
        return
    fi

    # Uploading to HF
    if echo "$raw" | grep -qE '(Upload complete|Uploading .* batches|New Data Upload)' 2>/dev/null; then
        echo "OK:uploading"
        return
    fi

    # Setup markers
    if echo "$raw" | grep -qE '(cloning|installed|Setup started|Pulling initial|repo cloned|deps installed|GPU OK)' 2>/dev/null; then
        echo "SETUP"
        return
    fi

    # Docker image still pulling
    if echo "$raw" | grep -qE 'No such container' 2>/dev/null; then
        echo "BOOT"
        return
    fi

    echo "WAIT"
}

# ─── Parallel polling ───

declare -A POLL_RESULTS

poll_all_workers() {
    local fleet_data="$1"
    POLL_RESULTS=()
    [ -z "$fleet_data" ] && return

    local poll_dir
    poll_dir=$(mktemp -d "$STATUS_DIR/poll.XXXXXX")

    while IFS=$'\t' read -r iid _rest; do
        (
            result=$(check_worker "$iid" 2>/dev/null) || result="BOOT"
            echo "$result" > "${poll_dir}/${iid}"
        ) &
    done <<< "$fleet_data"
    wait

    for f in "$poll_dir"/*; do
        [ -f "$f" ] || continue
        local iid
        iid=$(basename "$f")
        POLL_RESULTS["$iid"]=$(cat "$f")
    done
    rm -rf "$poll_dir"
}

# ─── Event log ───

EVENT_LOG=""  # set after STATUS_DIR is created

append_event() {
    local iid="$1" tag="$2" detail="${3:-}"
    printf '%s\t%s\t%s\t%s\n' "$(date +%s)" "$iid" "$tag" "$detail" >> "$EVENT_LOG"
}

# Last N events for an instance (most recent last)
last_events() {
    local iid="$1" n="${2:-5}"
    grep "	${iid}	" "$EVENT_LOG" 2>/dev/null | tail -n "$n"
}

rotate_log() {
    local lines
    lines=$(wc -l < "$EVENT_LOG" 2>/dev/null || echo 0)
    if [ "$lines" -gt 2000 ]; then
        local tmp="${EVENT_LOG}.tmp"
        tail -1000 "$EVENT_LOG" > "$tmp"
        mv "$tmp" "$EVENT_LOG"
    fi
}

# ─── Classify: apply stall detection + two-poll resilience ───

declare -A CLASSIFIED  # iid -> "TAG:DETAIL"

classify_all() {
    local fleet_data="$1"
    CLASSIFIED=()
    [ -z "$fleet_data" ] && return

    while IFS=$'\t' read -r iid _label _gpu _status _dph w_age _mid; do
        local raw="${POLL_RESULTS[$iid]:-BOOT}"
        local tag="${raw%%:*}"
        local detail="${raw#*:}"

        if [ "$tag" = "OK" ]; then
            # Stall detection: same batch in last 2 OK events + this one = 3 consecutive
            # Skip for young workers (<5min) — first batches take time
            local batch_num
            batch_num=$(echo "$detail" | grep -oE 'batch [0-9]+' | grep -oE '[0-9]+')
            if [ -n "$batch_num" ] && [ "${w_age:-0}" -gt 300 ]; then
                local prev_all prev_count prev_batches n_unique
                prev_all=$(grep "	${iid}	OK	batch " "$EVENT_LOG" 2>/dev/null | tail -2 | grep -oE 'batch [0-9]+' | grep -oE '[0-9]+')
                prev_count=$([ -n "$prev_all" ] && echo "$prev_all" | wc -l || echo 0)
                if [ "$prev_count" -ge 2 ]; then
                    prev_batches=$(echo "$prev_all" | sort -u)
                    n_unique=$(echo "$prev_batches" | wc -l)
                    if [ "$n_unique" -eq 1 ] && [ "$prev_batches" = "$batch_num" ]; then
                        CLASSIFIED["$iid"]="STALL:batch $batch_num stuck (3 consecutive)"
                        continue
                    fi
                fi
            fi
            CLASSIFIED["$iid"]="$raw"
        elif [ "$tag" = "BOOT" ] || [ "$tag" = "ERR" ]; then
            # Two-poll resilience: forgive one bad poll after OK
            local last_tag
            last_tag=$(grep "	${iid}	" "$EVENT_LOG" 2>/dev/null | tail -1 | cut -f3)
            if [ "$last_tag" = "OK" ]; then
                CLASSIFIED["$iid"]="POLL_TIMEOUT:$raw"
            else
                CLASSIFIED["$iid"]="$raw"
            fi
        else
            CLASSIFIED["$iid"]="$raw"
        fi
    done <<< "$fleet_data"
}

# ─── Offer search (delegates to find_offer.py) ───

find_cheapest_offer() {
    python3 "${SCRIPT_DIR}/find_offer.py" \
        --max-dph "$MAX_DPH" \
        --gpus "$(echo "$GPUS" | tr ' ' ',')" \
        --blacklist "$(load_blacklist)" \
        --preferred "$(preferred_for_python)" 2>/dev/null
}

find_worst_value() {
    local fleet_data="$1" exclude="${2:-}"
    echo "$fleet_data" | python3 "${SCRIPT_DIR}/find_offer.py" \
        --worst-value --exclude-iid "$exclude" 2>/dev/null
}

# ─── Blacklist helpers ───

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

# ─── Preferred hosts ───

declare -A PREFERRED_GPS  # machine_id -> observed games/s

load_preferred() {
    PREFERRED_GPS=()
    if [ -f "$PREFERRED_FILE" ]; then
        while IFS=$'\t' read -r mid gps; do
            [ -n "$mid" ] && PREFERRED_GPS["$mid"]="$gps"
        done < "$PREFERRED_FILE"
    fi
}

save_preferred() {
    [ "${#PREFERRED_GPS[@]}" -eq 0 ] && return
    mkdir -p "$(dirname "$PREFERRED_FILE")"
    : > "$PREFERRED_FILE"
    for mid in "${!PREFERRED_GPS[@]}"; do
        printf '%s\t%s\n' "$mid" "${PREFERRED_GPS[$mid]}"
    done >> "$PREFERRED_FILE"
}

preferred_for_python() {
    local out=""
    for mid in "${!PREFERRED_GPS[@]}"; do
        [ -n "$out" ] && out+=","
        out+="${mid}:${PREFERRED_GPS[$mid]}"
    done
    echo "$out"
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

    local offer_id gpu price pref pref_tag
    offer_id=$(echo "$offer" | cut -f1)
    gpu=$(echo "$offer" | cut -f2)
    price=$(echo "$offer" | cut -f3)
    pref=$(echo "$offer" | cut -f4)
    pref_tag=""
    [ "$pref" = "preferred" ] && pref_tag=" ${GREEN}★ preferred${RESET}"

    log "${GREEN}LAUNCH: $gpu @ \$$price/hr -> $wid (offer $offer_id)${pref_tag}${RESET}"

    local ONSTART='#!/bin/bash
set -e
echo "=== Setup started at $(date) ==="
cd /root/code
git pull --ff-only 2>/dev/null || true
echo "code updated"
huggingface-cli login --token $HF_TOKEN --add-to-git-credential 2>/dev/null || true
python -c "import torch; assert torch.cuda.is_available(), \"No CUDA\"; print(f\"GPU OK: {torch.cuda.get_device_name()}\")"
echo "=== Starting worker $WORKER_ID at $(date) ==="
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

    if echo "$result" | grep -qi 'lacks credit'; then
        return 2
    fi
}

# ─── Action execution ───

do_destroy() {
    local iid="$1" label="$2" mid="${3:-?}" reason="$4"
    if [ -n "$DRY_RUN" ]; then
        log "${YELLOW}DRY-RUN: would destroy $label ($iid)${RESET}"
        return 0
    fi
    blacklist_host "$mid" "$reason"
    vastai destroy instance "$iid" 2>/dev/null
}

do_replace() {
    local iid="$1" label="$2" mid="${3:-?}" reason="$4"
    do_destroy "$iid" "$label" "$mid" "$reason"

    if [ -n "$NO_CREDIT" ]; then
        log "${YELLOW}Skipping replacement spawn (no credit)${RESET}"
        return 1
    fi

    local wid
    wid=$(echo "$label" | grep -oE 'worker-vast-[0-9]+')
    log "${RED}REPLACE: $iid ($label, machine $mid) — spawning replacement${RESET}"
    spawn_worker "$wid"
    local rc=$?
    if [ "$rc" -eq 2 ]; then
        NO_CREDIT=1
        log "${RED}${BOLD}NO CREDIT — pausing fleet operations. Will probe every $(( CREDIT_WAIT_INTERVAL / 60 ))min.${RESET}"
    fi
    return "$rc"
}

# ─── Graceful shutdown ───

shutdown() {
    echo ""
    log "${BOLD}Monitor stopped.${RESET}"
    save_preferred
    [ "${#PREFERRED_GPS[@]}" -gt 0 ] && log "  Saved ${#PREFERRED_GPS[@]} preferred machines to $PREFERRED_FILE"
    # Kill any lingering poll subshells
    kill 0 2>/dev/null
    wait 2>/dev/null
    rm -f "$STATUS_DIR"/poll.* 2>/dev/null
    log "  Fleet is still running. Tear down: ${SCRIPT_DIR}/vast_down.sh $FLEET --force"
    exit 0
}

trap shutdown SIGINT SIGTERM

# ─── Global state ───

TOTAL_CHECKS=0
TOTAL_REPLACEMENTS=0
NO_CREDIT=""
UPGRADE_IID=""
UPGRADE_STARTED=0
UPGRADE_COOLDOWN=0

STATUS_DIR="/tmp/zeb-monitor-${FLEET}"
mkdir -p "$STATUS_DIR"
EVENT_LOG="${STATUS_DIR}/events.log"
# Fresh event log each startup — stale events from previous runs cause false stalls
: > "$EVENT_LOG"

# ─── Compute upload interval ───

if [ "$UPLOAD_INTERVAL" = "auto" ]; then
    UPLOAD_INTERVAL=$(python3 -c "import math; print(max(240, math.ceil($TARGET_WORKERS * 3600 / 100)))")
fi
COMMITS_PER_HR=$(python3 -c "print(f'{$TARGET_WORKERS * 3600 / $UPLOAD_INTERVAL:.0f}')")

# ─── Startup banner ───

echo "${BOLD}vast_monitor.sh — Autonomous fleet manager${RESET}"
echo ""
echo "  Fleet:    $FLEET"
echo "  Workers:  $TARGET_WORKERS"
echo "  Weights:  $WEIGHTS_NAME"
echo "  Repo:     $REPO_ID"
echo "  Examples: $EXAMPLES_REPO_ID"
echo "  Max \$/hr: $MAX_DPH"
echo "  GPUs:     $GPUS"
echo "  Upload:   every ${UPLOAD_INTERVAL}s (~${COMMITS_PER_HR} commits/hr, limit 128)"
if [ -f "$BLACKLIST_FILE" ]; then
    N_STRIKES=$(wc -l < "$BLACKLIST_FILE")
    _bl=$(load_blacklist)
    N_BLOCKED=$([ -n "$_bl" ] && echo "$_bl" | tr ',' '\n' | wc -l || echo 0)
    [ "$N_STRIKES" -gt 0 ] && echo "  Blacklist: $N_BLOCKED blocked, $N_STRIKES strikes total ($BLACKLIST_STRIKES to block)"
fi
load_preferred
[ "${#PREFERRED_GPS[@]}" -gt 0 ] && echo "  Preferred: ${#PREFERRED_GPS[@]} machines with observed performance"
[ -n "$DRY_RUN" ] && echo "  Mode:     ${YELLOW}DRY-RUN (no instances created/destroyed)${RESET}"
echo ""

# Bootstrap: verify weights exist
bootstrap_check || exit 1

if [ -n "$DRY_RUN" ]; then
    log "${YELLOW}DRY-RUN: config OK, would start monitoring ${TARGET_WORKERS} workers${RESET}"
    exit 0
fi

log "${GREEN}=== Monitor started. Fleet=$FLEET, target=${TARGET_WORKERS}w, max=\$$MAX_DPH/hr ===${RESET}"

# ═══════════════════════════════════════════════════════════
# ─── Main loop: observe → decide → act → report ───
# ═══════════════════════════════════════════════════════════

START_TIME=$(date +%s)

while true; do
    TOTAL_CHECKS=$((TOTAL_CHECKS + 1))
    elapsed=$(( $(date +%s) - START_TIME ))

    # ── OBSERVE ──
    fleet_data=$(get_fleet)

    if [ -z "$fleet_data" ]; then
        log "CHECK #$TOTAL_CHECKS: ${YELLOW}NO INSTANCES — spawning fleet${RESET}"
    else
        # Dedup: if two instances share a label, kill the older one
        dupes=$(echo "$fleet_data" | python3 -c "
import sys
from collections import defaultdict
by_label = defaultdict(list)
for line in sys.stdin:
    parts = line.strip().split('\t')
    if len(parts) >= 6:
        iid, label, age = parts[0], parts[1], int(parts[5])
        by_label[label].append((age, iid))
for label, instances in by_label.items():
    if len(instances) > 1:
        instances.sort(reverse=True)
        for age, iid in instances[1:]:
            print(f'{iid}\t{label}')
")
        if [ -n "$dupes" ]; then
            while IFS=$'\t' read -r dup_id dup_label; do
                log "${YELLOW}DEDUP: killing extra $dup_label ($dup_id)${RESET}"
                [ -z "$DRY_RUN" ] && vastai destroy instance "$dup_id" 2>/dev/null
            done <<< "$dupes"
            fleet_data=$(get_fleet)
        fi

        # Poll all workers in parallel
        poll_all_workers "$fleet_data"

        # Classify with stall detection + two-poll resilience
        classify_all "$fleet_data"

        # Append events to log
        while IFS=$'\t' read -r iid _rest; do
            local_tag="${CLASSIFIED[$iid]%%:*}"
            local_detail="${CLASSIFIED[$iid]#*:}"
            append_event "$iid" "$local_tag" "$local_detail"
        done <<< "$fleet_data"

        # ── DECIDE + ACT (health) ──
        total_dph=0
        total_gps=0
        n_ok=0
        n_up=$(echo "$fleet_data" | wc -l)
        over_target=$(( n_up > TARGET_WORKERS ? 1 : 0 ))
        perfline=""

        while IFS=$'\t' read -r iid label gpu status dph age mid; do
            total_dph=$(python3 -c "print(f'{$total_dph + $dph:.3f}')")

            # Skip upgrade trial worker
            if [ "$iid" = "$UPGRADE_IID" ]; then
                local_state="${CLASSIFIED[$iid]:-UNKNOWN}"
                perfline+=" upgrade=$(color_tag BOOT "${local_state%%:*}/$(fmt_duration "$age")")"
                continue
            fi

            # Cost guard
            if python3 -c "exit(0 if $dph > $MAX_DPH else 1)"; then
                log "$(color_tag ERR "COST: $label ($gpu) @ \$$dph > \$$MAX_DPH — removing")"
                do_destroy "$iid" "$label" "$mid" "over-cost"
                TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                n_up=$((n_up - 1)); over_target=$(( n_up > TARGET_WORKERS ? 1 : 0 ))
                perfline+=" ${label##*-}=$(color_tag ERR "REMOVED(\$$dph)")"
                continue
            fi

            wstate="${CLASSIFIED[$iid]:-UNKNOWN}"
            tag="${wstate%%:*}"
            detail="${wstate#*:}"

            case "$tag" in
                OK)
                    gps=$(echo "$detail" | grep -oE '[0-9]+\.[0-9]+ games/s' | head -1)
                    gps_num=$(echo "$gps" | grep -oE '[0-9]+\.[0-9]+')
                    total_gps=$(python3 -c "print(f'{$total_gps + ${gps_num:-0}:.1f}')")
                    n_ok=$((n_ok + 1))
                    batch=$(echo "$detail" | grep -oE 'batch [0-9]+' | head -1)
                    [ -n "${gps_num:-}" ] && [ "$mid" != "?" ] && PREFERRED_GPS["$mid"]="${gps_num}"
                    pref_star=""
                    [ -n "${PREFERRED_GPS[$mid]:-}" ] && pref_star="★"
                    perfline+=" ${label##*-}=$(color_tag OK "${gpu}/\$$dph/${gps}/${batch}/$(fmt_duration "$age")${pref_star}")"
                    ;;
                ERR)
                    log "$(color_tag ERR "ERROR: $label ($iid): $detail")"
                    if [ "$age" -gt 300 ]; then
                        do_replace "$iid" "$label" "$mid" "$label"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                        n_up=$((n_up - 1)); over_target=$(( n_up > TARGET_WORKERS ? 1 : 0 ))
                        perfline+=" ${label##*-}=$(color_tag ERR "REMOVED(err)")"
                    else
                        perfline+=" ${label##*-}=$(color_tag ERR "ERR($(fmt_duration "$age"))")"
                    fi
                    ;;
                SETUP)
                    perfline+=" ${label##*-}=$(color_tag SETUP "SETUP/$(fmt_duration "$age")")"
                    if [ "$age" -gt 600 ]; then
                        log "$(color_tag ERR "STALE: $label stuck in setup $(fmt_duration "$age") — removing")"
                        do_replace "$iid" "$label" "$mid" "$label"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                        n_up=$((n_up - 1)); over_target=$(( n_up > TARGET_WORKERS ? 1 : 0 ))
                    fi
                    ;;
                BOOT)
                    perfline+=" ${label##*-}=$(color_tag BOOT "BOOT/$(fmt_duration "$age")")"
                    if [ "$age" -gt 600 ]; then
                        log "$(color_tag ERR "STALE: $label no container after $(fmt_duration "$age") — removing")"
                        do_replace "$iid" "$label" "$mid" "$label"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                        n_up=$((n_up - 1)); over_target=$(( n_up > TARGET_WORKERS ? 1 : 0 ))
                    fi
                    ;;
                STALL)
                    log "$(color_tag ERR "STALL: $label ($gpu) — $detail — removing")"
                    do_replace "$iid" "$label" "$mid" "$label"
                    TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                    n_up=$((n_up - 1)); over_target=$(( n_up > TARGET_WORKERS ? 1 : 0 ))
                    perfline+=" ${label##*-}=$(color_tag ERR "STALL")"
                    ;;
                WAIT)
                    perfline+=" ${label##*-}=$(color_tag WAIT "WAIT/$(fmt_duration "$age")")"
                    if [ "$age" -gt 900 ]; then
                        log "$(color_tag ERR "STALE: $label waiting $(fmt_duration "$age") with no progress — removing")"
                        do_replace "$iid" "$label" "$mid" "$label"
                        TOTAL_REPLACEMENTS=$((TOTAL_REPLACEMENTS + 1))
                        n_up=$((n_up - 1)); over_target=$(( n_up > TARGET_WORKERS ? 1 : 0 ))
                    fi
                    ;;
                POLL_TIMEOUT)
                    # Forgiven single bad poll — show as OK with note
                    perfline+=" ${label##*-}=$(color_tag OK "${gpu}/\$$dph///$(fmt_duration "$age")${pref_star:-}")"
                    n_ok=$((n_ok + 1))
                    ;;
                *)
                    perfline+=" ${label##*-}=$(color_tag BOOT "POLL/$(fmt_duration "$age")")"
                    ;;
            esac
        done <<< "$fleet_data"

        log "CHECK #$TOTAL_CHECKS [$(fmt_duration "$elapsed")] ${n_ok}/${n_up}up ${total_gps} games/s \$$total_dph/hr |$perfline"
    fi

    # ── DECIDE + ACT (scaling) ──

    # Downscale: trim worst-value workers if over target
    if [ -z "$NO_CREDIT" ]; then
        live_data=$(get_fleet)
        live_count=$([ -n "$live_data" ] && echo "$live_data" | wc -l || echo 0)
        if [ "$live_count" -gt "$TARGET_WORKERS" ]; then
            excess=$((live_count - TARGET_WORKERS))
            log "${YELLOW}DOWNSCALE: $live_count alive > $TARGET_WORKERS target — trimming $excess${RESET}"
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
    iid, label, gpu, status, dph, age = parts[0], parts[1], parts[2], parts[3], float(parts[4]), int(parts[5])
    gps = expected_gps.get(gpu, 1.8)
    value = dph / gps
    tier = 0 if status != 'running' else (1 if age < 120 else 2)
    workers.append((tier, -value, iid, label, gpu, dph))
workers.sort()
for tier, neg_val, iid, label, gpu, dph in workers:
    print(f'{iid}\t{label}\t{gpu}\t{dph}')
")
            echo "$kill_order" | head -n "$excess" | while IFS=$'\t' read -r kill_id kill_label kill_gpu kill_dph; do
                log "${RED}TRIM: $kill_label ($kill_gpu @ \$$kill_dph/hr) — destroying${RESET}"
                [ -z "$DRY_RUN" ] && vastai destroy instance "$kill_id" 2>/dev/null
            done
        fi
    fi

    # Replenish: spawn missing workers up to target
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

    # ── Deal upgrade: trial a cheaper worker, swap out the worst ──
    if [ -z "$NO_CREDIT" ]; then
        if [ -n "$UPGRADE_IID" ]; then
            upgrade_state="${CLASSIFIED[$UPGRADE_IID]:-UNKNOWN}"
            upgrade_tag="${upgrade_state%%:*}"
            upgrade_age=$(( $(date +%s) - UPGRADE_STARTED ))
            if [ "$upgrade_tag" = "OK" ]; then
                # Trial producing — find and swap out worst current worker
                live_data=$(get_fleet)
                worst=$(find_worst_value "$live_data" "$UPGRADE_IID")
                if [ -n "$worst" ]; then
                    IFS=$'\t' read -r w_iid w_label w_gpu w_dph w_mid w_value <<< "$worst"
                    log "${GREEN}${BOLD}UPGRADE: trial $UPGRADE_IID producing — swapping out $w_label ($w_gpu @ \$$w_dph/hr, value=$w_value)${RESET}"
                    [ -z "$DRY_RUN" ] && vastai destroy instance "$w_iid" 2>/dev/null
                fi
                UPGRADE_IID=""
                UPGRADE_STARTED=0
            elif [ "$upgrade_age" -gt 900 ]; then
                # Trial took >15min — abort with cooldown
                log "${YELLOW}UPGRADE: trial $UPGRADE_IID failed to produce after $(fmt_duration "$upgrade_age") — aborting (30min cooldown)${RESET}"
                [ -z "$DRY_RUN" ] && vastai destroy instance "$UPGRADE_IID" 2>/dev/null
                trial_mid=$(echo "$fleet_data" | grep "$UPGRADE_IID" | cut -f7)
                [ -n "$trial_mid" ] && blacklist_host "$trial_mid" "upgrade-trial"
                UPGRADE_IID=""
                UPGRADE_STARTED=0
                UPGRADE_COOLDOWN=$(( $(date +%s) + 1800 ))
            fi
        elif [ "${n_ok:-0}" -eq "${n_up:-0}" ] && [ "${n_up:-0}" -ge "$TARGET_WORKERS" ] && [ "$(date +%s)" -gt "$UPGRADE_COOLDOWN" ]; then
            # Fleet full and healthy — look for a better deal
            live_data=$(get_fleet)
            worst_current=$(find_worst_value "$live_data")
            if [ -n "$worst_current" ]; then
                IFS=$'\t' read -r _ _ worst_gpu worst_dph _ worst_val <<< "$worst_current"
                best_offer=$(find_cheapest_offer 2>/dev/null) || best_offer=""
                if [ -n "$best_offer" ]; then
                    offer_id=$(echo "$best_offer" | cut -f1)
                    offer_gpu=$(echo "$best_offer" | cut -f2)
                    offer_dph=$(echo "$best_offer" | cut -f3)
                    offer_val=$(python3 -c "
gps_map = {'RTX 3060': 1.8, 'RTX 3060 Ti': 2.1, 'RTX 3070': 2.4, 'RTX 3070 Ti': 2.6,
    'RTX 3080': 2.6, 'RTX 3080 Ti': 2.6, 'RTX 3090': 2.6,
    'RTX 4060': 1.8, 'RTX 4060 Ti': 2.1, 'RTX 4070': 2.6, 'RTX 4070 Ti': 2.8}
print(f'{$offer_dph / gps_map.get(\"$offer_gpu\", 1.8):.4f}')
" 2>/dev/null)
                    is_better=$(python3 -c "print('yes' if $offer_val < $worst_val * 0.85 else 'no')" 2>/dev/null)
                    if [ "$is_better" = "yes" ]; then
                        log "${GREEN}${BOLD}UPGRADE: found $offer_gpu @ \$$offer_dph/hr (value=$offer_val) vs worst $worst_gpu @ \$$worst_dph/hr (value=$worst_val) — launching trial${RESET}"
                        wid="worker-vast-upgrade"
                        spawn_worker "$wid"
                        if [ $? -eq 0 ]; then
                            sleep 3
                            live_data=$(get_fleet)
                            UPGRADE_IID=$(echo "$live_data" | grep "worker-vast-upgrade" | head -1 | cut -f1)
                            [ -n "$UPGRADE_IID" ] && UPGRADE_STARTED=$(date +%s) && log "${GREEN}UPGRADE: trial worker $UPGRADE_IID launched${RESET}"
                        fi
                    fi
                fi
            fi
        fi
    fi

    # ── Summary + housekeeping ──
    if [ $((TOTAL_CHECKS % 10)) -eq 0 ]; then
        echo ""
        log "${BOLD}── Summary ($(fmt_duration "$elapsed")) ──${RESET}"
        log "  Checks: $TOTAL_CHECKS | Replacements: $TOTAL_REPLACEMENTS"
        echo ""
        save_preferred
    fi

    rotate_log

    # ── Credit-wait mode ──
    if [ -n "$NO_CREDIT" ]; then
        sleep "$CREDIT_WAIT_INTERVAL"
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
            NO_CREDIT=""
            log "${GREEN}${BOLD}Fleet is full — resuming normal monitoring${RESET}"
        fi
        continue
    fi

    # Adaptive polling
    if [ "$elapsed" -lt "$PHASE1_DURATION" ]; then
        sleep "$PHASE1_INTERVAL"
    else
        sleep "$PHASE2_INTERVAL"
    fi
done
