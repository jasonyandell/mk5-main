#!/usr/bin/env bash
# go-belief.sh — Launch the belief-head model fleet (high-throughput)
# Usage: ./go-belief.sh [N_WORKERS] [--dry-run]
#
# 5 workers × 3070 Ti+ GPUs ≈ $0.45/hr ≈ 15 g/s target
# $25 budget → ~55 hours of runtime
export ZEB_WEIGHTS_NAME="${ZEB_WEIGHTS_NAME:-large-belief}"
export ZEB_FLEET="${ZEB_FLEET:-zeb-belief}"
export ZEB_MAX_DPH="${ZEB_MAX_DPH:-0.10}"
export ZEB_GPUS="${ZEB_GPUS:-RTX_3070_Ti RTX_3080 RTX_3080_Ti RTX_3090 RTX_4070 RTX_4070_Ti}"
export ZEB_WORKER_MODE="${ZEB_WORKER_MODE:-selfplay}"
exec "$(dirname "$0")/vast_monitor.sh" 5 "$@"
