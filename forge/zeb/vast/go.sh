#!/usr/bin/env bash
# go.sh — Launch the large model fleet
# Usage: ./go.sh [N_WORKERS] [--dry-run]
export ZEB_WEIGHTS_NAME="${ZEB_WEIGHTS_NAME:-large}"
export ZEB_FLEET="${ZEB_FLEET:-zeb-large}"
export ZEB_WORKER_MODE="${ZEB_WORKER_MODE:-selfplay}"
exec "$(dirname "$0")/vast_monitor.sh" 8 "$@"
