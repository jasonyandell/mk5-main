#!/usr/bin/env bash
# go-experiment.sh — Launch eval-aux fleet with experiment-level overrides.
# Usage: ./go-experiment.sh [N_WORKERS] [--dry-run]

# Keep the same market/monitor defaults as go-belief-evalaux, but allow
# experiment-specific namespace/fleet values via env overrides.
export ZEB_WEIGHTS_NAME="${ZEB_WEIGHTS_NAME:-zeb-large-belief-eq}"
export ZEB_FLEET="${ZEB_FLEET:-zeb-belief-evalaux}"
export ZEB_WORKER_MODE=eval_aux
export ZEB_MAX_DPH="${ZEB_MAX_DPH:-0.10}"
export ZEB_GPUS="${ZEB_GPUS:-RTX_3070_Ti RTX_3080 RTX_3080_Ti RTX_3090 RTX_4070 RTX_4070_Ti}"

# Eval-aux defaults: keep policy supervision off unless explicitly enabled.
export ZEB_ORACLE_CHECKPOINT="${ZEB_ORACLE_CHECKPOINT:-forge/models/domino-qval-large-3.3M-qgap0.071-qmae0.94.ckpt}"
export ZEB_EQ_N_SAMPLES="${ZEB_EQ_N_SAMPLES:-100}"
export ZEB_EQ_GAMES_PER_BATCH="${ZEB_EQ_GAMES_PER_BATCH:-128}"
export ZEB_EQ_BATCH_SIZE="${ZEB_EQ_BATCH_SIZE:-0}"
export ZEB_EQ_TEMPERATURE="${ZEB_EQ_TEMPERATURE:-0.1}"
export ZEB_EQ_POLICY_TARGETS="${ZEB_EQ_POLICY_TARGETS:-false}"

exec "$(dirname "$0")/vast_monitor.sh" 1 "$@"
