#!/usr/bin/env bash
set -euo pipefail

# go-full-teacher.sh - start learner for the "full teacher" experiment.
#
# Full E[Q] signal: policy=1.0, value=1.0, belief=1.0 at 25% mix.
# Bootstrap from large-belief step 1920.
#
# Example:
#   ./forge/zeb/learner/go-full-teacher.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python}"
fi

REPO_ID="${ZEB_REPO_ID:-jasonyandell/zeb-42}"
EXAMPLES_REPO_ID="${ZEB_EXAMPLES_REPO_ID:-jasonyandell/zeb-42-examples}"
WEIGHTS_NAME="${ZEB_WEIGHTS_NAME:-lb-v-eq-1920}"
RUN_NAME="${ZEB_RUN_NAME:-$WEIGHTS_NAME}"
WANDB_PROJECT="${ZEB_WANDB_PROJECT:-zeb-mcts}"
CHECKPOINT="${ZEB_CHECKPOINT:-forge/zeb/checkpoints/lb-v-eq-1920-bootstrap.pt}"
KEEP_EXAMPLE_FILES="${ZEB_KEEP_EXAMPLE_FILES:-128}"
LOCAL_REPLAY_CACHE_DIR="${ZEB_LOCAL_REPLAY_CACHE_DIR:-$HOME/.cache/forge/zeb/replay}"
LOCAL_REPLAY_CACHE_SAVE_EVERY="${ZEB_LOCAL_REPLAY_CACHE_SAVE_EVERY:-25}"
PUSH_EVERY="${ZEB_PUSH_EVERY:-10}"
AMP_ENABLED="${ZEB_AMP_ENABLED:-1}"
AMP_DTYPE="${ZEB_AMP_DTYPE:-fp16}"

LOG_DIR="${ZEB_LOG_DIR:-history/t42-4xvg/logs}"
LOG_FILE="${ZEB_LEARNER_LOG_FILE:-$LOG_DIR/learner.log}"
PID_FILE="${ZEB_LEARNER_PID_FILE:-history/t42-4xvg/learner.pid}"

mkdir -p "$LOG_DIR"
mkdir -p "$(dirname "$PID_FILE")"

if [[ -f "$PID_FILE" ]]; then
  old_pid="$(cat "$PID_FILE" 2>/dev/null || true)"
  if [[ -n "${old_pid}" ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "Learner appears to be running already (pid=$old_pid, pid_file=$PID_FILE)."
    echo "If that is stale, stop it first and remove/update the pid file."
    exit 1
  fi
fi

# Guard against stale pid files by also checking live process table.
running_pid="$(
  ps -eo pid,comm,args \
  | awk -v wn="$WEIGHTS_NAME" '
      $2 == "python" && index($0, "-u -m forge.zeb.learner.run") && index($0, "--weights-name " wn) { print $1; exit }
    '
)"
if [[ -n "$running_pid" ]]; then
  echo "Learner process already running for weights-name=$WEIGHTS_NAME (pid=$running_pid)."
  echo "Refusing to launch a duplicate."
  exit 1
fi

: > "$LOG_FILE"

amp_args=(--no-amp)
if [[ "$AMP_ENABLED" == "1" || "$AMP_ENABLED" == "true" || "$AMP_ENABLED" == "yes" ]]; then
  amp_args=(--amp --amp-dtype "$AMP_DTYPE")
fi

nohup "$PYTHON_BIN" -u -m forge.zeb.learner.run \
  --repo-id "$REPO_ID" \
  --examples-repo-id "$EXAMPLES_REPO_ID" \
  --weights-name "$WEIGHTS_NAME" \
  --checkpoint "$CHECKPOINT" \
  --batch-size "${ZEB_BATCH_SIZE:-64}" \
  --replay-buffer-size "${ZEB_REPLAY_BUFFER_SIZE:-500000}" \
  --eval-aux-replay-buffer-size "${ZEB_EVAL_AUX_REPLAY_BUFFER_SIZE:-100000}" \
  --training-steps-per-cycle "${ZEB_TRAINING_STEPS_PER_CYCLE:-1000}" \
  --eval-aux-enabled \
  --eval-aux-batch-fraction "${ZEB_EVAL_AUX_BATCH_FRACTION:-0.25}" \
  --eval-aux-policy-weight "${ZEB_EVAL_AUX_POLICY_WEIGHT:-1.0}" \
  --eval-aux-value-weight "${ZEB_EVAL_AUX_VALUE_WEIGHT:-1.0}" \
  --eval-aux-belief-weight "${ZEB_EVAL_AUX_BELIEF_WEIGHT:-1.0}" \
  --eval-aux-max-model-lag "${ZEB_EVAL_AUX_MAX_MODEL_LAG:-400}" \
  --eval-aux-lag-half-life "${ZEB_EVAL_AUX_LAG_HALF_LIFE:-200}" \
  --eval-aux-min-keep-weight "${ZEB_EVAL_AUX_MIN_KEEP_WEIGHT:-0.10}" \
  --keep-example-files "$KEEP_EXAMPLE_FILES" \
  "${amp_args[@]}" \
  --local-replay-cache-enabled \
  --local-replay-cache-dir "$LOCAL_REPLAY_CACHE_DIR" \
  --local-replay-cache-save-every "$LOCAL_REPLAY_CACHE_SAVE_EVERY" \
  --push-every "$PUSH_EVERY" \
  --wandb \
  --wandb-project "$WANDB_PROJECT" \
  --run-name "$RUN_NAME" \
  >> "$LOG_FILE" 2>&1 &

pid="$!"
echo "$pid" > "$PID_FILE"

echo "Learner started (full-teacher experiment):"
echo "  pid:      $pid"
echo "  pid_file: $PID_FILE"
echo "  log_file: $LOG_FILE"
echo
echo "Config:"
echo "  weights:   $WEIGHTS_NAME"
echo "  bootstrap: $CHECKPOINT"
echo "  mix:       25% eval-aux"
echo "  policy_w:  1.0  value_w: 1.0  belief_w: 1.0"
echo
echo "Monitor:"
echo "  tail -f $LOG_FILE"
