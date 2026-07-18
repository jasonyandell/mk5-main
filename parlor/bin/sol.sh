#!/usr/bin/env bash
# Send a message to Sol's persistent session; message on stdin, reply on stdout.
# First call creates the session; later calls continue it.
set -euo pipefail
DIR="$(cd "$(dirname "$0")/.." && pwd)"
SDIR="$DIR/sessions/sol"
MSG="$(cat)"
ARGS=(--print --no-tools --provider openai-codex --model gpt-5.6-sol --session-dir "$SDIR")
if ls "$SDIR"/*.jsonl >/dev/null 2>&1; then
  ARGS+=(--continue)
fi
# leading space keeps a message that starts with '-' from parsing as a flag
pi "${ARGS[@]}" " $MSG"
