#!/usr/bin/env bash
# vast_replenish.sh — Replace dead workers
#
# Checks for workers that are no longer running and creates replacements.
# Safe to run from cron: only acts when workers are missing.
#
# Usage:
#   ./vast_replenish.sh 4          # ensure 4 workers exist
#   */5 * * * * cd /path/to/vast && ./vast_replenish.sh 4 >> /tmp/vast-replenish.log 2>&1

set -euo pipefail

DESIRED="${1:?Usage: vast_replenish.sh N_WORKERS}"

# Count running zeb-worker instances
RUNNING=$(vastai show instances --raw | python3 -c "
import sys, json
instances = json.load(sys.stdin)
workers = [i for i in instances
           if (i.get('label') or '').startswith('zeb-worker-')
           and i.get('actual_status') in ('running', 'loading')]
print(len(workers))
")

if [ "$RUNNING" -ge "$DESIRED" ]; then
    echo "$(date '+%H:%M') OK: ${RUNNING}/${DESIRED} workers running"
    exit 0
fi

NEEDED=$((DESIRED - RUNNING))
echo "$(date '+%H:%M') REPLENISH: ${RUNNING}/${DESIRED} workers running, launching ${NEEDED} more"

# Delegate to vast_up.sh
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/vast_up.sh" "$NEEDED"
