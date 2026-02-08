#!/usr/bin/env bash
# vast_replenish.sh — Replace dead workers
#
# Checks for workers that are no longer running and creates replacements.
# Safe to run from cron: only acts when workers are missing.
#
# Usage:
#   ./vast_replenish.sh 4                    # ensure 4 zeb workers exist
#   ./vast_replenish.sh 4 zeb-large          # ensure 4 zeb-large workers exist
#   */5 * * * * cd /path/to/vast && ./vast_replenish.sh 4 >> /tmp/vast-replenish.log 2>&1

set -euo pipefail

DESIRED="${1:?Usage: vast_replenish.sh N_WORKERS [FLEET]}"
FLEET="${2:-zeb}"

# Count running workers for this fleet
RUNNING=$(vastai show instances --raw | python3 -c "
import sys, json
fleet = '$FLEET'
instances = json.load(sys.stdin)
workers = [i for i in instances
           if (i.get('label') or '').startswith(fleet + '-worker-')
           and i.get('actual_status') in ('running', 'loading')]
print(len(workers))
")

if [ "$RUNNING" -ge "$DESIRED" ]; then
    echo "$(date '+%H:%M') OK: ${FLEET} ${RUNNING}/${DESIRED} workers running"
    exit 0
fi

NEEDED=$((DESIRED - RUNNING))
echo "$(date '+%H:%M') REPLENISH: ${FLEET} ${RUNNING}/${DESIRED} workers running, launching ${NEEDED} more"

# Delegate to vast_up.sh (inherits ZEB_* env vars for fleet config)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export ZEB_FLEET="$FLEET"
exec "${SCRIPT_DIR}/vast_up.sh" "$NEEDED"
