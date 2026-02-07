#!/usr/bin/env bash
# vast_down.sh — Destroy all zeb instances
#
# Usage:
#   ./vast_down.sh            # destroy all zeb-* instances (with confirmation)
#   ./vast_down.sh --force    # skip confirmation

set -euo pipefail

FORCE="${1:-}"

# Get zeb instance IDs
IDS=$(vastai show instances --raw | python3 -c "
import sys, json
instances = json.load(sys.stdin)
zeb = [i for i in instances if (i.get('label') or '').startswith('zeb-')]
for i in zeb:
    print(f\"{i['id']} {i.get('label', '?')} {i.get('gpu_name', '?')} {i.get('actual_status', '?')}\")
")

if [ -z "$IDS" ]; then
    echo "No zeb instances found."
    exit 0
fi

echo "Instances to destroy:"
echo "$IDS" | while read -r id label gpu status; do
    echo "  ${label} (${gpu}, ${status}) — ID ${id}"
done

N=$(echo "$IDS" | wc -l)

if [ "$FORCE" != "--force" ]; then
    echo ""
    read -p "Destroy ${N} instance(s)? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

echo "$IDS" | while read -r id rest; do
    echo "  Destroying ${id}..."
    vastai destroy instance "$id"
done

echo "Done. ${N} instance(s) destroyed."
