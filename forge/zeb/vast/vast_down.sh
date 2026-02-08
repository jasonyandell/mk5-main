#!/usr/bin/env bash
# vast_down.sh — Destroy zeb instances
#
# Usage:
#   ./vast_down.sh                # destroy all zeb-* instances (with confirmation)
#   ./vast_down.sh zeb-large      # destroy only zeb-large-* instances
#   ./vast_down.sh --force        # skip confirmation (all zeb-*)
#   ./vast_down.sh zeb-large --force

set -euo pipefail

# Parse args: optional fleet prefix and --force flag
FLEET="zeb"
FORCE=""
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE="--force"
    else
        FLEET="$arg"
    fi
done

# Get matching instance IDs
IDS=$(vastai show instances --raw | python3 -c "
import sys, json
fleet = '$FLEET'
instances = json.load(sys.stdin)
zeb = [i for i in instances if (i.get('label') or '').startswith(fleet + '-')]
for i in zeb:
    print(f\"{i['id']} {i.get('label', '?')} {i.get('gpu_name', '?')} {i.get('actual_status', '?')}\")
")

if [ -z "$IDS" ]; then
    echo "No ${FLEET}-* instances found."
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
