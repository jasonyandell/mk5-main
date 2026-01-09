#!/bin/bash
# get_ready_by_epic.sh - Returns one ready leaf bead ID from a parent bead
# Drills down through highest-priority children until reaching a leaf
# Usage: ./get_ready_by_epic.sh <bead_id>

set -e

if [ -z "$1" ]; then
    echo "Usage: $0 <bead_id>" >&2
    exit 1
fi

BEAD_ID="$1"

# Check if bead is closed or blocked
STATUS=$(bd show "$BEAD_ID" 2>/dev/null | grep -E "^Status:" | awk '{print $2}')
if [ "$STATUS" = "closed" ]; then
    echo "$BEAD_ID closed"
    exit 0
fi
if [ "$STATUS" = "blocked" ]; then
    echo "$BEAD_ID blocked"
    exit 0
fi

# Drill down through highest priority children until we hit a leaf
CURRENT="$BEAD_ID"
while true; do
    # Get direct children sorted by priority (bd list returns priority-sorted)
    CHILDREN=$(bd list --parent "$CURRENT" --status=open 2>/dev/null | awk '{print $1}' | head -1)

    if [ -z "$CHILDREN" ]; then
        # No open children - return CURRENT (could be input or a descendant)
        echo "$CURRENT"
        exit 0
    fi

    # Drill down to highest priority child
    CURRENT="$CHILDREN"
done
