#!/bin/bash

# Script to remove exit-on-stop hook from Claude Code settings

SETTINGS_FILE=".claude/settings.json"

if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Settings file not found: $SETTINGS_FILE"
    exit 1
fi

# Use jq to remove the Stop hook if available
if command -v jq &> /dev/null; then
    # Remove the Stop hook while preserving other hooks
    jq 'if .hooks then .hooks |= del(.Stop) else . end | if .hooks == {} then del(.hooks) else . end' "$SETTINGS_FILE" > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
    echo "Exit-on-stop hook removed successfully using jq"
else
    # Fallback: create empty settings file
    echo '{}' > "$SETTINGS_FILE"
    echo "Exit-on-stop hook removed successfully (reset all settings - jq not available)"
fi

echo "Updated configuration:"
cat "$SETTINGS_FILE"