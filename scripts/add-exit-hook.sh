#!/bin/bash

# Script to add exit-on-stop hook to Claude Code settings

SETTINGS_FILE=".claude/settings.json"

if [ ! -f "$SETTINGS_FILE" ]; then
    echo "Creating .claude directory and settings.json..."
    mkdir -p .claude
    echo '{}' > "$SETTINGS_FILE"
fi

# Create the hook configuration
HOOK_CONFIG='{
  "hooks": {
    "Stop": [
      {
        "matcher": "*",
        "hooks": [
          {
            "type": "command",
            "command": "pkill -TERM claude 2>/dev/null || true; sleep 1; pkill -KILL claude 2>/dev/null || true"
          }
        ]
      }
    ]
  }
}'

# Use jq to merge the hook configuration into existing settings
if command -v jq &> /dev/null; then
    # Use jq if available for proper JSON merging
    jq -s '.[0] * .[1]' "$SETTINGS_FILE" <(echo "$HOOK_CONFIG") > "${SETTINGS_FILE}.tmp" && mv "${SETTINGS_FILE}.tmp" "$SETTINGS_FILE"
    echo "Exit-on-stop hook added successfully using jq"
else
    # Fallback: write the complete configuration
    echo "$HOOK_CONFIG" > "$SETTINGS_FILE"
    echo "Exit-on-stop hook added successfully (overwrote existing settings - jq not available)"
fi

echo "Hook configuration:"
cat "$SETTINGS_FILE"