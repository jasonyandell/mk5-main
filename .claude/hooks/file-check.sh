#!/bin/bash

# Claude Code hook script to run checks on modified TypeScript and Svelte files
# Shows issues without blocking - always exits 0

# Read the hook input (JSON)
input=$(cat)

# Extract the file path from the hook input
# The input format includes a 'file_path' field when Write/Edit tools are used
file_path=$(echo "$input" | grep -o '"file_path":"[^"]*"' | cut -d'"' -f4)

# If no file_path found, exit successfully (not a file operation)
if [ -z "$file_path" ]; then
    exit 0
fi

# Get the file extension
extension="${file_path##*.}"

# Change to project directory
cd "$CLAUDE_PROJECT_DIR" || exit 0

# Run appropriate checks based on file extension
case "$extension" in
    ts)
        echo "📝 Checking $file_path..."
        
        # Run TypeScript type checking on single file
        npx tsc --noEmit --skipLibCheck "$file_path" 2>&1
        
        # Run ESLint on single file
        npx eslint "$file_path" 2>&1
        
        echo "---"
        ;;
        
    svelte)
        echo "📝 Checking $file_path..."
        
        # Run Svelte check on single file
        npx svelte-check --tsconfig ./tsconfig.json --output human --threshold warning --fail-on-warnings false "$file_path" 2>&1
        
        echo "---"
        ;;
        
    *)
        # Not a TypeScript or Svelte file, exit successfully
        exit 0
        ;;
esac

# Always exit successfully to avoid blocking
exit 0