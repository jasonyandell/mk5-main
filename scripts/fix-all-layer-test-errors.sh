#!/bin/bash
# Comprehensive fix for all layer test TypeScript errors

# Fix pattern: { id: N, teamId: T, hand: ... } missing name and marks
for file in src/tests/layers/**/*.test.ts; do
  # Pattern 1: { id: 0, teamId: 0, hand: ... }
  sed -i 's/{ id: 0, teamId: 0, hand:/{ id: 0, name: '\''P0'\'', teamId: 0, marks: 0, hand:/g' "$file"
  sed -i 's/{ id: 1, teamId: 1, hand:/{ id: 1, name: '\''P1'\'', teamId: 1, marks: 0, hand:/g' "$file"
  sed -i 's/{ id: 2, teamId: 0, hand:/{ id: 2, name: '\''P2'\'', teamId: 0, marks: 0, hand:/g' "$file"
  sed -i 's/{ id: 3, teamId: 1, hand:/{ id: 3, name: '\''P3'\'', teamId: 1, marks: 0, hand:/g' "$file"

  # Pattern 2: Remove undefined players (convert to just using default)
  # This is complex, skip for now as it requires understanding context
done

echo "Fixed player object patterns"
echo "Run 'npm run typecheck' to see remaining errors"
