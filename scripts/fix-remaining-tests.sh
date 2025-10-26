#!/bin/bash
# Final comprehensive fix for all remaining test errors

for file in src/tests/layers/**/*.test.ts; do
  # Fix GameState import
  sed -i 's/import type { GameState, Trick }/import type { Trick }/g' "$file"
  sed -i 's/import type { GameState, Play, Trick }/import type { Play, Trick }/g' "$file"
  sed -i 's/import type { GameState, Domino }/import type { Domino }/g' "$file"

  # Fix baseActions type
  sed -i 's/const baseActions = \[\]/const baseActions: never[] = []/g' "$file"

  # Fix GameAction property access - value
  sed -i "s/actions\[0\]?.value/actions[0] \&\& actions[0].type === 'bid' ? actions[0].value : undefined/g" "$file"
  sed -i "s/action?.value/action \&\& action.type === 'bid' ? action.value : undefined/g" "$file"

  # Fix GameAction property access - bid
  sed -i "s/actions\[0\]?.bid/actions[0] \&\& actions[0].type === 'bid' ? actions[0].bid : undefined/g" "$file"

  # Fix GameAction property access - dominoId
  sed -i "s/action?.dominoId/action \&\& action.type === 'play' ? action.dominoId : undefined/g" "$file"
  sed -i "s/\\.dominoId === '/.type === 'play' \&\& action.dominoId === '/g" "$file"

  # Add null guards for array access
  sed -i 's/const winner = .*players\[t\.winner\];/const winner = state.players[t.winner]; if (!winner) throw new Error(`Invalid winner index: ${t.winner}`);/g' "$file"
done

echo "Applied all fixes"
echo "Run 'npm run typecheck' to verify"
