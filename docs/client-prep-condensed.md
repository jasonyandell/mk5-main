# Client Preparation Plan (Condensed)

## Goal
Make state so clean that client bugs become nearly impossible. Enable undo/redo, clean PartyKit integration, and AI players.

## Problems to Fix

1. **Nullable fields**: `winningBidder`, `trump`, `currentSuit` are null when they shouldn't be
2. **Confusing Trump type**: Mix of magic numbers (7, 8) and objects
3. **No action history**: Can't undo or replay

## Implementation Steps

### Step 1: Define Actions from Existing Transition IDs
Extract from current getNextStates() transition IDs:
```typescript
type GameAction = 
  | { type: 'bid'; player: number; value: number }
  | { type: 'pass'; player: number }
  | { type: 'select-trump'; player: number; selection: TrumpSelection }
  | { type: 'play'; player: number; dominoId: string }
  | { type: 'complete-trick' }
  | { type: 'score-hand' }
```

### Step 2: Create Pure Action Functions
```typescript
function applyAction(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case 'bid': return applyBid(state, action.player, action.value);
    // ... etc
  }
}
```

### Step 3: Fix Trump Type  
Replace magic numbers (7=doubles, 8=no-trump) with clear types:
```typescript
interface TrumpSelection {
  type: 'none' | 'suit' | 'doubles' | 'no-trump';
  suit?: 0 | 1 | 2 | 3 | 4 | 5 | 6;  // Only when type === 'suit'
}
```

### Step 4: Essential Computed Fields to Keep
- `player.suitAnalysis` - CRITICAL to rules engine, do not remove
- `trick.winner` - Used throughout scoring

### Step 5: Replace Nulls with Empty States
```typescript
interface GameState {
  // Instead of null, use clear empty values:
  bidResult: { winner: number; bid: Bid } | { type: 'none' };
  trump: TrumpSelection;  // Never null, uses { type: 'none' }
  currentSuit: number;    // -1 when no trick in progress
  winningBidder: number;  // -1 during bidding
}
```

### Step 6: Add History
```typescript
interface GameEngine {
  private state: GameState;
  private history: GameAction[] = [];
  private stateSnapshots: GameState[] = [];
  
  executeAction(action: GameAction): void {
    this.history.push(action);
    this.stateSnapshots.push(this.state);
    this.state = applyAction(this.state, action);
  }
  
  undo(): void {
    if (this.stateSnapshots.length > 0) {
      this.history.pop();
      this.state = this.stateSnapshots.pop()!;
    }
  }
}
```

### Step 7: Update getNextStates()
```typescript
export function getNextStates(state: GameState): StateTransition[] {
  const validActions = getValidActions(state);
  return validActions.map(action => ({
    id: actionToId(action),
    label: actionToLabel(action), 
    newState: applyAction(state, action)
  }));
}
```

## Important Notes
- Extract actions from existing `getNextStates()` transition IDs
- Keep the same public API - `getNextStates()` still works but uses actions internally
- JSON serialization must work without custom handling
- Comprehensive test suite makes this refactor safe

## Success Criteria
- No nullable fields (use -1 or empty objects)
- Actions are pure data objects
- State changes only through applyAction()
- Full undo/redo support
- All tests still pass