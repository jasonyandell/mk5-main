# Client Preparation Plan

## Motivation

We want to make client development so simple it's hard to write bugs. Every nullable field, magic number, or implicit relationship is a place where clients can go wrong. By cleaning up our state to have zero tech debt, we enable:

1. Dead-simple client development
2. Perfect undo/redo for "what if we had..." discussions
3. Clean integration with PartyKit (just send action IDs)
4. AI players that can explore strategies without side effects

Our comprehensive test suite makes this refactor safe to attempt.

## Current Tech Debt

### Nullable Fields That Shouldn't Be
```typescript
winningBidder: number | null;  // Null during bidding
trump: Trump | null;           // Null until selected
currentSuit: number | null;    // Null between tricks
```

### Confusing Trump Type
```typescript
type Trump = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | { suit: string | number; followsSuit: boolean };
// What's 7? What's 8? Why sometimes an object?
```

### Mixed Concerns
```typescript
interface Player {
  hand: Domino[];          // Secret data
  suitAnalysis?: SuitAnalysis;  // ESSENTIAL - Keep this! Used extensively by rules engine
}
```

## Step-by-Step Cleanup

### Step 1: Extract Actions from getNextStates()

Our transition IDs already define our actions. Formalize them:

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

Extract the logic from `getNextStates()` into focused functions:

```typescript
function applyAction(state: GameState, action: GameAction): GameState {
  switch (action.type) {
    case 'bid': return applyBid(state, action.player, action.value);
    case 'play': return applyPlay(state, action.player, action.dominoId);
    // etc...
  }
}
```

### Step 3: Fix the Trump Type

Replace magic numbers with clear types that serialize cleanly:

```typescript
interface TrumpSelection {
  type: 'none' | 'suit' | 'doubles' | 'no-trump';
  suit?: 0 | 1 | 2 | 3 | 4 | 5 | 6;  // Only present when type === 'suit'
}

// In GameState, trump is never null:
interface GameState {
  trump: TrumpSelection;  // { type: 'none' } before selection
}
```

### Step 4: Keep Essential Computed Fields

After analyzing the codebase, these computed fields MUST stay:

```typescript
interface Player {
  hand: Domino[];
  suitAnalysis?: SuitAnalysis;  // CRITICAL - Core to rules engine, used in:
                               // - canFollowSuit() checks
                               // - getValidPlays() filtering
                               // - doubles trump handling
                               // - state cloning/serialization
}

interface Trick {
  plays: Play[];
  winner?: number;  // Keep - used extensively for scoring
  points: number;
}
```

**Why suitAnalysis is NOT just optimization - it's ESSENTIAL:**
- **Rules Engine Dependency**: `canFollowSuit()` and `getValidPlays()` in rules.ts rely on it
- **Doubles Trump Logic**: Special handling for doubles (suit 7) uses `suitAnalysis.rank.doubles`
- **Action System**: The entire valid play determination system depends on suit analysis
- **State Updates**: Automatically updated when trump changes or dominoes are played
- **Removing it would break**: The rules engine would need complete rewriting

These are exceptions to "no computed fields" because:
- **suitAnalysis**: Core to action system, prevents redundant calculations AND is deeply integrated
- **trick.winner**: Used throughout scoring logic, would be inefficient to recompute
- Both are deterministic from other state
- Removing them would complicate client code significantly

### Step 5: Clean Up Nullables While Keeping JSON-Friendly

Instead of discriminated unions (which don't serialize well), use clear empty states:

```typescript
interface GameState {
  phase: GamePhase;
  dealer: number;
  currentPlayer: number;
  
  // Always present, but with clear "empty" values
  bidResult: { winner: number; bid: Bid } | { type: 'none' };
  trump: TrumpSelection;  // Never null, uses { type: 'none' }
  
  // Use -1 for "no value" instead of null
  currentSuit: number;  // -1 when no trick in progress
  winningBidder: number;  // -1 during bidding
  
  // Always arrays, sometimes empty
  bids: Bid[];
  currentTrick: Play[];
  completedTricks: Trick[];
  
  players: Player[];
  teamScores: [number, number];
  teamMarks: [number, number];
}
```

This approach:
- Serializes perfectly to JSON
- No null checks needed
- TypeScript can still provide safety with type guards
- AI can work with consistent state shape

### Step 6: Add Action History

Simple history for undo/redo:

```typescript
interface GameEngine {
  private state: GameState;
  private history: GameAction[] = [];
  private stateSnapshots: GameState[] = [];  // For fast undo
  
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

### Step 7: Update getNextStates() to Use New System

Keep the same API but use actions internally:

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

## Benefits for Client Development

### Before: Client Nightmare
```typescript
// Is there a trump? Is it a number? What does 7 mean?
if (state.trump !== null) {
  if (typeof state.trump === 'number') {
    if (state.trump === 7) { /* doubles??? */ }
  }
}
```

### After: Client Dream
```typescript
// Clear and simple
if (state.trump.type === 'doubles') { /* no nulls, no magic numbers */ }
```

### PartyKit Integration: Trivial
```typescript
// Client sends:
{ actionId: "bid-30" }

// Server:
const action = parseActionId(actionId);  // { type: 'bid', player: 0, value: 30 }
state = applyAction(state, action);
broadcast(state);
```

### AI Player: Elegant
```typescript
// AI can explore without side effects
class SimpleAI {
  findBestMove(state: GameState): GameAction {
    const actions = getValidActions(state);
    
    return actions.reduce((best, action) => {
      const future = applyAction(state, action);
      const score = this.evaluate(future);
      return score > best.score ? { action, score } : best;
    }, { action: actions[0], score: -Infinity }).action;
  }
}
```

## Success Metrics

1. **No null checks needed** - Everything has a clear empty state
2. **Actions are just data** - `{ type: 'bid', player: 0, value: 30 }`
3. **Undo is one line** - `engine.undo()`
4. **Essential computed fields retained** - suitAnalysis is CRITICAL to rules engine, not just optimization
5. **Clear types** - No magic numbers, no confusing unions
6. **AI can explore freely** - Pure functions enable tree search
7. **JSON serialization just works** - No custom handling needed

## Migration Safety

- Keep all existing tests passing
- Extract one action type at a time
- Run parallel validation (old vs new should produce same state)
- Use TypeScript's type checking to catch issues

## Why This Approach Works

1. **For Clients**: No nullable fields means no forgotten null checks
2. **For PartyKit**: Actions as data means tiny network messages
3. **For AI**: Pure functions mean side-effect-free exploration
4. **For Testing**: Deterministic actions mean reproducible tests
5. **For "What If"**: Action history means perfect replay

The result: A foundation that makes clients, AI, and multiplayer all trivial to implement correctly.