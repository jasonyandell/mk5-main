# Architecture Deep Dive

## Mental Models

### Game as State Machine
Every game position has defined transitions. Actions are edges, states are nodes. AI explores this graph. Games flow through well-defined positions via actions.

### Layers as Lenses and Decorators
Each layer provides a lens (execution rules) AND decorator (action transformation). Stack layers to create new game modes. Nello adds a "3-player trick" lens, OneHand scripts bidding as a decorator.

### Capabilities as Keys
Each capability unlocks specific functionality. `observe-all-hands` unlocks full visibility, `act-as-player` unlocks actions for a seat.

### Kernel as Pure Function
Given state and action, always produces same new state. No hidden state or side effects. `newState = f(oldState, action)` always holds.

## GameRules Interface (14 Methods)

```typescript
interface GameRules {
  // WHO: Player determination
  getTrumpSelector(state, winningBid): number
  getFirstLeader(state, trumpSelector, trump): number
  getNextPlayer(state, currentPlayer): number

  // WHEN: Timing and completion
  isTrickComplete(state): boolean
  checkHandOutcome(state): HandOutcome

  // HOW: Game mechanics
  getLedSuit(state, domino): LedSuit
  calculateTrickWinner(state, trick): number

  // VALIDATION: Legality
  isValidPlay(state, domino, playerId): boolean
  getValidPlays(state, playerId): Domino[]
  isValidBid(state, bid, playerHand?): boolean

  // SCORING: Outcomes
  getBidComparisonValue(bid): number
  isValidTrump(trump): boolean
  calculateScore(state): [number, number]

  // LIFECYCLE
  getPhaseAfterHandComplete(state): Phase
}
```

Each rule gets `prev` parameter - delegate to previous layer or override. Compose via reduce.

## Layer Interface

```typescript
interface Layer {
  name: string;
  rules?: {
    // Each rule receives `prev` (previous layer's result) as last param
    isTrickComplete?: (state: GameState, prev: boolean) => boolean;
    getTrumpSelector?: (state: GameState, winningBid: Bid, prev: number) => number;
    // ... etc for each GameRules method
  };
  getValidActions?: (state, prev) => GameAction[];  // Transform actions
}
```

**Note**: `rules` is NOT `Partial<GameRules>` - it's a threaded composition type where each method receives the previous layer's result as its last parameter.

### Execution Rules Surface
Define HOW the game executes (who acts, when tricks complete, how winners determined).

### Action Generation Surface
Transform WHAT actions are possible:
- **Filter**: Remove actions (tournament removes special bids)
- **Annotate**: Add metadata (hints, autoExecute flags)
- **Script**: Inject actions (oneHand scripts bidding)
- **Replace**: Swap action types (oneHand replaces score-hand with end-game)

## Terminology Note

Traditional Texas 42 term "follow-me" is called `no-trump` in the codebase. When reading game rules or talking to players, "follow-me" = `{ type: 'no-trump' }` in code.

## How to Add a New Layer

1. Create `src/game/layers/myLayer.ts`:
```typescript
export const myLayer: Layer = {
  name: 'my-layer',
  rules: {
    // Override execution behavior
    isTrickComplete: (state, prev) =>
      state.trump.type === 'my-contract' ? customLogic : prev,
  },
  getValidActions: (state, prev) => {
    // Transform actions
    const actions = prev(state);
    return actions.filter(/* custom logic */);
  }
};
```

2. Add to `src/game/layers/utilities.ts:getEnabledLayers()`:
```typescript
if (config.enableMyLayer) layers.push(myLayer);
```

3. Add config flag to `GameConfig` type.

**That's it.** No changes to core executors.

## How to Add a New Rule Method

When a new mode needs different execution semantics:

### Step 1: Identify the extension point
- Executor needs to know about behavior? → Add GameRules method
- Just different available actions? → Use Layer's getValidActions

### Step 2: Add to GameRules interface
```typescript
// src/game/layers/types.ts
export interface GameRules {
  /** Doc comment explaining rule */
  myNewRule(state: GameState, ...params): ReturnType;
}
```

### Step 3: Implement in base layer
```typescript
// src/game/layers/base.ts
export const baseRules: GameRules = {
  myNewRule: (state, ...params) => defaultValue
};
```

### Step 4: Override in mode layers
```typescript
// src/game/layers/myMode.ts
myNewRule: (state, ...params, prev) => {
  if (state.trump.type === 'my-mode') return modeSpecificValue;
  return prev;
}
```

### Step 5: Update compose.ts
Add rule to composition reduce pattern.

### Step 6: Use in executors
```typescript
// DON'T: if (state.mode === 'special') { ... }
// DO: Delegate to rule
const value = rules.myNewRule(state, action.param);
```

## Layer State Inspection Pattern

Layers check `state.trump.type` to determine if their rules apply:

```typescript
// nello.ts
isTrickComplete: (state, prev) =>
  state.trump?.type === 'nello'
    ? state.currentTrick.length === 3
    : prev
```

**This is intentional.** Texas 42's special contracts are:
1. Composed at config time (nello layer always composed when enabled)
2. *Activated* by player choice during trump selection

The state check is:
- **Explicit** - Says exactly what it means
- **Local** - All nello logic in nello.ts
- **Necessary** - Matches how Texas 42 special contracts work

## Architectural Invariants

1. **Pure State Storage**: Room stores unfiltered GameState, filters on-demand
2. **Server Authority**: Client trusts server's validActions, never refilters
3. **Capability-Based Access**: Permissions via tokens, not identity checks
4. **Single Composition Point**: Room/HeadlessRoom ONLY (ESLint enforced)
5. **Zero Coupling**: Core engine has zero knowledge of multiplayer/layers
6. **Parametric Polymorphism**: Executors call `rules.method()`, never inspect mode
7. **Event Sourcing**: State derivable from `replayActions(config, history)`

**Violation of any invariant = architectural regression.**

## Common Pitfalls

| Pitfall | Solution |
|---------|----------|
| Conditional logic in executors | Delegate to `rules.method()` |
| Modifying state directly | Spread operator for new objects |
| Identity checks for permissions | Capability tokens |
| Game logic in Room | Put in pure helpers or Layers |
| Client-side validation | Trust server's validActions |
| Deep coupling between layers | Single responsibility per layer |

## Capability System

```typescript
type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-hands'; playerIndices: number[] | 'all' };
```

**Standard builders** (`src/multiplayer/capabilities.ts`):
- `humanCapabilities(playerIndex)` → act + observe own hand
- `aiCapabilities(playerIndex)` → act + observe own hand
- `spectatorCapabilities()` → observe all hands

## Action Authority

```typescript
type ActionAuthority = 'player' | 'system';
```

- **Player**: Authorized by session capabilities
- **System**: Deterministic game script (oneHand transformer), bypasses capability checks

## Pacing Layers

Two layers control game flow with opposite philosophies:

### Consensus Layer

Gates progress actions (`complete-trick`, `score-hand`) behind human acknowledgment.

```typescript
// Without consensus: complete-trick executes immediately
// With consensus: all humans must agree-trick first

getValidActions: (state, prev) => {
  // If complete-trick available but not all humans acknowledged
  // → replace with agree-trick actions for each unacknowledged human
  // Once all humans acknowledged → allow complete-trick
}
```

**Key characteristics**:
- Reads `state.playerTypes` to identify human players
- AI players don't vote (they're not waiting to see results)
- Derives acknowledgment state from `actionHistory` (pure function)
- Introduces new action types: `agree-trick`, `agree-score`

**Use for**: Multiplayer human games needing "tap to continue" pacing.

### Speed Layer

Auto-executes forced moves (single legal actions).

```typescript
getValidActions: (state, prev) => {
  // Group by player, if player has exactly one action
  // → annotate with autoExecute: true, authority: 'system'
}
```

**Key characteristics**:
- Marks actions with `autoExecute: true` for immediate execution
- Sets `authority: 'system'` to bypass capability checks
- Adds metadata: `speedMode: true`, `reason: 'only-legal-action'`

**Use for**: AI-only games, single-player practice, faster gameplay.

### Composing Pacing Layers

These layers can be composed together:
- With both enabled, consensus gates progress
- But speed auto-executes the agree actions when players have no other choices
- Result: Human sees trick result, taps once, game advances
