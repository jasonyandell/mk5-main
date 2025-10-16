# Game Onboarding - Texas 42 Architecture

**Purpose**: Get new developers productive quickly with the Texas 42 codebase architecture.

---

## North Star: Pure Functional Composition

### Core Principles

1. **Event Sourcing**: `state = replayActions(initialConfig, actionHistory)`
2. **Pure Functions**: All state transitions are pure, reproducible
3. **Single Transformer Surface**: Variants only transform `getValidActions` (the state machine)
4. **Composition Over Configuration**: `f(g(h(base)))` instead of flags and branches
5. **Zero Coupling**: Core engine has no variant/multiplayer awareness

### Why This Matters

- **Reproducibility**: Any state can be reconstructed from config + history
- **Testability**: Pure functions compose and test in isolation
- **Debuggability**: Time-travel debugging, replay from any point
- **Extensibility**: Add variants without touching core
- **Simplicity**: One composition point, not scattered hooks

---

## Architecture Layers

```
┌─────────────────────────────────┐
│  UI Layer (Svelte)              │  src/App.svelte, src/stores/
│  Reactive stores, view logic    │
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Client Layer                   │  src/game/multiplayer/
│  NetworkGameClient, protocol    │
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Server Layer                   │  src/server/
│  GameHost, GameRegistry         │
│  **COMPOSITION HAPPENS HERE**   │
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Variant Layer                  │  src/game/variants/
│  Pure function transformers     │
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Core Engine                    │  src/game/core/
│  Pure game logic, zero coupling │
└─────────────────────────────────┘
```

---

## Initialization Flow

### User Loads Page

```
1. src/App.svelte
   └─> Imports gameStore

2. src/stores/gameStore.ts (lines 28-35)
   - Creates InProcessAdapter()
   - Creates NetworkGameClient(adapter, config)
   - Client internally sends CREATE_GAME message

3. src/server/offline/InProcessAdapter.ts
   - Handles CREATE_GAME message
   - Calls GameRegistry.createGame(config)

4. src/server/game/GameHost.ts (GameRegistry.createGame, line 404)
   - Creates PlayerSessions from config
   - Calls new GameHost(gameId, config, players)

5. GameHost Constructor (line 52-93) ⭐ COMPOSITION POINT
   - Converts variant format (old → new)
   - Calls applyVariants(baseGetValidActions, variantConfigs)
   - Stores composed function in this.getValidActionsComposed
   - Creates initial state via createInitialState()

6. Game Ready
   - GameHost.createView() uses composed function
   - UI subscribes to state updates
```

### Key Composition Code

```typescript
// GameHost.ts:72-81
const variantConfigs: VariantConfig[] = config.variant
  ? [{ type: config.variant.type, config: config.variant.config }]
  : [];

this.getValidActionsComposed = applyVariants(
  baseGetValidActions,  // from src/game/core/gameEngine.ts
  variantConfigs
);
```

---

## Variant System

### What Variants Are

**Variants transform the state machine** (function that produces valid actions).

```typescript
// Type signature
type StateMachine = (state: GameState) => GameAction[]
type Variant = (base: StateMachine) => StateMachine
```

### How They Work

```typescript
// Composition (src/game/variants/registry.ts:35-41)
function applyVariants(base, variants) {
  return variants
    .map(v => getVariant(v.type, v.config))
    .reduce((machine, variant) => variant(machine), base);
}

// Result: tournament(oneHand(baseGetValidActions))
```

### Variant Operations

Variants can:
1. **Filter**: Remove actions (tournament removes nello/splash/plunge)
2. **Annotate**: Add metadata (hints, autoExecute flags)
3. **Script**: Inject new actions (one-hand injects scripted bidding)
4. **Replace**: Swap actions (one-hand replaces score-hand)

### Example: Tournament Variant

**File**: `src/game/variants/tournament.ts` (19 lines)

```typescript
export const tournamentVariant: VariantFactory = () => (base) => (state) => {
  const actions = base(state);

  if (state.phase !== 'bidding') return actions;

  // Filter out special bids
  return actions.filter(action =>
    action.type !== 'bid' ||
    !['nello', 'splash', 'plunge'].includes(action.bid)
  );
};
```

**Effect**: Only point and mark bids available.

### Example: One-Hand Variant

**File**: `src/game/variants/oneHand.ts` (79 lines)

```typescript
export const oneHandVariant: VariantFactory = () => (base) => (state) => {
  const baseActions = base(state);

  // At start, inject scripted bidding sequence
  if (state.phase === 'bidding' && state.bids.length === 0) {
    return [
      { type: 'bid', player: 3, bid: 'points', value: 30, autoExecute: true, meta: {...} },
      { type: 'pass', player: 0, autoExecute: true, meta: {...} },
      { type: 'pass', player: 1, autoExecute: true, meta: {...} },
      { type: 'pass', player: 2, autoExecute: true, meta: {...} }
    ];
  }

  // After scoring, end game instead of new hand
  if (state.phase === 'scoring' && state.consensus.scoreHand.size === 4) {
    return [
      { type: 'score-hand', autoExecute: true, meta: { scriptId: 'one-hand-end' } }
    ];
  }

  return baseActions;
};
```

**Effect**: Skips bidding via scripted actions, ends after one hand.

### Scripted Actions

Actions with `autoExecute: true` and `meta.scriptId`:

- **autoExecute**: Signals GameHost to execute immediately
- **meta.scriptId**: Debug identifier, test assertions
- **Deterministic**: Must be pure functions of (state, history, params)

---

## Event Sourcing

### The Fundamental Equation

```typescript
state = replayActions(initialConfig, actionHistory)
```

### State Structure

```typescript
interface GameState {
  // Source of truth
  initialConfig: GameConfig;      // How game was configured
  actionHistory: GameAction[];    // Every action taken

  // Derived state (can be recomputed from above)
  phase: GamePhase;
  players: Player[];
  // ... everything else
}
```

### Replay Function

**File**: `src/game/core/replay.ts:16-41`

```typescript
export function replayActions(
  config: GameConfig,
  actions: GameAction[]
): GameState {
  // Compose variants
  const variantConfigs = config.variant
    ? [{ type: config.variant.type, config: config.variant.config }]
    : (config.variants || []);

  // Create initial state
  let state = createInitialState({
    playerTypes: config.playerTypes,
    shuffleSeed: config.shuffleSeed
  });

  // Add initialConfig
  state = { ...state, initialConfig: config };

  // Replay all actions
  for (const action of actions) {
    state = executeAction(state, action);
  }

  return state;
}
```

### Use Cases

1. **URL Replay**: Encode config + actions in URL, reconstruct state
2. **Testing**: Create state at specific point via action sequence
3. **Debugging**: Replay to any step in history
4. **Network Sync**: Send config + actions instead of full state

---

## Key Files Reference

### Core Engine (src/game/core/)

- **gameEngine.ts**: `getValidActions()` - base state machine
- **actions.ts**: `executeAction()` - pure state transitions
- **state.ts**: `createInitialState()` - state creation
- **rules.ts**: `isValidBid()`, `getValidPlays()` - game rules
- **replay.ts**: `replayActions()` - event sourcing

### Variants (src/game/variants/)

- **types.ts**: `StateMachine`, `Variant` type definitions
- **registry.ts**: `applyVariants()` - composition function
- **tournament.ts**: Tournament mode (filter special bids)
- **oneHand.ts**: One-hand mode (scripted actions)

### Server (src/server/)

- **game/GameHost.ts**: Game authority, composition happens here
- **game/GameHost.ts:52-93**: Constructor with variant composition
- **game/GameHost.ts:208-212**: `createView()` uses composed actions
- **offline/InProcessAdapter.ts**: In-memory game adapter

### Multiplayer (src/game/multiplayer/)

- **NetworkGameClient.ts**: Client-side game interface
- **authorization.ts**: `authorizeAndExecute()` - validate actions
- **types.ts**: `MultiplayerGameState`, `PlayerSession` types

### Stores (src/stores/)

- **gameStore.ts**: Svelte reactive stores, client creation
- **gameStore.ts:28-35**: Initial client setup
- **gameStore.ts:196**: `startOneHand()` - variant creation example

### Protocol (src/shared/multiplayer/)

- **protocol.ts**: `GameConfig`, `VariantConfig`, message types

---

## Action Flow

### User Clicks Action

```
1. UI Button Click
   └─> gameActions.executeAction(transition)

2. src/stores/gameStore.ts:123
   └─> gameClient.requestAction(playerId, action)

3. src/game/multiplayer/NetworkGameClient.ts
   └─> adapter.send({ type: 'EXECUTE_ACTION', ... })

4. src/server/offline/InProcessAdapter.ts
   └─> GameHost.executeAction(playerId, action)

5. src/server/game/GameHost.ts:97-116
   └─> authorizeAndExecute(mpState, request)

6. src/game/multiplayer/authorization.ts:57-120
   ├─> Check: Is action in getValidActions()?
   ├─> Check: Is player authorized?
   └─> Execute: executeAction(state, action)

7. src/game/core/actions.ts:16
   └─> Pure state transition, returns new state

8. GameHost.notifyListeners()
   └─> All subscribers get new state

9. UI Re-renders
```

### Valid Actions Flow

```
GameHost.createView() called
  ↓
this.getValidActionsComposed(state)
  ↓
Executes composed variants:
  tournament(oneHand(baseGetValidActions))(state)
  ↓
  1. baseGetValidActions(state) - returns all actions
  2. oneHand wraps it - injects scripted actions if needed
  3. tournament wraps that - filters special bids
  ↓
Returns filtered/transformed action list
  ↓
Authorization filters by player
  ↓
Client receives personalized action list
```

---

## Adding a New Variant

### Step-by-Step

1. **Create variant file**: `src/game/variants/myVariant.ts`

```typescript
import type { VariantFactory } from './types';

export const myVariant: VariantFactory = (config?) => (base) => (state) => {
  const actions = base(state);

  // Your transformation logic
  // - Filter: actions.filter(...)
  // - Annotate: actions.map(a => ({ ...a, hint: '...' }))
  // - Script: return [{ autoExecute: true, ... }]

  return modifiedActions;
};
```

2. **Register variant**: `src/game/variants/registry.ts`

```typescript
import { myVariant } from './myVariant';

const VARIANT_REGISTRY: Record<string, VariantFactory> = {
  'tournament': tournamentVariant,
  'one-hand': oneHandVariant,
  'my-variant': myVariant,  // ADD THIS
};
```

3. **Use variant**: `src/stores/gameStore.ts`

```typescript
const config: GameConfig = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],
  variant: { type: 'my-variant', config: { /* params */ } }
};
```

4. **Test variant**:

```typescript
// Unit test
const base = (state) => [{ type: 'bid', value: 30 }];
const variant = myVariant();
const composed = variant(base);
const result = composed(testState);
expect(result).toEqual([/* expected actions */]);
```

### Variant Patterns

**Filter Pattern**:
```typescript
return actions.filter(action => /* keep condition */);
```

**Annotate Pattern**:
```typescript
return actions.map(action => ({
  ...action,
  metadata: 'value',
  autoExecute: shouldAutoExecute(action)
}));
```

**Script Pattern**:
```typescript
if (shouldInjectScript(state)) {
  return [
    { type: 'action', autoExecute: true, meta: { scriptId: 'my-script' } }
  ];
}
return actions;
```

**Replace Pattern**:
```typescript
return actions.map(action => {
  if (action.type === 'score-hand') {
    return { type: 'end-game', ...customData };
  }
  return action;
});
```

---

## Testing Patterns

### Unit Test Variants

```typescript
import { myVariant } from '../variants/myVariant';

test('my variant filters special bids', () => {
  const baseActions = [
    { type: 'bid', bid: 'points', value: 30 },
    { type: 'bid', bid: 'nello', value: 1 }
  ];

  const base = () => baseActions;
  const composed = myVariant()(base);
  const result = composed(mockState);

  expect(result).toHaveLength(1);
  expect(result[0].bid).toBe('points');
});
```

### Integration Test with Replay

```typescript
import { replayActions } from '../core/replay';

test('can replay game with variant', () => {
  const config = {
    playerTypes: ['human', 'ai', 'ai', 'ai'],
    shuffleSeed: 42069,
    variant: { type: 'one-hand' }
  };

  const actions = [
    // Actions recorded during gameplay
  ];

  const state = replayActions(config, actions);

  expect(state.phase).toBe('playing');
  expect(state.actionHistory).toEqual(actions);
});
```

### Test State Creation

```typescript
// Via replay (preferred)
const state = replayActions(
  { shuffleSeed: 123, variant: { type: 'tournament' } },
  [
    { type: 'bid', player: 0, bid: 'points', value: 30 },
    { type: 'pass', player: 1 }
  ]
);

// Via helper (during transition)
const state = createInitialState({ shuffleSeed: 123 });
```

---

## Common Tasks

### Start One-Hand Game

```typescript
// From UI
gameVariants.startOneHand(seed)

// From code
const config: GameConfig = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],
  shuffleSeed: 12345,
  variant: { type: 'one-hand' }
};
const client = new NetworkGameClient(adapter, config);
```

### Check Variant Composition

```typescript
// In GameHost constructor
console.log('Variants:', variantConfigs);
// In createView
console.log('Actions before:', baseGetValidActions(state).length);
console.log('Actions after:', this.getValidActionsComposed(state).length);
```

### Debug Action Authorization

```typescript
// In authorization.ts:57-120
const validActions = getValidActions(state);
console.log('Valid actions:', validActions);
console.log('Requested action:', action);
console.log('Player index:', playerIndex);
console.log('Is valid:', isValidAction);
```

### Time-Travel Debug

```typescript
// Replay to specific action
const stateAtAction50 = replayActions(
  state.initialConfig,
  state.actionHistory.slice(0, 50)
);
console.log('State at action 50:', stateAtAction50);
```

---

## Architecture Decisions

### Why Single Transformer Surface?

**Alternative**: Multiple hooks (transformInitialState, transformExecuteAction, transformGetValidActions)

**Problem**: Variants scattered across 3 hooks, composition unclear, easy to miss coverage

**Solution**: Single surface - variants only transform action list

**Tradeoff**: Variants must use scripted actions for setup (more complex) but composition is clean

### Why Event Sourcing?

**Benefits**:
- Perfect reproducibility (URL sharing, bug reports)
- Time-travel debugging
- Undo/redo for free
- Network sync simplified (send actions, not state)

**Cost**: Replay can be expensive for long games

**Mitigation**: Only replay when needed (URL load, debug), not on every action

### Why Pure Functions?

**Benefits**:
- Deterministic testing
- No hidden state
- Composable
- Debuggable

**Cost**: Slightly more verbose (create new objects)

**Mitigation**: Use immutable patterns, structural sharing

---

## Troubleshooting

### Variant Not Applying

1. Check registration in `registry.ts`
2. Check variant config in GameConfig
3. Add logging in GameHost constructor
4. Verify variant function returns modified actions

### Actions Not Authorized

1. Check `getValidActions()` includes action
2. Check player index matches action.player field
3. Add logging in `authorizeAndExecute()`
4. Verify GameHost is using composed function

### Replay Not Working

1. Check `initialConfig` is set in GameState
2. Verify all actions are in `actionHistory`
3. Check variant config is in `initialConfig`
4. Test replay function in isolation

### State Out of Sync

1. Verify all state changes go through `executeAction()`
2. Check no direct state mutation
3. Verify `actionHistory` is append-only
4. Test with `replayActions()` to validate

---

## Next Steps

**Implement Missing Features** (see `h-outstanding.md`):
- Auto-execute handler in GameHost
- Capability system for visibility filtering
- Speed mode variant
- Daily challenge variant
- URL encoding updates

**Read Architecture Docs**:
- `docs/remixed-855ccfd5.md` - Full multiplayer architecture
- `docs/composition-refactor-2.md` - Variant system design doc
- `docs/protocol-flows.md` - Message flow documentation

**Explore Codebase**:
- Run `npm run typecheck` - ensure zero errors
- Run `npm test` - see test patterns
- Read existing variants for patterns
- Trace action flow with debugger

---

## Quick Reference

**Composition Point**: `src/server/game/GameHost.ts:72-81`
**Variant Template**: `src/game/variants/tournament.ts` (simple) or `oneHand.ts` (complex)
**Replay Function**: `src/game/core/replay.ts:16`
**Action Execution**: `src/game/core/actions.ts:16`
**Authorization**: `src/game/multiplayer/authorization.ts:57`
**Client Creation**: `src/stores/gameStore.ts:28-35`
**Valid Actions**: `src/game/core/gameEngine.ts:82`

**Pattern**: Pure → Compose → Execute → Subscribe → React
