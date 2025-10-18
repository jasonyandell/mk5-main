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
│  Sees: FilteredGameState only   │
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Client Layer                   │  src/game/multiplayer/
│  NetworkGameClient, protocol    │
│  Caches: FilteredGameState      │
└─────────────────────────────────┘
              ↕ (FilteredGameState via protocol)
┌─────────────────────────────────┐
│  Server Layer                   │  src/server/
│  GameHost, GameRegistry         │
│  **COMPOSITION HAPPENS HERE**   │
│  Has: Full GameState            │
│  Sends: FilteredGameState       │
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

## State Encapsulation & Security

### Client-Server Boundary

**Server-side (GameHost)**:
- Has full `GameState` with all information
- Filters state based on player capabilities before sending
- Returns `FilteredGameState` via `getVisibleStateForSession()`

**Client-side (NetworkGameClient)**:
- Receives `FilteredGameState` (not full GameState)
- Opponent hands filtered to empty arrays
- Cannot access information player shouldn't see
- Type system enforces this at compile time

### FilteredGameState Type

```typescript
type FilteredGameState = Omit<GameState, 'players'> & {
  players: Array<{
    id: number;
    name: string;
    teamId: 0 | 1;
    marks: number;
    hand: Domino[];  // Empty if can't observe
    handCount: number;
    suitAnalysis?: SuitAnalysis;
  }>;
};
```

**Key Properties**:
- Structurally compatible with GameState (core libraries unchanged)
- Filtering based on capabilities (observe-own-hand, observe-all-hands, etc.)
- AI strategies work with FilteredGameState (no changes needed)
- Type-safe: impossible to access unfiltered state at compile time

### Capability System

Players receive different views based on their capabilities:
- `observe-own-hand`: See only your own hand (default for players)
- `observe-all-hands`: See all hands (spectators)
- `observe-full-state`: No filtering (debug/admin)

**Filtering happens in**: `src/game/multiplayer/capabilityUtils.ts:getVisibleStateForSession()`

**Example filtering logic**:
```typescript
// For each player in state
const canSee = hasCapabilityType(session, 'observe-full-state') ||
               canSeeHand(session, playerIndex);

if (canSee) {
  // Return full player data including hand
  return { ...player, handCount: player.hand.length };
} else {
  // Hide hand but keep handCount for UI
  return { ...player, hand: [], handCount: player.hand.length };
}
```

---

## Initialization Flow

### Overview: Two Phases

**Phase 1: Module Load (Synchronous)**
- JavaScript modules load and execute top-level code
- Adapter and client objects created
- Async initialization triggered

**Phase 2: Async Initialization**
- CREATE_GAME message sent to adapter
- GameHost created with variant composition ⭐
- Initial state computed and cached
- Subscriptions established, UI ready

### Detailed Sequence

```
┌─────────────────────────────────────────────────────────┐
│ PHASE 1: MODULE LOAD (Synchronous)                     │
└─────────────────────────────────────────────────────────┘

1. Browser loads src/main.ts
   └─> Imports App.svelte (line 4)
   └─> Imports gameStore (line 6)

2. src/stores/gameStore.ts TOP-LEVEL CODE EXECUTES
   Lines 34-40:
   ├─ const adapter = new InProcessAdapter()           (line 34)
   ├─ const config: GameConfig = { ... }               (lines 35-38)
   └─ gameClient = new NetworkGameClient(adapter, config) (line 40)

3. NetworkGameClient constructor (NetworkGameClient.ts:40-53)
   ├─ Generates sessionId (line 42)
   ├─ Subscribes to adapter messages (line 45)
   └─ Triggers async init: this.initPromise = this.createGame(config) (line 51)
       │
       └─> Returns immediately (async, doesn't block module load)

4. gameStore.ts continues (lines 49-56)
   ├─ Creates clientState writable store (line 49)
   ├─ Subscribes to gameClient updates (lines 52-54)
   └─ Calls setPerspective('player-0') (line 56)

┌─────────────────────────────────────────────────────────┐
│ PHASE 2: ASYNC INITIALIZATION                          │
└─────────────────────────────────────────────────────────┘

5. NetworkGameClient.createGame() (NetworkGameClient.ts:160-188)
   ├─ Sends CREATE_GAME message via adapter.send() (line 162)
   ├─ Polls for gameId to be set (lines 170-183)
   │   └─> Waits for handleServerMessage() to process GAME_CREATED
   └─> initPromise resolves when gameId exists

6. InProcessAdapter.handleCreateGame() (InProcessAdapter.ts:180-218)
   ├─ Calls GameRegistry.createGame(config) (line 187)
   └─> Creates game instance

7. GameRegistry.createGame() (GameHost.ts:469-504)
   ├─ Generates unique gameId (line 470)
   ├─ Creates 4 PlayerSession objects (lines 473-491)
   │   └─ Each has playerId, playerIndex, controlType, capabilities
   └─ Creates GameHost instance (line 493)

8. GameHost constructor (GameHost.ts:53-127) ⭐ COMPOSITION POINT
   ├─ Normalizes variant configs to array (lines 81-86)
   │   • Single variant → [{ type, config }]
   │   • Multiple variants → already an array
   │
   ├─ Composes variants (line 89)
   │   this.getValidActionsComposed = applyVariants(
   │     baseGetValidActions,  // from gameEngine.ts:82
   │     variantConfigs
   │   )
   │
   ├─ Creates initial GameState (line 92)
   │   └─> createInitialState({ playerTypes, shuffleSeed, ... })
   │
   ├─ Converts to FilteredGameState format (lines 101-118)
   │
   ├─ Wraps in MultiplayerGameState (lines 120-123)
   │
   └─ Processes auto-execute actions (line 126)
       └─> Variants can inject scripted actions with autoExecute: true

9. InProcessAdapter sends GAME_CREATED (InProcessAdapter.ts:214-217)
   └─> Broadcasts { type: 'GAME_CREATED', gameId, view }

10. NetworkGameClient.handleServerMessage() (NetworkGameClient.ts:195-198)
    ├─ Sets this.gameId (line 196)
    ├─ Sets this.cachedView (line 197)
    └─ Calls notifyListeners() (line 198)

11. gameStore subscription fires (gameStore.ts:52-54)
    └─> clientState.set(state)

12. Derived stores recompute (gameStore.ts:59, 62, 105)
    ├─ gameState = derived(clientState, ...)
    ├─ playerSessions = derived(clientState, ...)
    └─ viewProjection = derived([gameState, playerSessions, ...], ...)

13. UI Components render
    └─> App.svelte and children react to store changes

┌─────────────────────────────────────────────────────────┐
│ GAME READY                                              │
└─────────────────────────────────────────────────────────┘
```

### Key Composition Code

The **ONLY** place variant composition happens:

```typescript
// GameHost.ts:81-89
const variantConfigs: VariantConfig[] = [
  ...(config.variant
    ? [{ type: config.variant.type, ...(config.variant.config ? { config: config.variant.config } : {}) }]
    : []),
  ...(config.variants ?? [])
];

this.getValidActionsComposed = applyVariants(
  baseGetValidActions,  // Pure function from src/game/core/gameEngine.ts:82
  variantConfigs        // Array of variant descriptors
);
```

This composed function is stored and used for ALL `getValidActions()` calls throughout the game.

---

## Client-Side Store Architecture

### Overview

The client uses **Svelte stores** for reactive state management. Data flows from GameHost → NetworkGameClient → Svelte stores → UI components.

### Store Hierarchy

```
gameClient (NetworkGameClient instance)
    ↓ (subscription)
clientState (writable<MultiplayerGameState>)
    ↓ (derived)
    ├─> gameState (derived<FilteredGameState>)
    ├─> playerSessions (derived<PlayerSession[]>)
    └─> viewProjection (derived<ViewProjection>)
            ↓
         UI Components
```

### Core Stores

**1. gameClient** (gameStore.ts:40)
```typescript
const gameClient = new NetworkGameClient(adapter, config);
```
- **Type**: `NetworkGameClient` instance (not a store)
- **Purpose**: Client interface to game server
- **Methods**: `requestAction()`, `setPlayerControl()`, `subscribe()`

**2. clientState** (gameStore.ts:49)
```typescript
export const clientState = writable<MultiplayerGameState>(gameClient.getState());

gameClient.subscribe(state => {
  clientState.set(state);
});
```
- **Type**: `writable<MultiplayerGameState>`
- **Purpose**: Primary store tracking game state + sessions
- **Updates**: Whenever GameHost broadcasts state changes
- **Contains**: `{ state: FilteredGameState, sessions: PlayerSession[] }`

**3. gameState** (gameStore.ts:59)
```typescript
export const gameState = derived(clientState, $clientState => $clientState.state);
```
- **Type**: `Readable<FilteredGameState>`
- **Purpose**: Extract just the game state (no sessions)
- **Used by**: UI components that don't need session info
- **Security**: Already filtered by GameHost based on capabilities

**4. playerSessions** (gameStore.ts:62)
```typescript
export const playerSessions = derived(clientState, $clientState => $clientState.sessions);
```
- **Type**: `Readable<PlayerSession[]>`
- **Purpose**: Track player control types and capabilities
- **Contains**: `[{ playerId, playerIndex, controlType, capabilities }, ...]`

**5. viewProjection** (gameStore.ts:105-140)
```typescript
export const viewProjection = derived(
  [gameState, playerSessions, currentSessionId],
  ([$gameState, $sessions, $sessionId]) => {
    // Get current session
    const session = $sessions.find(s => s.playerId === $sessionId) ?? $sessions[0];

    // Get valid actions from client
    const allowedActions = gameClient.getValidActions(session?.playerIndex);

    // Get transitions with labels
    const allTransitions = getNextStates($gameState);

    // Create view projection with UI helpers
    return createViewProjection($gameState, transitions, {
      isTestMode: testMode,
      viewingPlayerIndex: session?.playerIndex,
      canAct: hasCapabilityType(session, 'act-as-player'),
      isAIControlled: (player) => $sessions.find(s => s.playerIndex === player)?.controlType === 'ai'
    });
  }
);
```
- **Type**: `Readable<ViewProjection>`
- **Purpose**: Combine state + actions + UI metadata for components
- **Contains**:
  - `dominoes` - Hand with play metadata
  - `actionsByPhase` - Grouped actions for UI
  - `gameInfo` - Score, phase, trump, etc.
  - `ui` - Active view, button states, etc.
- **Used by**: Almost all UI components

### Perspective Management

Users can switch perspective to see different player views:

```typescript
// gameStore.ts:81-88
export async function setPerspective(sessionId: string): Promise<void> {
  currentSessionIdStore.set(sessionId);
  await gameClient.setPlayerId(sessionId);
}
```

**Flow:**
1. User selects perspective from dropdown
2. `setPerspective('player-2')` called
3. GameClient subscribes to that player's view
4. GameHost filters state for that player's capabilities
5. clientState updates with new filtered view
6. UI shows player-2's perspective (their hand, their valid actions)

### Subscription Chain

```
┌──────────────────────────────────────────────────────────┐
│ Server Side (GameHost)                                   │
│                                                           │
│ Action executed → state changed → notifyListeners()      │
└──────────────────────────────────────────────────────────┘
                           ↓
              (filtered GameView for each player)
                           ↓
┌──────────────────────────────────────────────────────────┐
│ Client Side (NetworkGameClient)                          │
│                                                           │
│ handleServerMessage() → cachedView = view                │
│                      → notifyListeners()                 │
└──────────────────────────────────────────────────────────┘
                           ↓
              (MultiplayerGameState)
                           ↓
┌──────────────────────────────────────────────────────────┐
│ Svelte Stores                                            │
│                                                           │
│ clientState.set(state)                                   │
│      ↓                                                    │
│ gameState recomputes (derived)                           │
│ playerSessions recomputes (derived)                      │
│ viewProjection recomputes (derived)                      │
└──────────────────────────────────────────────────────────┘
                           ↓
              (Components subscribe)
                           ↓
┌──────────────────────────────────────────────────────────┐
│ UI Components                                            │
│                                                           │
│ $viewProjection, $gameState reactively update            │
│ DOM re-renders with new state                            │
└──────────────────────────────────────────────────────────┘
```

### Example: UI Component Usage

```svelte
<script>
  import { gameState, viewProjection } from '../stores/gameStore';
</script>

<!-- Direct state access -->
<div>Phase: {$gameState.phase}</div>
<div>Dealer: Player {$gameState.dealer + 1}</div>

<!-- View projection helpers -->
{#each $viewProjection.dominoes as domino}
  <DominoCard
    {domino}
    canPlay={domino.canPlay}
    isRecommended={domino.recommended}
  />
{/each}
```

**Key Points:**
- `$gameState` - Direct access to FilteredGameState
- `$viewProjection` - UI-friendly helpers and metadata
- Reactive: Components update automatically when stores change
- No manual subscription management needed (Svelte handles it)

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
- **game/GameHost.ts:53-127**: Constructor with variant composition ⭐
- **game/GameHost.ts:81-89**: Variant composition code
- **game/GameHost.ts:249-333**: `createView()` uses composed actions
- **game/GameHost.ts:469-504**: `GameRegistry.createGame()` - creates instances
- **offline/InProcessAdapter.ts**: In-memory game adapter
- **offline/InProcessAdapter.ts:180-218**: Handles CREATE_GAME message

### Multiplayer (src/game/multiplayer/)

- **NetworkGameClient.ts**: Client-side game interface (caches FilteredGameState)
- **NetworkGameClient.ts:40-53**: Constructor, triggers async init
- **NetworkGameClient.ts:160-188**: `createGame()` - sends CREATE_GAME message
- **NetworkGameClient.ts:193-238**: `handleServerMessage()` - processes server messages
- **authorization.ts**: `authorizeAndExecute()` - validate actions
- **capabilityUtils.ts**: State filtering logic, `getVisibleStateForSession()`
- **types.ts**: `MultiplayerGameState`, `PlayerSession`, `Capability` types

### Stores (src/stores/)

- **gameStore.ts**: Svelte reactive stores, client creation
- **gameStore.ts:34-40**: Initial client setup (adapter + NetworkGameClient)
- **gameStore.ts:49-54**: clientState writable + subscription
- **gameStore.ts:59**: gameState derived store
- **gameStore.ts:62**: playerSessions derived store
- **gameStore.ts:105-140**: viewProjection derived store
- **gameStore.ts:180-196**: executeAction() - send actions to server
- **gameStore.ts:272-307**: `startOneHand()` - variant creation example

### Protocol & Shared Types

- **game/types.ts**: `GameState`, `FilteredGameState`, `GameAction` core types
- **game/types/config.ts**: `GameConfig`, `VariantConfig`, `GameVariant`
- **shared/multiplayer/protocol.ts**: Message envelopes (`GameView`, `ValidAction`, `PlayerInfo`)
  - `GameView.state` contains `FilteredGameState` (capability-filtered)
  - Protocol layer enforces client never receives unfiltered state

---

## Action Flow

### User Clicks Action

```
1. UI Button Click
   └─> gameActions.executeAction(transition)

2. src/stores/gameStore.ts:180-196
   └─> gameClient.requestAction(playerId, action)

3. src/game/multiplayer/NetworkGameClient.ts:73-98
   └─> adapter.send({ type: 'EXECUTE_ACTION', gameId, playerId, action })

4. src/server/offline/InProcessAdapter.ts:223-241
   └─> GameHost.executeAction(playerId, action)

5. src/server/game/GameHost.ts:140-165
   └─> authorizeAndExecute(mpState, request, getValidActionsComposed)

6. src/game/multiplayer/authorization.ts
   ├─> Check: Is action in getValidActions()?
   ├─> Check: Is player authorized?
   └─> Execute: executeAction(state, action)

7. src/game/core/actions.ts:16
   └─> Pure state transition, returns new state

8. GameHost.notifyListeners() (GameHost.ts:449-457)
   ├─> Creates filtered view for each listener
   └─> Calls listener(view) for all subscribers

9. NetworkGameClient.handleServerMessage() (NetworkGameClient.ts:205-209)
   ├─> Updates cachedView
   └─> Calls notifyListeners()

10. gameStore subscription fires (gameStore.ts:52-54)
    └─> clientState.set(state)

11. Derived stores recompute
    └─> viewProjection, gameState update

12. UI Re-renders (Svelte reactivity)
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
import type { GameConfig } from '../game/types/config';

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

**Composition Point**: `src/server/game/GameHost.ts:81-89` ⭐
**State Filtering**: `src/game/multiplayer/capabilityUtils.ts:getVisibleStateForSession()`
**Variant Template**: `src/game/variants/tournament.ts` (simple) or `oneHand.ts` (complex)
**Replay Function**: `src/game/core/replay.ts:16`
**Action Execution**: `src/game/core/actions.ts:16`
**Authorization**: `src/game/multiplayer/authorization.ts`
**Client Creation**: `src/stores/gameStore.ts:34-40`
**Client State**: `src/stores/gameStore.ts:49-54` (writable + subscription)
**Derived Stores**: `src/stores/gameStore.ts:59, 62, 105-140`
**Valid Actions**: `src/game/core/gameEngine.ts:82`
**GameHost Constructor**: `src/server/game/GameHost.ts:53-127`
**Async Init**: `src/game/multiplayer/NetworkGameClient.ts:160-188`

**Pattern**: Pure → Compose → Execute → Filter → Subscribe → React
