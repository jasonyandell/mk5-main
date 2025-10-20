# Game Onboarding - Texas 42 Architecture

**Purpose**: Get new developers productive quickly with the Texas 42 codebase architecture.

**Last Updated**: 2025-01-18 (Post-Architecture Alignment with remixed-855ccfd5.md)

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
│  Caches: MultiplayerGameState + │
│          per-seat ValidAction[] │
│  Trust: Server-filtered actions │
└─────────────────────────────────┘
              ↕ (GameView via protocol)
┌─────────────────────────────────┐
│  Server Layer (Authority)       │  src/server/
│  GameHost - created via factory │
│  **COMPOSITION HAPPENS HERE**   │
│  Stores: Pure GameState         │
│  Filters: On-demand per-client  │
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
- Stores **pure GameState** with all information (authoritative truth)
- Filters state **on-demand** per client request
- Returns **FilteredGameState** via `getVisibleStateForSession()`
- Produces per-session `ValidAction[]` via the capability system
- Sends snapshots shaped like `MultiplayerGameState`, but with **already-filtered** `coreState` values tailored to the subscribing session

**Client-side (NetworkGameClient)**:
- Receives **host-filtered** `MultiplayerGameState` snapshots plus seat-specific `ValidAction[]`
- Caches both for synchronous reads; exposes async `getState()` / `getActions()` APIs
- Never reconstructs hidden information—whatever the host removed remains absent
- Opponent hands remain filtered in the delivered `FilteredGameState` views
- **Trusts server** – no client-side filtering logic, simply hydrates what the authority supplied

### MultiplayerGameState (Authority Storage)

```typescript
// src/game/multiplayer/types.ts:66-73
interface MultiplayerGameState {
  gameId: string;                    // Unique game identifier
  coreState: GameState;              // Pure, unfiltered state
  players: readonly PlayerSession[]; // Immutable player sessions
  createdAt: number;                 // Creation timestamp
  lastActionAt: number;              // Last activity timestamp
  enabledVariants: VariantConfig[];  // Active rule modifications
}
```

> ℹ️ **Host vs. client snapshots**: On the authority, `coreState` truly is the unfiltered `GameState`. When a snapshot is serialized for a specific client, GameHost redacts any data that viewer cannot see (for example, other players' hands) but leaves the shape as `MultiplayerGameState`. NetworkGameClient caches that redacted snapshot exactly as delivered—it never attempts to reconstruct hidden fields.

**Key Change from Spike**: Authority now stores pure `GameState` instead of pre-filtered `FilteredGameState`. Filtering happens on-demand in `createView()`.

### FilteredGameState Type

```typescript
type FilteredGameState = Omit<GameState, 'players'> & {
  players: Array<{
    id: number;
    name: string;
    teamId: 0 | 1;
    marks: number;
    hand: Domino[];      // Empty if can't observe
    handCount: number;   // Always present for UI
    suitAnalysis?: SuitAnalysis;
  }>;
};
```

**Properties**:
- Structurally compatible with GameState (core libraries unchanged)
- Filtering based on capabilities (observe-own-hand, observe-all-hands, etc.)
- AI strategies work with FilteredGameState (no changes needed)
- Type-safe: impossible to access unfiltered state at compile time

### Capability System

Players receive different views based on their capabilities:

```typescript
type Capability =
  | { type: 'act-as-player'; playerIndex: number }  // Can execute actions for this player
  | { type: 'observe-own-hand' }                    // See only your hand
  | { type: 'observe-all-hands' }                   // See all hands (spectators)
  | { type: 'observe-full-state' }                  // No filtering (debug/admin)
  | { type: 'see-hints' }                           // See hint metadata
  | { type: 'see-ai-intent' }                       // See AI reasoning
  | { type: 'replace-ai' }                          // Can hot-swap with AI
  | { type: 'configure-variant' }                   // Can change variants
  | { type: 'undo-actions' };                       // Can undo moves
```

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

**Authorization**: `canPlayerExecuteAction()` checks `act-as-player` capability before allowing actions.

### Standard Capability Builders (Vision §4.3)

The codebase provides standard builders for common player types:

**File**: `src/game/multiplayer/capabilities.ts`

```typescript
// Human player
humanCapabilities(playerIndex) → [
  { type: 'act-as-player', playerIndex },
  { type: 'observe-own-hand' }
]

// AI player (can be replaced)
aiCapabilities(playerIndex) → [
  { type: 'act-as-player', playerIndex },
  { type: 'observe-own-hand' },
  { type: 'replace-ai' }
]

// Spectator (watch only)
spectatorCapabilities() → [
  { type: 'observe-all-hands' },
  { type: 'observe-full-state' }
]

// Coach (see student's hand + hints)
coachCapabilities(studentIndex) → [
  { type: 'observe-hand', playerIndex: studentIndex },
  { type: 'see-hints' }
]

// Tutorial student (hints + undo)
tutorialCapabilities(playerIndex) → [
  { type: 'act-as-player', playerIndex },
  { type: 'observe-own-hand' },
  { type: 'see-hints' },
  { type: 'undo-actions' }
]
```

**Fluent Builder API:**
```typescript
import { buildCapabilities } from 'src/game/multiplayer/capabilities';

const customCaps = buildCapabilities()
  .actAsPlayer(0)
  .observeOwnHand()
  .seeHints()
  .build();
```

**Used in**: `GameHost.buildBaseCapabilities()`, `createGameAuthority.generateDefaultSessions()`

---

## Initialization Flow

### Overview: Two Phases

**Phase 1: Module Load (Synchronous)**
- JavaScript modules load and execute top-level code
- Adapter and client objects created
- Async initialization triggered (returns promise immediately)

**Phase 2: Async Initialization**
- CREATE_GAME message sent to adapter
- GameHost created via factory with variant composition ⭐
- Initial state computed and cached
- Promise resolves when GAME_CREATED received
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

3. NetworkGameClient constructor (NetworkGameClient.ts:42-53)
   ├─ Generates sessionId (line 44)
   ├─ Subscribes to adapter messages (lines 47-49)
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

5. NetworkGameClient.createGame() (NetworkGameClient.ts:167-197)
   ├─ Creates promise with resolver stored in field (lines 170-182)
   ├─ Sends CREATE_GAME message via adapter.send() (lines 185-189)
   └─> Waits for promise resolution (no polling, clean async)

6. InProcessAdapter.handleCreateGame() (InProcessAdapter.ts)
   ├─ Generates gameId: `game-${Date.now()}-${random}`
   ├─ Calls createGameAuthority(gameId, config) ⭐ FACTORY PATTERN
   └─> Creates GameHost instance

7. createGameAuthority() (createGameAuthority.ts:14-22)
   ├─ Generates default PlayerSessions if not provided (lines 21)
   │   └─ Each session has: playerId, playerIndex, controlType, capabilities
   └─ Returns new GameHost(gameId, config, sessions) (line 22)

8. GameHost constructor (GameHost.ts:53-113) ⭐ COMPOSITION POINT
   ├─ Normalizes variant configs to array (lines 81-86)
   │
   ├─ Composes variants (line 89)
   │   this.getValidActionsComposed = applyVariants(
   │     baseGetValidActions,  // from gameEngine.ts:82
   │     variantConfigs
   │   )
   │
   ├─ Creates initial pure GameState (line 92)
   │   └─> createInitialState({ playerTypes, shuffleSeed, ... })
   │
   ├─ Stores in MultiplayerGameState (lines 101-109)
   │   {
   │     gameId,
   │     coreState: initialState,  // PURE GameState
   │     players: normalizedPlayers,
   │     createdAt: Date.now(),
   │     lastActionAt: Date.now(),
   │     enabledVariants: variantConfigs
   │   }
   │
   └─ Processes auto-execute actions (line 112)
       └─> Variants can inject scripted actions with autoExecute: true

9. InProcessAdapter sends GAME_CREATED (InProcessAdapter.ts)
   └─> Broadcasts { type: 'GAME_CREATED', gameId, view }

10. NetworkGameClient.handleServerMessage() (NetworkGameClient.ts:204-220)
    ├─ Sets this.gameId (line 205)
    ├─ Sets this.cachedView (line 206)
    ├─ **Resolves createGame promise** (lines 209-213) ⭐ NO POLLING
    └─ Calls notifyListeners() (line 215)

11. gameStore subscription fires (gameStore.ts:52-54)
    └─> clientState.set(state)

12. Derived stores recompute (gameStore.ts:60, 70, 113)
    ├─ gameState = derived(clientState, ...)
    ├─ playerSessions = derived(clientState, ...)
    └─ viewProjection = derived([gameState, playerSessions, ...], ...)

13. UI Components render
    └─> App.svelte and children react to store changes

┌─────────────────────────────────────────────────────────┐
│ GAME READY                                              │
└─────────────────────────────────────────────────────────┘
```

### Key Architectural Changes

**1. No GameRegistry** - Replaced with factory pattern:
```typescript
// OLD (GameRegistry pattern)
const registry = new GameRegistry();
const instance = registry.createGame(config);

// NEW (Factory pattern)
import { createGameAuthority } from './createGameAuthority';
const authority = createGameAuthority(gameId, config);
```

**2. No Polling** - Clean promise-based async:
```typescript
// OLD (10ms polling loop)
await new Promise<void>((resolve) => {
  const checkInterval = setInterval(() => {
    if (this.gameId) {
      clearInterval(checkInterval);
      resolve();
    }
  }, 10);
});

// NEW (Promise resolver stored in field)
const promise = new Promise((resolve) => {
  this.createGameResolver = resolve;
});
await promise;
// Resolved in handleServerMessage when GAME_CREATED received
```

**3. Pure State Storage** - Authority stores unfiltered GameState:
```typescript
// OLD (Pre-filtered storage)
this.mpState = {
  state: filteredGameState,  // Already filtered
  sessions: playerSessions
};

// NEW (Pure storage, filter on-demand)
this.mpState = {
  gameId: 'game-123',
  coreState: pureGameState,  // Host-side canonical state (unfiltered)
  players: playerSessions,
  createdAt: Date.now(),
  lastActionAt: Date.now(),
  enabledVariants: []
};
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
    ↓ (async hydrate + subscription)
clientState (writable<MultiplayerGameState>)
    └─> actionsByPlayer (writable<Record<string, ValidAction[]>>)
            ↓
    derived stores
        ├─> gameState (FilteredGameState for current perspective)
        ├─> playerSessions (session list)
        ├─> allowedActionsStore (ValidAction[] for current session)
        └─> viewProjection (ViewProjection for UI)
                ↓
             UI Components
```

### Core Stores

**1. gameClient / networkClient** (gameStore.ts:44-89)
```typescript
const gameClient = new NetworkGameClient(adapter, config);
```
- **Type**: `NetworkGameClient` instance (not a store)
- **Purpose**: Client interface to game server
- **Methods**: `getState()`, `getActions(playerId)`, `executeAction(request)`, `joinGame(session)`, `leaveGame(playerId)`, `setPlayerControl()`, `subscribe()`

**2. clientState & actionsByPlayer** (gameStore.ts:90-124)
```typescript
export const clientState = writable<MultiplayerGameState>(createPendingState());

gameClient.getState()
  .then(state => {
    clientState.set(state);
    actionsByPlayer.set(networkClient.getCachedActionsMap());
  })
  .catch(error => {
    console.error('Failed to obtain initial game state:', error);
  });

unsubscribeFromClient = gameClient.subscribe(state => {
  clientState.set(state);
  actionsByPlayer.set(networkClient.getCachedActionsMap());
});

const actionsByPlayer = writable<Record<string, ValidAction[]>>({});
const allowedActionsStore = derived(
  [actionsByPlayer, currentSessionId],
  ([$actions, $sessionId]) => $actions[$sessionId] ?? $actions['__unfiltered__'] ?? []
);
```
- **Type**: `writable<MultiplayerGameState>` plus a companion `writable<Record<string, ValidAction[]>>`
- **Purpose**: Track authoritative multiplayer state and the host-supplied per-session action lists
- **Hydration**: Starts from a placeholder, then hydrates asynchronously once the authority responds
- **Contains**: `{ gameId, coreState, players, createdAt, lastActionAt, enabledVariants }`
- **Action cache**: Mirrors the host’s filtered `ValidAction[]` map so the UI never refilters client-side

**3. gameState** (gameStore.ts:137-145)
```typescript
export const gameState = derived(
  [clientState, currentSession],
  ([$clientState, $session]) =>
    $session
      ? getVisibleStateForSession($clientState.coreState, $session)
      : convertToFilteredState($clientState.coreState)
);
```
- **Type**: `Readable<FilteredGameState>`
- **Purpose**: Deliver per-view filtered state using the shared capability helper
- **Used by**: UI components that render the board
- **Security**: Host already filtered; store simply mirrors the authoritative projection

**4. playerSessions** (gameStore.ts:70)
```typescript
export const playerSessions = derived(clientState, $clientState =>
  Array.from($clientState.players)
);
```
- **Type**: `Readable<PlayerSession[]>`
- **Purpose**: Track player control types and capabilities
- **Contains**: `[{ playerId, playerIndex, controlType, capabilities }, ...]`

**5. viewProjection** (gameStore.ts:113-145)
```typescript
export const viewProjection = derived(
  [gameState, playerSessions, currentSessionId],
  ([$gameState, $sessions, $sessionId]) => {
    const session = $sessions.find(s => s.playerId === $sessionId) ?? $sessions[0];
    const canAct = !!(session && hasCapabilityType(session, 'act-as-player'));

    // Get valid actions from client (server already filtered)
    const allowedActions = gameClient.getValidActions();

    // Get transitions with labels
    const allTransitions = getNextStates($gameState);
    const usedTransitions = testMode
      ? allTransitions
      : allTransitions.filter(t => allowedKeys.has(actionKey(t.action)));

    // Create view projection with UI helpers
    return createViewProjection($gameState, transitions, {
      isTestMode: testMode,
      viewingPlayerIndex: session?.playerIndex,
      canAct,
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
- **Key Change**: Client trusts `gameClient.getValidActions()` - no filtering

### Perspective Management

Users can switch perspective to see different player views:

```typescript
// gameStore.ts:89-95
export async function setPerspective(sessionId: string): Promise<void> {
  const current = get(currentSessionIdStore);
  if (current !== sessionId) {
    currentSessionIdStore.set(sessionId);
  }
  await (gameClient as NetworkGameClient).setPlayerId(sessionId);
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
│   ↓                                                       │
│ createView(playerId) for each subscriber                │
│   ↓                                                       │
│ getVisibleStateForSession(coreState, session)           │
│ filterActionsForSession(session, allActions)            │
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
// Composition (src/game/variants/registry.ts:28-35)
function applyVariants(base: StateMachine, variants: VariantConfig[]): StateMachine {
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

**File**: `src/game/variants/tournament.ts`

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

- **server/game/GameHost.ts**: Game authority, composes variant-aware state machine and routes all mutations
  - `constructor` - builds canonical multiplayer state via `createMultiplayerGame()` and composes variants
  - `executeAction()` - requires timestamped requests, delegates to `authorizeAndExecute()`
  - `createView()` - filters `coreState` on demand
- **server/game/createGameAuthority.ts**: Factory for creating `GameHost` instances ⭐
- **server/offline/InProcessAdapter.ts**: In-memory adapter that forwards protocol messages to `GameHost`

### Multiplayer (src/game/multiplayer/)

- **capabilities.ts**: Standard capability builders per Vision §4.3 ⭐
  - `humanCapabilities()`, `aiCapabilities()`, `spectatorCapabilities()`
  - `coachCapabilities()`, `tutorialCapabilities()`
  - `CapabilityBuilder` - Fluent API for custom sets
- **NetworkGameClient.ts**: Protocol-speaking client that sends timestamped `EXECUTE_ACTION` messages and caches filtered views (core state remains canonical on the host)
- **authorization.ts**: `authorizeAndExecute()` first verifies an action exists in the composed variant state machine, then re-validates against capability-filtered visibility ⭐
  - `canPlayerExecuteAction()` - Checks if session can execute specific action
  - `getValidActionsForPlayer()` - Pure function (Vision §3.2.3) that composes variants and filters by capabilities ⭐
- **stateLifecycle.ts**: Pure helpers for creating/updating multiplayer state (`createMultiplayerGame`, `addPlayer`, `updatePlayerSession`, `removePlayer`)
- **capabilityUtils.ts**: Filters actions/state per session capability set
- **types.ts**: Canonical multiplayer data structures (`MultiplayerGameState`, `ActionRequest` with timestamp, `Result<T>` with `success` field)

### Stores (src/stores/)

- **gameStore.ts**: Svelte reactive stores, client creation
- **gameStore.ts:34-40**: Initial client setup (adapter + NetworkGameClient)
- **gameStore.ts:49-54**: clientState writable + subscription
- **gameStore.ts:60-67**: Derived `gameState` currently rewraps `coreState`; server now provides canonical `coreState`, so client-side conversion will be removed once filtered views ship separately ⭐
- **gameStore.ts:70**: playerSessions derived store
- **gameStore.ts:113-145**: viewProjection derived store
- **gameStore.ts:180-196**: executeAction() - send actions to server

### Protocol & Shared Types

- **game/types.ts**: `GameState`, `FilteredGameState`, `GameAction` core types
- **game/types/config.ts**: `GameConfig`, `VariantConfig`, `GameVariant`
- **shared/multiplayer/protocol.ts**: Message envelopes (`GameView`, `ValidAction`, `PlayerInfo`)
  - `EXECUTE_ACTION` messages include `timestamp` to keep authority time monotonic
  - `GameView.state` contains `FilteredGameState` (capability-filtered)
  - `Result<T>` - Standardized as `{ success: true; value: T } | { success: false; error: string }`
- **game/multiplayer/types.ts**: Multiplayer domain types
  - `ActionRequest` - Requires `{ playerId: string, action, timestamp }`
  - `Result<T>` - Uses same `success` field (aligned with protocol layer)
  - `PlayerSession` - Player identity with `playerId: string` and capabilities

---

## Action Flow

### User Clicks Action

```
1. UI Button Click
   └─> gameActions.executeAction(transition)

2. src/stores/gameStore.ts:180-196
   └─> gameClient.requestAction(playerId: string, action)

3. src/game/multiplayer/NetworkGameClient.ts:79-104
   └─> adapter.send({ type: 'EXECUTE_ACTION', gameId, playerId: string, action, timestamp: Date.now() })
   // Note: playerId is string (e.g., "player-0", "ai-1"), not number

4. src/server/offline/InProcessAdapter.ts
   └─> GameHost.executeAction(playerId, action, timestamp)

5. src/server/game/GameHost.ts:118-184
   └─> authorizeAndExecute(mpState, { playerId, action, timestamp }, getValidActionsComposed) // composed variants

6. src/game/multiplayer/authorization.ts:38-106
   ├─> Find player session by playerId
   ├─> Verify action exists in composed `getValidActions(coreState)` (variants applied)
   ├─> Filter actions through capability pipeline (`filterActionsForSession`)
   └─> Execute: `executeAction(coreState, action)` and update `lastActionAt`

7. src/game/core/actions.ts:16
   └─> Pure state transition, returns new state

8. GameHost.notifyListeners() (GameHost.ts:462-470)
   ├─> Creates filtered view for each listener
   │   └─> getVisibleStateForSession(coreState, session) ⭐
   └─> Calls listener(view) for all subscribers

9. NetworkGameClient.handleServerMessage() (NetworkGameClient.ts:222-226)
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
GameHost.createView() called for specific playerId
  ↓
1. Get all valid actions from composed state machine:
   this.getValidActionsComposed(coreState)
  ↓
   Executes composed variants:
   tournament(oneHand(baseGetValidActions))(coreState)
  ↓
   a. baseGetValidActions(coreState) - returns all actions
   b. oneHand wraps it - injects scripted actions if needed
   c. tournament wraps that - filters special bids
  ↓
2. Filter actions by player session's capabilities:
   filterActionsForSession(session, allActions)
  ↓
   Checks each action's metadata for requiredCapabilities
   Filters based on player field + act-as-player capability
  ↓
3. Return GameView with:
   - state: getVisibleStateForSession(coreState, session)
   - validActions: filtered list (server-side authority)
   - players: session info
  ↓
Client receives GameView
  ↓
Client trusts validActions list (no client-side filtering)
  ↓
UI displays only those actions
```

---

## Architecture Decisions

### Why Store Pure State in Authority?

**Alternative**: Store pre-filtered `FilteredGameState` per player

**Problem**:
- Multiple copies of state (one per player view)
- State updates require filtering N times
- Hard to debug (which filtered version is correct?)

**Solution**: Store single pure `GameState`, filter on-demand per request

**Benefits**:
- Single source of truth
- Filtering logic centralized in one place
- Easy to debug (check pure state)
- Scales to any number of viewers

### Why Trust Server on Client?

**Alternative**: Client filters `validActions` by player index

**Problem**:
- Duplicates logic between client and server
- Client must understand authorization rules
- Can diverge from server's filtering

**Solution**: Client trusts `GameView.validActions` from server

**Benefits**:
- Single source of truth (server)
- Client stays dumb (no game logic)
- Easier to maintain (change in one place)
- Enables complex filtering (capabilities, variants) without client changes

### Why Factory Pattern vs Registry?

**Alternative**: `GameRegistry` class with `createGame()` method

**Problem**:
- Unnecessary indirection
- Unclear ownership (registry manages instances)
- More stateful code

**Solution**: `createGameAuthority(gameId, config)` factory function

**Benefits**:
- Direct, clear
- No registry state to manage
- Easier to test (just call function)
- Adapters own their GameHost instances

### Why No Polling?

**Alternative**: `setInterval()` polling for `gameId` to be set

**Problem**:
- Fragile timing
- Wasted CPU cycles
- Hard to reason about timing
- 10ms arbitrary constant

**Solution**: Promise resolver stored in field, resolved in message handler

**Benefits**:
- Clean async semantics
- No wasted work
- Immediate resolution when ready
- Easy to add timeout

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
console.log('Actions before:', baseGetValidActions(coreState).length);
console.log('Actions after:', this.getValidActionsComposed(coreState).length);
```

### Debug Capability Authorization

```typescript
// In authorization.ts:canPlayerExecuteAction
console.log('Session:', session);
console.log('Action:', action);
console.log('Has capability:', hasCapability(session, {
  type: 'act-as-player',
  playerIndex: action.player
}));
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

## Troubleshooting

### Variant Not Applying

1. Check registration in `registry.ts`
2. Check variant config in GameConfig
3. Add logging in GameHost constructor
4. Verify variant function returns modified actions

### Actions Not Authorized

1. Check `getValidActionsComposed(coreState)` includes action
2. Check session has `act-as-player` capability for action.player
3. Add logging in `canPlayerExecuteAction()`
4. Verify GameHost is using composed function

### State Filtering Issues

1. Check `getVisibleStateForSession()` is called in `createView()`
2. Verify session capabilities are set correctly
3. Test filtering function in isolation
4. Check `FilteredGameState` structure matches expectations

### Client Not Getting Actions

1. Verify `GameHost` is broadcasting the `actions` map alongside each `STATE_UPDATE`
2. Check `NetworkGameClient.getActions(playerId)` returns the cached list (state must be hydrated first)
3. Ensure `actionsByPlayer` / `allowedActionsStore` wiring in `gameStore.ts` is updating
4. Confirm the subscription from `NetworkGameClient.subscribe()` is still active

---

## Next Steps

**Read Architecture Docs**:
- `docs/remixed-855ccfd5.md` - Canonical multiplayer architecture vision
- `docs/handoff.md` - Architecture alignment refactoring notes
- `docs/rules.md` - Official Texas 42 rules

**Explore Codebase**:
- Run `npm run typecheck` - ensure zero errors
- Run `npm test` - see test patterns
- Read existing variants for patterns
- Trace action flow with debugger

---

## Quick Reference

**Factory Function**: `src/server/game/createGameAuthority.ts` ⭐
**Composition Point**: `src/server/game/GameHost.ts:81-89` ⭐
**State Filtering**: `src/game/multiplayer/capabilityUtils.ts:getVisibleStateForSession()` ⭐
**Capability Auth**: `src/game/multiplayer/authorization.ts:canPlayerExecuteAction()` ⭐
**Standard Capabilities**: `src/game/multiplayer/capabilities.ts` ⭐
**Get Valid Actions (Pure)**: `src/game/multiplayer/authorization.ts:getValidActionsForPlayer()` ⭐
**Client Init**: `src/stores/gameStore.ts:44-124` (NetworkGameClient setup + async hydration) ⭐
**Client Actions**: `src/game/multiplayer/NetworkGameClient.ts:60-190` (async API, per-seat caches) ⭐
**Host Subscriptions**: `src/server/game/GameHost.ts:207-262` (HostViewUpdate payloads) ⭐
**Action Cache Store**: `src/stores/gameStore.ts:118-124` (per-session ValidAction map) ⭐
**Variant Template**: `src/game/variants/tournament.ts` (simple) or `oneHand.ts` (complex)
**Replay Function**: `src/game/core/replay.ts:16`
**Action Execution**: `src/game/core/actions.ts:16`
**Client Creation**: `src/stores/gameStore.ts:34-40`
**Client State**: `src/stores/gameStore.ts:49-54` (writable + subscription)
**Derived Stores**: `src/stores/gameStore.ts:60, 70, 113-145`

**Pattern**: Pure → Compose → Execute → Filter → Trust → React

**Key Architectural Invariants**:
- Authority stores pure `GameState`, filters on-demand
- Client trusts server's `validActions`, never filters
- Capabilities determine visibility and authorization
- No polling - clean promise-based async
- Factory pattern, not registry pattern
