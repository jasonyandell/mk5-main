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

## How Everything Connects

The architecture composes in layers from pure utilities up through multiplayer:

```
Core Engine (pure utilities)
    ↓
Layer System (13 threaded rules via GameRules interface)
    ↓
Variant System (transforms actions from layered state machine)
    ↓
GameHost (composes layers + variants, stores pure state)
    ↓
NetworkGameClient (caches filtered views, trusts server)
    ↓
Svelte Stores (reactive UI state)
    ↓
UI Components
```

### Complete Request/Response Flow

```
User clicks button
    ↓
executeAction({ playerId, action, timestamp })
    ↓
NetworkGameClient.executeAction() → adapter.send(EXECUTE_ACTION)
    ↓
InProcessAdapter → GameHost.executeAction()
    ↓
authorizeAndExecute() checks capabilities & validates action
    ↓
executeAction(coreState, action, rules) → new coreState
    ↓
GameHost.notifyListeners() → createView() for each subscriber
    ↓
getVisibleStateForSession(coreState, session) → FilteredGameState
filterActionsForSession(session, allActions) → ValidAction[]
    ↓
adapter.send(STATE_UPDATE) → { state, validActions }
    ↓
NetworkGameClient caches state + actions map
    ↓
clientState.set() + actionsByPlayer.set()
    ↓
Derived stores (gameState, viewProjection) recompute
    ↓
UI components reactively update
```

### Key Composition Points

**1. Single Composition Point (GameHost constructor)**:
- The ONLY place layers thread through base state machine and variants wrap the result
- Creates `getValidActionsComposed` used for ALL action generation throughout the game
- Two-level composition:
  1. **Layers** provide the rules (13 GameRules methods) using reduce pattern
  2. **Variants** transform the actions produced by those layered rules
- Result: `applyVariants(getValidActions(state, layers, rules), variantConfigs)`
- Ensures variants see layer-modified actions (e.g., nello bids if layer enabled)

**2. Trust Boundary (GameHost ↔ NetworkGameClient)**:
- **Server (GameHost)**: Authority stores pure GameState, filters on-demand per request
- **Client (NetworkGameClient)**: Trusts server-filtered data, never reconstructs hidden info
- **Protocol**: Host sends `{ state: FilteredGameState, validActions: ValidAction[] }`
- **Security**: Client receives already-filtered data shaped for that session's capabilities

**3. Perspective Management**:
- Client can switch which player's view to display: `setPerspective('player-2')`
- Server refilters state for new perspective's capabilities
- Client caches the new filtered view
- UI shows new player's hand and valid actions

See [State Encapsulation & Security](#state-encapsulation--security) for filtering details.
See [Action Flow](#action-flow) for complete execution path.
See [Client-Side Store Architecture](#client-side-store-architecture) for reactive state management.

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
│  **LAYER+VARIANT COMP HERE**    │
│  Threads layers + applies vars  │
│  Stores: Pure GameState         │
│  Filters: On-demand per-client  │
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Variant Layer                  │  src/game/variants/
│  Pure function transformers     │
│  Filters/annotates layer acts   │
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Layer System                   │  src/game/layers/
│  13 threaded rules (GameRules)  │
│  Base + special contracts       │
│  Composes via reduce pattern    │
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Core Engine                    │  src/game/core/
│  Pure utilities, zero coupling  │
└─────────────────────────────────┘
```

---

## Layer System Architecture

### Overview

The layer system implements **parametric polymorphism** - executors delegate ALL game decisions to injected rules. This enables special contracts (nello, splash, plunge, sevens) to be added without modifying core code.

**Key Principle**: Core knows nothing about special contracts. All variant behavior lives in layers.

### The GameRules Interface (13 Methods)

**File**: `src/game/layers/types.ts`

```typescript
export interface GameRules {
  // ============================================
  // WHO RULES: Determine which player acts (3)
  // ============================================
  getTrumpSelector(state: GameState, winningBid: Bid): number;
  getFirstLeader(state: GameState, trumpSelector: number, trump: TrumpSelection): number;
  getNextPlayer(state: GameState, currentPlayer: number): number;

  // ============================================
  // WHEN RULES: Determine timing (2)
  // ============================================
  isTrickComplete(state: GameState): boolean;
  checkHandOutcome(state: GameState): HandOutcome | null;

  // ============================================
  // HOW RULES: Determine mechanics (2)
  // ============================================
  getLedSuit(state: GameState, domino: Domino): LedSuit;
  calculateTrickWinner(state: GameState, trick: Play[]): number;

  // ============================================
  // VALIDATION RULES: Determine legality (3)
  // ============================================
  isValidPlay(state: GameState, domino: Domino, playerId: number): boolean;
  getValidPlays(state: GameState, playerId: number): Domino[];
  isValidBid(state: GameState, bid: Bid, playerHand?: Domino[]): boolean;

  // ============================================
  // SCORING RULES: Determine outcomes (3)
  // ============================================
  getBidComparisonValue(bid: Bid): number;
  isValidTrump(trump: TrumpSelection): boolean;
  calculateScore(state: GameState): [number, number];
}
```

### Layer Composition Pattern

Layers override only the rules that differ from the base game:

```typescript
// Base layer - standard Texas 42
const baseLayer: GameLayer = {
  name: 'base',
  rules: {
    isTrickComplete: (state) => state.currentTrick.length === 4,
    isValidBid: (state, bid, playerHand) => {
      // Only POINTS and MARKS bids allowed
      if (bid.type === 'NELLO' || bid.type === 'SPLASH' || bid.type === 'PLUNGE') {
        return false;
      }
      return /* validation logic */;
    },
    calculateScore: (state) => /* standard scoring */
    // ... all 13 rules implemented
  }
};

// Nello layer - overrides specific rules
const nelloLayer: GameLayer = {
  name: 'nello',
  rules: {
    // Partner sits out - only 3 plays per trick
    isTrickComplete: (state, prev) =>
      state.trump.type === 'nello' ? state.currentTrick.length === 3 : prev,

    // Allow NELLO bids
    isValidBid: (state, bid, playerHand, prev) => {
      if (bid.type === 'NELLO') {
        return /* nello validation */;
      }
      return prev; // delegate to other layers/base
    },

    // Nello scoring - bidder must take 0 tricks
    calculateScore: (state, prev) => {
      if (state.currentBid.type !== 'NELLO') return prev;
      const biddingTeamTricks = /* count tricks */;
      return biddingTeamTricks === 0
        ? /* award marks to bidder */
        : /* award marks to opponents */;
    }
  }
};

// Composition via reduce
const rules = composeRules([baseLayer, nelloLayer, splashLayer, plungeLayer]);
// Later layers override earlier ones
```

### How It Works

**1. Composition (src/game/layers/compose.ts)**:
```typescript
export function composeRules(layers: GameLayer[]): GameRules {
  return {
    isTrickComplete: (state) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.isTrickComplete?.(state, prev) ?? prev,
        isTrickCompleteBase(state)  // Base implementation
      ),
    // ... repeat for all 13 rules
  };
}
```

**2. Threading through executors**:
```typescript
// gameEngine.ts - passes rules to action generation
export function getValidActions(
  state: GameState,
  layers?: GameLayer[],
  rules?: GameRules
): GameAction[] {
  const threadedRules = rules || composeRules(layers || [baseLayer]);

  // Use threaded rules for validation
  const validPlays = threadedRules.getValidPlays(state, currentPlayer);
  // ...
}

// actions.ts - uses rules for execution
export function executeAction(
  state: GameState,
  action: GameAction,
  rules: GameRules = defaultRules
): GameState {
  // Use threaded rules
  const newMarks = rules.calculateScore(state);
  // ...
}
```

**3. GameHost integration (src/server/game/GameHost.ts)**:
```typescript
constructor(gameId, config, sessions) {
  // Compose layers based on config
  this.layers = [baseLayer, ...getEnabledLayers(config)];
  this.rules = composeRules(this.layers);

  // Thread layers through base state machine
  const baseWithLayers = (state) =>
    getValidActions(state, this.layers, this.rules);

  // Apply variants on top of layered actions
  this.getValidActionsComposed = applyVariants(
    baseWithLayers,
    variantConfigs
  );
}
```

### Available Layers

**Base Layer** (`src/game/layers/base.ts`):
- Standard Texas 42 rules
- POINTS and MARKS bids only
- Filters out special contracts (NELLO/SPLASH/PLUNGE)
- 4 players, 4 plays per trick

**Nello Layer** (`src/game/layers/nello.ts`):
- Allows NELLO bids (1-4 marks)
- Partner sits out (3 plays per trick)
- Bidder must take 0 tricks to win
- Overrides: isTrickComplete, getNextPlayer, isValidBid, calculateScore

**Splash Layer** (`src/game/layers/splash.ts`):
- Allows SPLASH bids (2-3 marks, requires 3+ doubles)
- Partner selects trump
- Bidding team must take all 7 tricks
- Overrides: getTrumpSelector, checkHandOutcome, isValidBid, calculateScore

**Plunge Layer** (`src/game/layers/plunge.ts`):
- Allows PLUNGE bids (4+ marks, requires 4+ doubles)
- Partner selects trump
- Bidding team must take all 7 tricks
- Overrides: getTrumpSelector, checkHandOutcome, isValidBid, calculateScore

**Sevens Layer** (`src/game/layers/sevens.ts`):
- Allows 'sevens' trump selection for MARKS bids
- No follow-suit requirement (any domino can be played)
- Closest to 7 total pips wins trick
- Bidding team must take all 7 tricks
- Overrides: calculateTrickWinner, checkHandOutcome, isValidPlay, getValidPlays, isValidTrump, calculateScore

### Layer Isolation Pattern

The layer system keeps core game logic completely pure by moving all variant-specific behavior into composable layers. Each layer can override or extend game logic without contaminating the core.

**Core Game State** (`src/game/core/`)
- Contains only standard Texas 42 rules (points and marks contracts)
- No knowledge of special contracts (nello, splash, plunge, etc.)
- Pure functions with zero variant-specific references

**Base Layer** (`src/game/layers/base.ts`)
- Wraps core game logic in the layer interface
- Implements standard contract behavior (points/marks)
- Provides default implementations for all game functions

```typescript
// Base layer handles standard contracts
getBidComparisonValue: (bid) => {
  switch (bid.type) {
    case BID_TYPES.POINTS: return bid.value;
    case BID_TYPES.MARKS: return bid.value * 42;
    default: return 0;  // Unknown contracts have no value
  }
}
```

**Variant Layers** (`src/game/layers/nello.ts`, `splash.ts`, etc.)
- Each layer adds support for one special contract type
- Delegates to previous layer for non-matching contracts
- Composes cleanly without core modifications

```typescript
// Nello layer extends base behavior
getBidComparisonValue: (bid, prev) => {
  if (bid.type === BID_TYPES.NELLO) return bid.value! * 42;
  return prev;  // Delegate to base layer
}
```

**Composition**
Layers compose from most specific to least specific:
```typescript
const gameLogic = composeGameLogic([nelloLayer, splashLayer, baseLayer]);
```

This architecture ensures core files remain pure, variants stay isolated, and new contracts can be added without touching existing code.

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
  enabledLayers?: string[];          // Active layer names
}
```

> ℹ️ **Host vs. client snapshots**: On the authority, `coreState` truly is the unfiltered `GameState`. When a snapshot is serialized for a specific client, GameHost redacts any data that viewer cannot see (for example, other players' hands) but leaves the shape as `MultiplayerGameState`. NetworkGameClient caches that redacted snapshot exactly as delivered—it never attempts to reconstruct hidden fields.

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
  | { type: 'observe-hand'; playerIndex: number }   // See specific player's hand
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

### Standard Capability Builders

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
- GameHost created via factory with variant composition
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
   ├─ Calls createGameAuthority(gameId, config)
   └─> Creates GameHost instance

7. createGameAuthority() (createGameAuthority.ts:14-22)
   ├─ Generates default PlayerSessions if not provided (lines 21)
   │   └─ Each session has: playerId, playerIndex, controlType, capabilities
   └─ Returns new GameHost(gameId, config, sessions) (line 22)

8. GameHost constructor (GameHost.ts:53-113)
   ├─ Normalizes variant configs to array (lines 81-86)
   │
   ├─ **Composes LAYERS**
   │   const enabledLayers = getEnabledLayers(config);  // Based on config
   │   this.layers = [baseLayer, ...enabledLayers];
   │   this.rules = composeRules(this.layers);  // 13 threaded rules
   │
   ├─ **Composes VARIANTS with layers** (line 89-98)
   │   // Thread layers through base state machine
   │   const baseWithLayers = (state: GameState) =>
   │     getValidActions(state, this.layers, this.rules);
   │
   │   // Apply variants on top of layered actions
   │   this.getValidActionsComposed = applyVariants(
   │     baseWithLayers,  // Layers threaded through
   │     variantConfigs
   │   );
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
    ├─ **Resolves createGame promise** (lines 209-213)
    └─ Calls notifyListeners() (line 215)

11. gameStore subscription fires (gameStore.ts:134-137)
    └─> clientState.set(state)

12. Derived stores recompute (gameStore.ts:60, 70, 113)
    ├─ gameState = derived(clientState, ...)
    ├─ playerSessions = derived(clientState, ...)
    └─> viewProjection = derived([gameState, playerSessions, ...], ...)

13. UI Components render
    └─> App.svelte and children react to store changes

┌─────────────────────────────────────────────────────────┐
│ GAME READY                                              │
└─────────────────────────────────────────────────────────┘
```

### Key Composition Code

The **ONLY** place layer + variant composition happens:

```typescript
// GameHost.ts:81-98
// 1. Normalize variant configs
const variantConfigs: VariantConfig[] = [
  ...(config.variant
    ? [{ type: config.variant.type, ...(config.variant.config ? { config: config.variant.config } : {}) }]
    : []),
  ...(config.variants ?? [])
];

// 2. Compose layers based on config
const enabledLayers = getEnabledLayers(config);
this.layers = [baseLayer, ...enabledLayers];
this.rules = composeRules(this.layers);

// 3. Thread layers through base state machine
const baseWithLayers = (state: GameState) =>
  getValidActions(state, this.layers, this.rules);

// 4. Apply variants on top of layered actions
this.getValidActionsComposed = applyVariants(
  baseWithLayers,  // Layers already threaded
  variantConfigs   // Array of variant descriptors
);
```

This composed function is stored and used for ALL `getValidActions()` calls throughout the game.

**Two-level composition**: Layers provide the rules (13 methods), variants transform the actions produced by those rules.

---

## Client-Side Store Architecture

### Overview

The client uses **Svelte stores** for reactive state management. Data flows from GameHost → NetworkGameClient → Svelte stores → UI components.

### Store Hierarchy

```
gameClient (NetworkGameClient instance)
    ↓ (async hydrate + subscription)
clientState (writable<MultiplayerGameState>)
actionsByPlayer (writable<Record<string, ValidAction[]>>)
    ↓
derived stores
    ├─> playerSessions (session list)
    ├─> currentSessionIdStore (writable<string>)
    ├─> currentSessionId (derived from currentSessionIdStore)
    ├─> currentSession (derived from playerSessions + currentSessionId)
    ├─> gameState (FilteredGameState for current perspective)
    ├─> allowedActionsStore (ValidAction[] for current session)
    └─> viewProjection (ViewProjection for UI)
            ↓
         UI Components

Note: currentSessionIdStore, currentSessionId, and currentSession work together:
- currentSessionIdStore is the internal writable store
- currentSessionId is a public derived store (read-only access)
- currentSession is derived from both playerSessions and currentSessionId
```

### Core Stores

**1. gameClient / networkClient** (gameStore.ts:84-85)
```typescript
gameClient = new NetworkGameClient(adapter, config);
let networkClient: NetworkGameClient = gameClient as NetworkGameClient;
```
- **Type**: `NetworkGameClient` instance (not a store)
- **Purpose**: Client interface to game server
- **Methods**: `getState()`, `executeAction(request)`, `setPlayerControl()`, `subscribe()`
- **Lifecycle**: Created at module load, destroyed on page unload

**2. clientState & actionsByPlayer** (gameStore.ts:120-137)
```typescript
export const clientState = writable<MultiplayerGameState>(createPendingState());
const actionsByPlayer = writable<Record<string, ValidAction[]>>({});

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
```
- **Type**: `writable<MultiplayerGameState>` plus a companion `writable<Record<string, ValidAction[]>>`
- **Purpose**: Track authoritative multiplayer state and the host-supplied per-session action lists
- **Hydration**: Starts from a placeholder, then hydrates asynchronously once the authority responds
- **Contains**: `{ gameId, coreState, players, createdAt, lastActionAt, enabledVariants, enabledLayers }`
- **Action cache**: Mirrors the host's filtered `ValidAction[]` map so the UI never refilters client-side

**3. playerSessions** (gameStore.ts:141)
```typescript
export const playerSessions = derived(clientState, $clientState =>
  Array.from($clientState.players)
);
```
- **Type**: `Readable<PlayerSession[]>`
- **Purpose**: Track player control types and capabilities
- **Contains**: `[{ playerId, playerIndex, controlType, capabilities }, ...]`

**4. currentSessionIdStore, currentSessionId, currentSession** (gameStore.ts:143-150)
```typescript
const currentSessionIdStore = writable<string>(DEFAULT_SESSION_ID);
export const currentSessionId = derived(currentSessionIdStore, (value) => value);
export const currentSession = derived(
  [playerSessions, currentSessionId],
  ([$sessions, $sessionId]) => $sessions.find(session => session.playerId === $sessionId)
);
```
- **Type**: Internal writable → public derived → public derived
- **Purpose**: Track which player's perspective is currently active
- **Default**: 'player-0'

**5. gameState** (gameStore.ts:152-160)
```typescript
export const gameState = derived(
  [clientState, currentSession],
  ([$clientState, $session]) => {
    if ($session) {
      return getVisibleStateForSession($clientState.coreState, $session);
    }
    return convertToFilteredState($clientState.coreState);
  }
);
```
- **Type**: `Readable<FilteredGameState>`
- **Purpose**: Deliver per-view filtered state using the shared capability helper
- **Used by**: UI components that render the board
- **Security**: Host already filtered; store simply mirrors the authoritative projection

**6. allowedActionsStore** (gameStore.ts:162-165)
```typescript
const allowedActionsStore = derived(
  [actionsByPlayer, currentSessionId],
  ([$actions, $sessionId]) => $actions[$sessionId] ?? $actions['__unfiltered__'] ?? []
);
```
- **Type**: `Readable<ValidAction[]>`
- **Purpose**: Extract allowed actions for the current session from the actions map

**7. viewProjection** (gameStore.ts:200-243)

This derived store combines gameState, playerSessions, currentSessionId, and allowedActionsStore to create a comprehensive view projection for the UI. It performs several tasks:

- Finds the current session and determines if it can act
- Builds a set of allowed action keys from the server-supplied ValidAction list
- Computes all possible transitions using getNextStates()
- In test mode, uses all transitions; otherwise filters to only allowed actions
- Constructs viewProjection options including testMode, canAct, and an isAIControlled callback
- Returns the result of createViewProjection() with state, filtered transitions, and options

See gameStore.ts:200-243 for full implementation.

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
// gameStore.ts:175-183
export async function setPerspective(sessionId: string): Promise<void> {
  const current = get(currentSessionIdStore);
  if (current !== sessionId) {
    currentSessionIdStore.set(sessionId);
  }
  await networkClient.setPlayerId(sessionId);
  actionsByPlayer.set(networkClient.getCachedActionsMap());
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

## Action Flow

### User Clicks Action

```
1. UI Button Click
   └─> gameActions.executeAction(transition)

2. src/stores/gameStore.ts:282-327
   └─> gameClient.executeAction({ playerId, action, timestamp })

3. src/game/multiplayer/NetworkGameClient.ts:151-179
   └─> adapter.send({ type: 'EXECUTE_ACTION', gameId, playerId: string, action, timestamp: Date.now() })
   // Note: playerId is string (e.g., "player-0", "ai-1"), not number

4. src/server/offline/InProcessAdapter.ts
   └─> GameHost.executeAction(playerId, action, timestamp)

5. src/server/game/GameHost.ts:308-334
   └─> authorizeAndExecute(mpState, { playerId, action, timestamp }, getValidActionsComposed, rules) // composed variants

6. src/game/multiplayer/authorization.ts:110-151
   ├─> Find player session by playerId
   ├─> Verify action exists in composed `getValidActions(coreState)` (variants applied)
   ├─> Filter actions through capability pipeline (`filterActionsForSession`)
   └─> Execute: `executeAction(coreState, action, rules)` and update `lastActionAt`

7. src/game/core/actions.ts:16
   └─> Pure state transition, returns new state

8. GameHost.notifyListeners() (GameHost.ts:713-749)
   ├─> Creates filtered view for each listener
   │   └─> getVisibleStateForSession(coreState, session)
   └─> Calls listener(view) for all subscribers

9. NetworkGameClient.handleServerMessage() (NetworkGameClient.ts:276-304)
   ├─> Updates cachedView
   └─> Calls notifyListeners()

10. gameStore subscription fires (gameStore.ts:134-137)
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

## How It Works

### Pure State Storage Model

The game authority stores a single canonical `GameState` object and filters it on-demand for each viewing request. This creates a single source of truth where all state mutations happen in one place.

When a player requests their view, the filtering logic runs once, producing a `FilteredGameState` that contains only information visible to that player. The filtered view is computed fresh for each request rather than cached per player.

This approach centralizes filtering logic, makes debugging straightforward (inspect the pure state), and scales to any number of concurrent viewers without maintaining multiple state copies.

### Server Authority Model

The client receives a `GameView` containing `validActions` directly from the server and trusts it completely. The client does not perform any authorization checks or filter the action list based on player index.

Server-side filtering handles all authorization logic, including role-based capabilities and game variant rules. The client remains stateless and dumb—it only renders what the server provides and sends user intentions back.

This keeps game logic centralized on the server. Changes to authorization rules or capabilities require no client updates. The client's role is purely presentational: display the state, accept user input, send actions.

### Factory Function Pattern

Game instances are created via a simple factory function: `createGameAuthority(gameId, config)`. This function returns a new `GameAuthority` instance with no registry or instance tracking.

The caller (typically an adapter like `RoomAdapter`) owns the returned instance and manages its lifecycle. There is no global registry maintaining references to active games.

This pattern stays direct and testable. Creating a game is just a function call. Testing requires no setup of registry state. The factory returns a clean instance each time.

### Promise-Based Initialization

Asynchronous initialization uses stored promise resolvers rather than polling. When the `RoomAdapter` needs to wait for a `gameId`, it creates a Promise and stores the resolver function in a field.

When the initialization message arrives with the `gameId`, the message handler calls the stored resolver immediately. No `setInterval()` checks or arbitrary delays occur.

This provides clean async semantics. The waiting code gets resolved exactly when the value becomes available, with no wasted CPU cycles or timing fragility. Timeouts can be added declaratively using `Promise.race()`.

### Event Sourcing Model

Game state is computed by replaying a sequence of `GameAction` objects from the initial configuration. The authoritative representation is the action log, not the current state snapshot.

This enables perfect reproducibility: the same action sequence always produces the same final state. Bug reports can share URLs containing the complete action history. Debugging tools can time-travel through game history by replaying partial action sequences.

Network synchronization becomes simple: send actions, not state deltas. Each peer replays the same action log to arrive at the same state.

The cost is replay performance for long games. This is mitigated by only replaying on-demand (URL load, debug mode) rather than on every action during normal gameplay.

### Pure Function Architecture

State transitions are implemented as pure functions: `(state, action) => newState`. No function modifies existing state objects or maintains hidden internal state.

This makes every state transition deterministic and testable. Given the same inputs, the function always produces the same output. No setup or teardown is required for tests—just call the function.

Functions compose cleanly. Complex state transitions are built from simple pure functions. Debugging is straightforward: log inputs and outputs, no hidden state to track.

The pattern requires creating new objects rather than mutating existing ones. This slight verbosity is handled using immutable update patterns and structural sharing where objects are copied shallowly with only changed fields replaced.

---

## Common Tasks

### Enable Special Contracts

```typescript
// Enable nello via config
const config: GameConfig = {
  playerTypes: ['human', 'ai', 'ai', 'ai'],
  shuffleSeed: 12345,
  enableNello: true,        // Adds nello layer
  enableSplash: true,       // Adds splash layer
  enablePlunge: true,       // Adds plunge layer
  enableSevens: true        // Adds sevens layer
};

// getEnabledLayers(config) will return the appropriate layers
// They compose in GameHost constructor
```

### Add a New Layer

```typescript
// 1. Create layer file: src/game/layers/myLayer.ts
import { GameLayer } from './types';

export const myLayer: GameLayer = {
  name: 'my-layer',
  rules: {
    // Only override rules that differ from base/previous layers
    isValidBid: (state, bid, playerHand, prev) => {
      if (bid.type === 'MY_SPECIAL_BID') {
        return /* your validation logic */;
      }
      return prev;  // Delegate to other layers
    },

    calculateScore: (state, prev) => {
      if (state.currentBid.type !== 'MY_SPECIAL_BID') return prev;
      return /* your scoring logic */;
    }
  }
};

// 2. Add to getEnabledLayers in layers/utilities.ts
export function getEnabledLayers(config: GameConfig): GameLayer[] {
  const layers: GameLayer[] = [];
  if (config.enableMyLayer) layers.push(myLayer);
  // ... existing layers
  return layers;
}

// 3. Add config flag to GameConfig type
interface GameConfig {
  // ... existing fields
  enableMyLayer?: boolean;
}
```

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

### Check Layer + Variant Composition

```typescript
// In GameHost constructor
console.log('Layers:', this.layers.map(l => l.name));
console.log('Variants:', variantConfigs);

// In createView - check layer threading
const baseActions = getValidActions(coreState, [baseLayer]);
const layeredActions = getValidActions(coreState, this.layers, this.rules);
const finalActions = this.getValidActionsComposed(coreState);

console.log('Base actions:', baseActions.length);
console.log('After layers:', layeredActions.length);
console.log('After variants:', finalActions.length);
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

### Layer Not Applying

1. Check `getEnabledLayers()` includes your layer based on config flag
2. Verify config flag is set (e.g., `enableNello: true`)
3. Add logging in GameHost constructor: `console.log('Layers:', this.layers.map(l => l.name))`
4. Check layer rules are returning correct values (not `undefined`)
5. Verify `prev` parameter is being used to delegate to other layers

### Special Contract Not Available

1. Verify corresponding layer is enabled in config
2. Check that base layer is NOT filtering out the bid type
3. Test `isValidBid()` directly with the bid type
4. Look for bid type in action list: `actions.filter(a => a.type === 'bid')`

### Variant Not Applying

1. Check registration in `registry.ts`
2. Check variant config in GameConfig
3. Add logging in GameHost constructor
4. Verify variant function returns modified actions
5. Remember: variants work on top of layers (layers first, then variants)

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
- Read layer files in `src/game/layers/` - understand rule composition
- Read existing variants for patterns
- Compare base layer vs special contract layers - see how they differ
- Trace action flow with debugger

**Understand the Layer System**:
1. Read `src/game/layers/types.ts` - The 13 GameRules methods
2. Read `src/game/layers/base.ts` - Standard Texas 42
3. Read `src/game/layers/nello.ts` - How layers override rules
4. Read `src/game/layers/compose.ts` - How reduce pattern works
5. Trace how layers thread through GameHost → getValidActions → executeAction

---

## Quick Reference

**Factory Function**: `src/server/game/createGameAuthority.ts`
**Layer+Variant Composition**: `src/server/game/GameHost.ts:81-98`
**Layer Composition**: `src/game/layers/compose.ts:composeRules()`
**GameRules Interface**: `src/game/layers/types.ts` (13 methods)
**Base Layer**: `src/game/layers/base.ts` (standard Texas 42)
**Special Contract Layers**: `src/game/layers/{nello,splash,plunge,sevens}.ts`
**State Filtering**: `src/game/multiplayer/capabilityUtils.ts:getVisibleStateForSession()`
**Capability Auth**: `src/game/multiplayer/authorization.ts:canPlayerExecuteAction()`
**Standard Capabilities**: `src/game/multiplayer/capabilities.ts`
**Get Valid Actions (Pure)**: `src/game/multiplayer/authorization.ts:getValidActionsForPlayer()`
**Client Init**: `src/stores/gameStore.ts:78-137` (NetworkGameClient setup + async hydration)
**Client Actions**: `src/game/multiplayer/NetworkGameClient.ts:60-190` (async API, per-seat caches)
**Host Subscriptions**: `src/server/game/GameHost.ts:207-262` (HostViewUpdate payloads)
**Action Cache Store**: `src/stores/gameStore.ts:120-137` (per-session ValidAction map)
**Variant Template**: `src/game/variants/tournament.ts` (simple) or `oneHand.ts` (complex)
**Replay Function**: `src/game/core/replay.ts:16`
**Action Execution**: `src/game/core/actions.ts:16`
**Valid Actions**: `src/game/core/gameEngine.ts:getValidActions()`
**Client Creation**: `src/stores/gameStore.ts:84-85`
**Client State**: `src/stores/gameStore.ts:120-137` (writable + subscription)
**Derived Stores**: `src/stores/gameStore.ts:141, 143-165, 200-243`

**Pattern**: Layers → Variants → Execute → Filter → Trust → React
**Two-Level Composition**: Layers define rules (13 methods) → Variants transform actions

**Key Architectural Invariants**:
- Authority stores pure `GameState`, filters on-demand
- Client trusts server's `validActions`, never filters
- Capabilities determine visibility and authorization
- No polling - clean promise-based async
- Factory pattern, not registry pattern
