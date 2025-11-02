# Texas 42 - Architecture Orientation

**New to the codebase?** This document provides the mental models needed to navigate the architecture.

## Related Documentation
- **Concepts**: [CONCEPTS.md](CONCEPTS.md) - Complete reference of all architectural concepts
- **Synthesis**: [ARCHITECTURE_SYNTHESIS.md](ARCHITECTURE_SYNTHESIS.md) - Distilled overview of core components
- **Deep Dives**:
  - [remixed-855ccfd5.md](remixed-855ccfd5.md) - Multiplayer architecture specification
  - [pure-layers-threaded-rules.md](pure-layers-threaded-rules.md) - Layer system implementation
  - [GAME_ONBOARDING.md](GAME_ONBOARDING.md) - Detailed implementation guide

---

## Core Architecture Pattern

The entire system is built around this fundamental transformation:

```
STATE → ACTION → NEW STATE
```

Everything else exists to:
1. **Generate** valid actions (what can happen)
2. **Execute** actions deterministically (make it happen)
3. **Filter** results (who sees what)

## Philosophy

1. **Event Sourcing**: `state = replayActions(initialConfig, actionHistory)` - state is derived, actions are truth
2. **Pure Functions**: All state transitions are pure, reproducible, testable
3. **Composition Over Configuration**: `f(g(h(base)))` instead of flags and conditionals
4. **Zero Coupling**: Core engine has no multiplayer/variant awareness
5. **Parametric Polymorphism**: Executors delegate to injected rules, never inspect state

---

## The Stack

```
┌─────────────────────────────────┐
│  UI Components (Svelte)         │  Reactive views
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Svelte Stores                  │  Reactive state + derived views
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  NetworkGameClient              │  Caches filtered state + actions
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Transport Layer                │  Message routing (in-process/Worker/WS)
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  GameServer (Orchestrator)      │  Routes messages, manages lifecycle
│  - Creates and owns GameKernel  │  Spawns/destroys AI clients
│  - Broadcasts state updates     │  Handles player sessions
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  GameKernel (Authority)         │  ★ COMPOSITION POINT ★
│  - Composes layers → rules      │  Pure state storage
│  - Composes variants → actions  │  Per-request filtering
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Variant System                 │  Transform actions (what's possible)
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Layer System                   │  Provide rules (how execution works)
│  13 GameRules methods           │  Special contracts live here
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Core Engine                    │  Pure utilities, zero coupling
└─────────────────────────────────┘
```

---

## Mental Models

Think of the system as:
- **A state machine** - Games flow through well-defined positions via actions
- **A composition system** - Behaviors stack like camera filters
- **A pure function** - `newState = f(oldState, action)`
- **A trust hierarchy** - Server validates, clients display

---

## Two-Level Composition (The Key Insight)

**Problem**: Special contracts (nello, splash, plunge, sevens) need to change BOTH what's possible AND how execution works.

**Solution**: Two orthogonal composition surfaces:

```
LAYERS (execution rules) × VARIANTS (action transformation) = Game Configuration
```

### Level 1: Layers → Execution Rules

**Purpose**: Define HOW the game executes (who acts, when tricks complete, how winners determined)

**Interface**: `GameRules` - 13 pure methods grouped into:
- **WHO** (3): getTrumpSelector, getFirstLeader, getNextPlayer
- **WHEN** (2): isTrickComplete, checkHandOutcome
- **HOW** (2): getLedSuit, calculateTrickWinner
- **VALIDATION** (3): isValidPlay, getValidPlays, isValidBid
- **SCORING** (3): getBidComparisonValue, isValidTrump, calculateScore

**Composition**: Layers override only what differs from base. Compose via reduce:
```typescript
const rules = composeRules([baseLayer, nelloLayer, splashLayer]);
// nelloLayer.isTrickComplete overrides base (3 plays not 4)
// Other rules pass through unchanged
```

**Example**: Nello partner sits out → override `isTrickComplete` to return true at 3 plays instead of 4.

### Level 2: Variants → Action Transformation

**Purpose**: Transform WHAT actions are possible (filter, annotate, script, replace)

**Interface**: `Variant = (StateMachine) → StateMachine` where `StateMachine = (state) => GameAction[]`

**Operations**:
- **Filter**: Remove actions (tournament removes special bids)
- **Annotate**: Add metadata (hints, autoExecute flags)
- **Script**: Inject actions (oneHand scripts bidding)
- **Replace**: Swap action types (oneHand replaces score-hand with end-game)

**Composition**: Variants wrap the state machine via reduce:
```typescript
const composed = applyVariants(baseStateMachine, [tournament, oneHand]);
// tournament filters, then oneHand scripts
```

### The Composition Point (GameKernel Constructor)

**ONE place** where everything composes:

```typescript
// 1. Layers provide rules
const layers = [baseLayer, ...getEnabledLayers(config)];
const rules = composeRules(layers);

// 2. Thread layers through base state machine
const baseWithLayers = (state) => getValidActions(state, layers, rules);

// 3. Variants transform the layered actions
const getValidActionsComposed = applyVariants(baseWithLayers, variantConfigs);

// 4. Create ExecutionContext (groups the triple together)
this.ctx = Object.freeze({
  layers: Object.freeze(layers),
  rules,
  getValidActions: getValidActionsComposed
});

// 5. Use context everywhere
// buildKernelView(state, forPlayerId, this.ctx, metadata)
// executeKernelAction(mpState, playerId, action, timestamp, this.ctx)
```

**Result**: Layers change execution semantics, variants change action availability. Executors have ZERO conditional logic - they call `ctx.rules.method()` and trust the result. ExecutionContext groups the always-traveling-together parameters (layers, rules, getValidActions) into a single immutable object.

---

## File Map

**Core Engine** (pure utilities):
- `src/game/core/gameEngine.ts` - State machine, action generation
- `src/game/core/actions.ts` - Action executors (thread rules through)
- `src/game/core/rules.ts` - Pure game rules (suit following, trump)
- `src/game/types.ts` - Core types (GameState, GameAction, etc)

**Layer System** (execution semantics):
- `src/game/layers/types.ts` - GameRules interface (13 methods), GameLayer
- `src/game/layers/compose.ts` - composeRules() - reduce pattern
- `src/game/layers/base.ts` - Standard Texas 42
- `src/game/layers/{nello,splash,plunge,sevens}.ts` - Special contracts
- `src/game/layers/utilities.ts` - getEnabledLayers() - config → layers

**Variant System** (action transformation):
- `src/game/variants/registry.ts` - applyVariants() - composition
- `src/game/variants/{tournament,oneHand}.ts` - Variant implementations

**Multiplayer** (authorization, visibility):
- `src/game/multiplayer/types.ts` - MultiplayerGameState, PlayerSession, Capability
- `src/game/multiplayer/authorization.ts` - authorizeAndExecute(), filterActionsForSession()
- `src/game/multiplayer/capabilityUtils.ts` - getVisibleStateForSession()
- `src/game/multiplayer/capabilities.ts` - Standard capability builders

**Server** (orchestration + authority):
- `src/server/GameServer.ts` - Orchestrator, routes messages, manages lifecycle
- `src/kernel/GameKernel.ts` - Authority, composition point, pure game logic
- `src/server/transports/Transport.ts` - Transport abstraction interface
- `src/server/transports/InProcessTransport.ts` - In-browser transport

**Client** (trust boundary):
- `src/game/multiplayer/NetworkGameClient.ts` - Client interface, caches filtered state
- `src/stores/gameStore.ts` - Svelte stores, reactive state management
- `src/App.svelte` - UI entry point

---

## Essential Components Summary

| Component | Purpose | Key Insight |
|-----------|---------|-------------|
| **GameState** | Immutable game position | Never mutated, only transformed |
| **GameAction** | State transition unit | Source of truth via event sourcing |
| **GameRules** | 13 execution methods | Injected, never inspected |
| **GameLayer** | Rule overrides | Compose via reduce pattern |
| **Variant** | Action transformer | Wrap state machine functions |
| **Capability** | Permission token | Replace identity with permissions |
| **ExecutionContext** | Composed configuration | Bundles layers + rules + actions |
| **GameKernel** | Game authority | Single composition point |
| **GameServer** | Orchestrator | Manages lifecycle, not logic |

---

## Key Abstractions

### GameRules (13 Methods)

Interface executors call to determine behavior. Layers override specific methods:

```typescript
interface GameRules {
  // WHO: Player determination
  getTrumpSelector(state, winningBid): number
  getFirstLeader(state, trumpSelector, trump): number
  getNextPlayer(state, currentPlayer): number

  // WHEN: Timing and completion
  isTrickComplete(state): boolean
  checkHandOutcome(state): HandOutcome | null

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
}
```

**Key**: Each rule gets `prev` parameter - delegate to previous layer or override. Compose via reduce.

### GameLayer

Layer that overrides specific rules and/or transforms actions:

```typescript
interface GameLayer {
  name: string;
  rules?: Partial<GameRules>;  // Override execution behavior
  getValidActions?: (state, prev) => GameAction[];  // Transform actions
}
```

**Pattern**: Check state, return override or pass through `prev`.

### Variant

Function that transforms the state machine:

```typescript
type StateMachine = (state: GameState) => GameAction[];
type Variant = (base: StateMachine) => StateMachine;
```

**Pattern**: Call base, then filter/annotate/replace actions.

### Capability

Token that grants permission to act or observe:

```typescript
type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-own-hand' }
  | { type: 'observe-all-hands' }
  | { type: 'see-hints' }
  // ... 5 more
```

**Purpose**: Replace boolean flags with composable permissions. Determines:
1. What actions player can execute (authorization)
2. What information player can see (visibility)

---

## Request Flow

```
User clicks button
  ↓
gameStore.executeAction({ playerId, action, timestamp })
  ↓
NetworkGameClient.executeAction() → connection.send(EXECUTE_ACTION)
  ↓
Transport.send() → GameServer.handleMessage()
  ↓
GameServer routes to GameKernel.executeAction()
  ↓
GameKernel.executeAction():
  - authorizeAndExecute(mpState, request, composedStateMachine, rules)
  - Find session by playerId
  - Get all valid actions via composed state machine
  - Filter by session capabilities
  - Execute: executeAction(coreState, action, rules)
  - Update lastActionAt
  - Process auto-execute actions
  ↓
GameKernel.notifyListeners():
  - For each subscriber (with perspective)
  - getVisibleStateForSession(coreState, session) → filter state
  - filterActionsForSession(session, allActions) → filter actions
  - Build KernelUpdate { view, state, actions }
  ↓
GameServer receives update and broadcasts via Transport
  ↓
Transport.send() → connection.onMessage(STATE_UPDATE)
  ↓
NetworkGameClient.handleServerMessage()
  - Cache filtered state + actions map
  - Call notifyListeners()
  ↓
gameStore subscription fires
  - clientState.set(state)
  - actionsByPlayer.set(actionsMap)
  ↓
Derived stores recompute
  - gameState (FilteredGameState)
  - viewProjection (UI metadata)
  ↓
UI components reactively update
```

**Key**: GameKernel filters once per subscriber perspective, Transport routes messages, client trusts filtered data.

---

## How to Extend

### Add a New Layer (Special Contract)

1. Create `src/game/layers/myContract.ts`:
```typescript
export const myContractLayer: GameLayer = {
  name: 'my-contract',
  rules: {
    // Override only what differs from base
    isTrickComplete: (state, prev) =>
      state.trump.type === 'my-contract' ? customLogic : prev,
    calculateScore: (state, prev) =>
      state.currentBid.type === 'MY_CONTRACT' ? customScore : prev
  }
};
```

2. Add to `src/game/layers/utilities.ts:getEnabledLayers()`:
```typescript
if (config.enableMyContract) layers.push(myContractLayer);
```

3. Add config flag to `GameConfig` type

**That's it**. No changes to core executors.

### Add a New Variant

1. Create `src/game/variants/myVariant.ts`:
```typescript
export const myVariant: VariantFactory = (config) => (base) => (state) => {
  const actions = base(state);
  // Filter, annotate, script, or replace
  return modifiedActions;
};
```

2. Register in `src/game/variants/registry.ts`

3. Use in config: `{ variant: { type: 'my-variant' } }`

### Enable a Feature

Most features are config flags:
```typescript
const config: GameConfig = {
  enableNello: true,      // Adds nello layer
  enableSplash: true,     // Adds splash layer
  enablePlunge: true,     // Adds plunge layer
  enableSevens: true,     // Adds sevens layer
  variant: { type: 'tournament' }  // Applies tournament variant
};
```

Flags → getEnabledLayers() → compose in GameKernel constructor.

---

## Architectural Invariants

1. **Pure State Storage**: GameKernel stores unfiltered GameState, filters on-demand per request
2. **Server Authority**: Client trusts server's validActions list, never refilters
3. **Capability-Based Access**: Permissions via capability tokens, not identity checks
4. **Single Composition Point**: GameKernel constructor is ONLY place layers/variants compose
5. **Zero Coupling**: Core engine has zero knowledge of multiplayer/variants/special contracts
6. **Parametric Polymorphism**: Executors call `rules.method()`, never `if (nello)`
7. **Event Sourcing**: State derivable from `replayActions(config, history)`
8. **Clean Separation**: GameServer orchestrates, GameKernel executes, Transport routes

**Violation of any invariant = architectural regression.**

---

## Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| Adding conditional logic in executors | Use parametric polymorphism - delegate to `rules.method()` |
| Modifying state directly | Create new state objects via spread operator |
| Checking player identity for permissions | Use capability tokens instead |
| Adding game logic to GameServer | Put it in GameKernel or layers |
| Filtering state at storage | Store unfiltered, filter on-demand per request |
| Client-side validation | Trust server's validActions list completely |
| Deep coupling between layers | Keep layers focused on single responsibility |

---

## What's Not Here Yet

- **Offline Mode**: Web Worker transport (spec ready, not implemented)
- **Online Mode**: Cloudflare Workers/Durable Objects (spec ready, not implemented)
- **Progressive Enhancement**: Offline → online transition (spec ready, not implemented)

Current transport: `InProcessTransport` (in-browser, single-process)

---

## Quick Start Guide

### Commands
```bash
npm run typecheck   # Ensure zero errors (run this often!)
npm test           # Run all tests
npm run dev        # Start dev server
npm run build      # Production build
```

### Debugging Tips

| Issue | Where to Look |
|-------|---------------|
| Action not available | GameKernel constructor (composition), `authorizeAndExecute` (filtering) |
| Wrong game behavior | Layer implementation, `GameRules` methods |
| State not updating | `executeAction`, check if action is authorized |
| UI not reactive | Svelte stores, `ViewProjection` computation |
| Message not routing | `GameServer.handleMessage()`, Transport layer |

### Understanding the Code

1. **Start with types** - The architecture is type-driven. Follow these key types:
   - `GameRules` → Understand execution semantics
   - `GameLayer` → See how rules compose
   - `Variant` → Learn action transformation
   - `Capability` → Grasp permission model

2. **Trace a request** - Follow an action from click to UI update using the Request Flow diagram above

3. **Read the tests** - Tests demonstrate intended usage and edge cases

---

**Next Steps**:
- For detailed concepts → [CONCEPTS.md](CONCEPTS.md)
- For implementation guide → [GAME_ONBOARDING.md](GAME_ONBOARDING.md)
- For multiplayer details → [remixed-855ccfd5.md](remixed-855ccfd5.md)
