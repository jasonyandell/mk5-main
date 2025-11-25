# Texas 42 - Architecture Orientation

**New to the codebase?** This document provides the mental models needed to navigate the architecture.

## Related Documentation
- **Vision**: [VISION.md](VISION.md) - Strategic direction and north star outcomes
- **Principles**: [ARCHITECTURE_PRINCIPLES.md](ARCHITECTURE_PRINCIPLES.md) - Design philosophy and mental models
- **Reference**: [CONCEPTS.md](CONCEPTS.md) - Complete implementation reference
- **Deep Dives**:
  - [remixed-855ccfd5.md](remixed-855ccfd5.md) - Multiplayer architecture specification
  - [archive/pure-layers-threaded-rules.md](archive/pure-layers-threaded-rules.md) - Layer system implementation (historical)
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
4. **Zero Coupling**: Core engine has no multiplayer/layer awareness
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
│  Transport Layer                │  Connection objects (self-contained)
│  Connection.reply() pattern     │  Each connection routes to itself
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Room (Orchestrator)            │  ★ COMPOSITION POINT ★
│  - Owns state and context       │  Routes via connection.reply()
│  - Manages sessions and AI      │  Spawns/destroys AI clients
│  - Delegates to pure helpers    │  Broadcasts state updates
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Pure Helpers                   │  Stateless game logic
│  - executeKernelAction          │  - buildKernelView
│  - buildActionsMap              │  - processAutoExecute
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Layer System                   │  Unified composition system
│  Execution rules + actions      │  Special contracts + transformers
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Core Engine                    │  Pure utilities, zero coupling
└─────────────────────────────────┘
```

---

## Mental Models

Understanding the architecture requires thinking about it in multiple ways:

### The Game as a State Machine
Every game position has defined transitions to next positions. Actions are edges, states are nodes. AI explores this graph to make decisions. Games flow through well-defined positions via actions.

### Layers as Lenses and Decorators
Each layer provides both a lens (execution rules) and decorator (action transformation). Stack layers to create new game modes. Nello adds a "3-player trick" lens, OneHand scripts bidding as a decorator.

### Capabilities as Keys
Each capability unlocks specific functionality. Collect keys to gain more power. `observe-all-hands` unlocks full visibility, `act-as-player` unlocks actions for a seat.

### The Kernel as a Pure Function
Given state and action, always produces same new state. No hidden state or side effects. `newState = f(oldState, action)` always holds.

### A Trust Hierarchy
Server validates, clients display. Clear security boundary: server is authoritative, clients are delegating.

---

## Unified Layer System

**Problem**: Special contracts (nello, splash, plunge, sevens) need to change BOTH what's possible AND how execution works.

**Solution**: Unified Layer composition with two orthogonal surfaces:

```
LAYERS (execution rules + action generation) = Game Configuration
```

### Execution Rules Surface

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

### When to Add New Rules Methods

The GameRules interface currently has 13 methods, but **this number is not fixed**. Add new rules methods when you need:

**New terminal states**: Mode needs hand to end for new reasons
- Example: `checkHandOutcome` was added to support nello/plunge early termination
- Pattern: Add method to GameRules → Base returns default → Layers override

**New execution semantics**: Mode changes who/when/how core mechanics work
- Example: `isTrickComplete` enables nello's 3-player tricks
- Pattern: Executor delegates to rule → Layer provides mode-specific logic

**Mode-specific validation**: Mode changes what's legal
- Example: `isValidBid` allows Layers to validate special bid types
- Pattern: Executor calls rule for all validation → No state inspection

**The key insight**: When executors need mode-specific behavior, add a new rule method. Never add conditionals to executors.

### Action Generation Surface

**Purpose**: Transform WHAT actions are possible (filter, annotate, script, replace)

**Interface**: Layers implement `getValidActions?: (state, prev) => GameAction[]`

**Operations**:
- **Filter**: Remove actions (tournament removes special bids)
- **Annotate**: Add metadata (hints, autoExecute flags)
- **Script**: Inject actions (oneHand scripts bidding)
- **Replace**: Swap action types (oneHand replaces score-hand with end-game)

**Composition**: Layers compose action generation via reduce pattern

### The Composition Point (Room Constructor)

**ONE place** where everything composes:

```typescript
// 1. Layers provide rules and action generation
const layers = [baseLayer, ...getEnabledLayers(config)];
const rules = composeRules(layers);

// 2. Compose action generation from layers
const getValidActionsComposed = composeLayerActions(layers);

// 3. Create ExecutionContext (groups everything together)
this.ctx = Object.freeze({
  layers: Object.freeze(layers),
  rules,
  getValidActions: getValidActionsComposed
});

// 4. Use context everywhere
// buildKernelView(state, forPlayerId, this.ctx, metadata)
// executeKernelAction(mpState, playerId, action, this.ctx)
```

**Result**: Layers change both execution semantics and action availability. Executors have ZERO conditional logic - they call `ctx.rules.method()` and trust the result. ExecutionContext groups the always-traveling-together parameters (layers, rules, getValidActions) into a single immutable object.

---

## HeadlessRoom API (for Tools and Simulations)

For tools, scripts, and simulations that need deterministic game execution without the full transport layer, use **HeadlessRoom**:

```typescript
import { HeadlessRoom } from '@/server/HeadlessRoom';

// Create room with config and seed
const room = new HeadlessRoom(config, seed);

// Execute actions directly
room.executeAction(playerId, action);

// Get current state
const state = room.getState();

// Get filtered view for a player
const view = room.getView(playerId);
```

### When to Use HeadlessRoom

**Use HeadlessRoom for:**
- **Simulation tools** (gameSimulator) - Run AI vs AI games
- **Replay utilities** (urlReplay) - Deterministic replay from URLs
- **Scripts and CLI tools** - Command-line game execution
- **Performance testing** - Batch processing without transport overhead
- **Debugging tools** - Direct state inspection and manipulation

**Do NOT use HeadlessRoom for:**
- **Production game instances** - Use Room (includes transport, sessions, AI management)
- **Unit tests of composition** - Use `createTestContext()` from test helpers
- **Multiplayer games** - Use Room with proper transport layer

### HeadlessRoom vs Room

| Feature | Room | HeadlessRoom |
|---------|------|--------------|
| **Purpose** | Production multiplayer | Tools & simulations |
| **Transport** | ✅ Full protocol support | ❌ None (direct API) |
| **Sessions** | ✅ Multi-session mgmt | ❌ Single execution context |
| **AI Management** | ✅ Spawn/destroy AI | ❌ Manual AI execution |
| **Filtering** | ✅ Per-perspective views | ✅ View API available |
| **Composition** | ✅ Full ExecutionContext | ✅ Full ExecutionContext |
| **Use case** | Real games | Scripts & testing |

### Design Philosophy

Both Room and HeadlessRoom compose ExecutionContext the same way - they are the **ONLY** places where composition happens. This ensures:

1. **Configuration consistency** - Same rules everywhere
2. **Clear ownership** - Composition logic in one place
3. **Enforced invariants** - ESLint prevents direct composition elsewhere
4. **Testability** - Clear boundaries for what's being tested

---

---

## File Map

**Core Engine** (pure utilities):
- `src/game/core/gameEngine.ts` - State machine, action generation
- `src/game/core/actions.ts` - Action executors (thread rules through)
- `src/game/core/rules.ts` - Pure game rules (suit following, trump)
- `src/game/types.ts` - Core types (GameState, GameAction, etc)

**Layer System** (unified execution + actions):
- `src/game/layers/types.ts` - GameRules interface (13 methods), Layer
- `src/game/layers/compose.ts` - composeRules() - reduce pattern
- `src/game/layers/base.ts` - Standard Texas 42
- `src/game/layers/{nello,splash,plunge,sevens,tournament,oneHand}.ts` - Layer implementations
- `src/game/layers/utilities.ts` - getEnabledLayers() - config → layers

**Multiplayer** (authorization, visibility):
- `src/game/multiplayer/types.ts` - MultiplayerGameState, PlayerSession, Capability
- `src/game/multiplayer/authorization.ts` - authorizeAndExecute(), filterActionsForSession()
- `src/game/multiplayer/capabilityUtils.ts` - getVisibleStateForSession()
- `src/game/multiplayer/capabilities.ts` - Standard capability builders

**Server** (orchestration + authority):
- `src/server/Room.ts` - Room orchestrator (combines server + kernel)
- `src/kernel/kernel.ts` - Pure helper functions (executeKernelAction, buildKernelView)
- `src/server/transports/Transport.ts` - Transport abstraction, Connection interface with reply()
- `src/server/transports/InProcessTransport.ts` - In-browser transport implementation

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
| **Layer** | Unified execution + actions | Compose via reduce pattern |
| **Capability** | Permission token | Replace identity with permissions |
| **ExecutionContext** | Composed configuration | Bundles layers + rules + actions |
| **Room** | Game orchestrator | Owns state, sessions, AI, transport; delegates to pure helpers |

---

## Key Abstractions

### GameRules (13 Methods)

Interface executors call to determine behavior. RuleSets override specific methods:

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
}
```

**Key**: Each rule gets `prev` parameter - delegate to previous ruleset or override. Compose via reduce.

**Pattern**: Discriminated union makes invalid states unrepresentable
- `checkHandOutcome` returns `HandOutcome` with two cases:
  - `{ determined: false }` - outcome not yet determined
  - `{ determined: true, reason: string, decidedAtTrick?: number }` - outcome determined
- TypeScript enforces: can't access reason unless determined === true
- Aligns with Result<T> pattern used throughout codebase

### Layer

Unified composition unit that overrides specific rules and/or transforms actions:

```typescript
interface Layer {
  name: string;
  rules?: Partial<GameRules>;  // Override execution behavior
  getValidActions?: (state, prev) => GameAction[];  // Transform actions
}
```

**Pattern**: Check state, return override or pass through `prev`.

### Capability

Token that grants permission to act or observe:

```typescript
type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-hands'; playerIndices: number[] | 'all' };
```

**Purpose**: Replace boolean flags with composable permissions. Determines:
1. What actions player can execute (authorization)
2. What information player can see (visibility)

**Standard sets**:
- **Player**: `[{ type: 'act-as-player', playerIndex: N }, { type: 'observe-hands', playerIndices: [N] }]`
- **Spectator**: `[{ type: 'observe-hands', playerIndices: 'all' }]`

### Action Authority

Actions can execute with two different authority models:

```typescript
type ActionAuthority = 'player' | 'system';
```

**Player authority** (default): Action must be authorized by a player session's capabilities
- User-initiated actions (clicks, button presses)
- Requires session to have `act-as-player` capability
- Standard authorization flow

**System authority**: Action executes as part of deterministic game script, bypassing capability checks
- Scripted setup actions (one-hand transformer)
- Auto-executed game flow (forced moves in speed mode)
- Still validated for structural correctness (must be valid in current state)
- Set via `action.meta.authority = 'system'`

---

## Request Flow

```
User clicks button
  ↓
gameStore.executeAction({ playerId, action })
  ↓
NetworkGameClient.executeAction() → connection.send(EXECUTE_ACTION)
  ↓
Transport.send() → Room.handleMessage()
  ↓
Room routes to executeKernelAction()
  ↓
executeKernelAction():
  - authorizeAndExecute(mpState, request, ctx.getValidActions, ctx.rules)
  - Check action authority: system actions bypass capability checks
  - Find session by playerId
  - Get all valid actions via composed state machine
  - Filter by session capabilities (if player authority)
  - Execute: executeAction(coreState, action, ctx.rules)
  - Process auto-execute actions
  ↓
Room.notifyListeners():
  - For each subscriber (with perspective)
  - buildKernelView(mpState, playerId, ctx, metadata) → filter state + build view
  - Build GameView { state, transitions, metadata }
  - connection.reply(STATE_UPDATE) → direct delivery to client
  ↓
Connection delivers message to client handler
  ↓
NetworkGameClient.handleServerMessage()
  - Cache GameView (filtered state + transitions)
  - Call notifyListeners()
  ↓
gameStore subscription fires
  - Updates stores from view.state and view.transitions
  ↓
Derived stores recompute
  - gameState (from view)
  - viewProjection (UI metadata)
  ↓
UI components reactively update
```

**Key**: Room filters once per subscriber perspective via pure helpers, broadcasts only GameView (no unfiltered state), client trusts server completely.

---

## How to Extend

### Add a New Layer

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

3. Add config flag to `GameConfig` type

**That's it**. No changes to core executors.

### Enable a Feature

Most features are config flags:
```typescript
const config: GameConfig = {
  enableNello: true,      // Adds nello layer
  enableSplash: true,     // Adds splash layer
  enablePlunge: true,     // Adds plunge layer
  enableSevens: true,     // Adds sevens layer
  enableTournament: true, // Adds tournament layer
};
```

Flags → getEnabledLayers() → compose in Room constructor.

### Add Mode-Specific Execution Behavior

When a new mode needs different execution semantics (not just different actions), follow this pattern:

#### Step 1: Identify the extension point
Ask: "Does the executor need to know about this behavior?"
- YES → Add GameRules method
- NO → Use Layer's getValidActions

#### Step 2: Add method to GameRules interface
```typescript
// src/game/layers/types.ts
export interface GameRules {
  // ... existing methods ...

  /**
   * Your new rule with clear doc comment.
   * Explain what base returns and when RuleSets override.
   */
  myNewRule(state: GameState, ...params): ReturnType;
}
```

#### Step 3: Implement in base Layer
```typescript
// src/game/layers/base.ts
export const baseRules: GameRules = {
  // ... existing rules ...

  myNewRule: (state, ...params) => {
    // Default behavior for standard Texas 42
    return defaultValue;
  }
};
```

#### Step 4: Override in mode Layers
```typescript
// src/game/layers/myMode.ts
export const myModeLayer: Layer = {
  name: 'my-mode',
  rules: {
    myNewRule: (state, ...params, prev) => {
      if (state.trump.type === 'my-mode') {
        return modeSpecificValue;
      }
      return prev; // Delegate to previous layer
    }
  }
};
```

#### Step 5: Update compose.ts
Add the new rule to the composition reduce pattern:
```typescript
// src/game/layers/compose.ts
export function composeRules(layers: Layer[]): GameRules {
  // ... existing code ...

  myNewRule: (state, ...params) => {
    let result = baseRules.myNewRule(state, ...params);
    for (const layer of layers) {
      if (layer.rules?.myNewRule) {
        result = layer.rules.myNewRule(state, ...params, result);
      }
    }
    return result;
  }
}
```

#### Step 6: Use in executors
Executors delegate to the rule, never inspect state:
```typescript
// src/game/core/actions.ts
function myExecutor(state: GameState, action: GameAction, rules: GameRules) {
  // DON'T: if (state.mode === 'special') { ... }
  // DO: Delegate to rule
  const value = rules.myNewRule(state, action.param);
  // Use value to determine behavior
}
```

**Real Example - Implemented**: One-hand mode needs a new terminal state.

- **Added**: `getPhaseAfterHandComplete` rule method to GameRules interface
- **Base implementation** (`src/game/layers/base.ts`): Returns `'bidding'` (continue to next hand)
- **OneHand override** (`src/game/layers/oneHand.ts`): Returns `'one-hand-complete'` (terminal state)
- **Executor** (`executeScoreHand` in `src/game/core/actions.ts`): Delegates to rule instead of hardcoding phase
- **See**: ADR-20251112-onehand-terminal-phase.md for full implementation details

**Key Principle**: Executors must remain mode-agnostic. If you're tempted to add `if (mode)` to an executor, add a rule method instead.

---

## Architectural Invariants

1. **Pure State Storage**: Room stores unfiltered GameState, filters on-demand per request
2. **Server Authority**: Client trusts server's validActions list, never refilters
3. **Capability-Based Access**: Permissions via capability tokens, not identity checks
4. **Single Composition Point**: Room/HeadlessRoom are ONLY places where ExecutionContext is composed
   - ✅ Enforced via ESLint rules (`no-restricted-imports`)
   - ✅ Verified by architecture tests (`composition.test.ts`)
   - ✅ Documented in ADR-20251110-single-composition-point.md
5. **Zero Coupling**: Core engine has zero knowledge of multiplayer/layers/special contracts
6. **Parametric Polymorphism**: Executors call `rules.method()`, never `if (nello)`
7. **Event Sourcing**: State derivable from `replayActions(config, history)`
8. **Clean Separation**: Room orchestrates, pure helpers execute, Transport routes

**Violation of any invariant = architectural regression.**

---

## Common Pitfalls & Solutions

| Pitfall | Solution |
|---------|----------|
| Adding conditional logic in executors | Use parametric polymorphism - delegate to `rules.method()` |
| Modifying state directly | Create new state objects via spread operator |
| Checking player identity for permissions | Use capability tokens instead |
| Adding game logic to Room | Put it in pure helpers or Layers |
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
| Action not available | Room constructor (composition), `authorizeAndExecute` (filtering) |
| Wrong game behavior | Layer implementation, `GameRules` methods |
| State not updating | `executeAction`, check if action is authorized |
| UI not reactive | Svelte stores, `ViewProjection` computation |
| Message not routing | `Room.handleMessage()`, Transport layer |

### Understanding the Code

1. **Start with types** - The architecture is type-driven. Follow these key types:
   - `GameRules` → Understand execution semantics
   - `Layer` → See how rules and actions compose
   - `Capability` → Grasp permission model

2. **Trace a request** - Follow an action from click to UI update using the Request Flow diagram above

3. **Read the tests** - Tests demonstrate intended usage and edge cases

---

## Design Philosophy

These principles guide all architectural decisions:

### Simplicity Through Composition
Complex behavior emerges from composing simple, pure functions. No monolithic classes or deep inheritance hierarchies.

### Correct by Construction
Use the type system to prevent errors at compile time. Make illegal states unrepresentable. If it compiles, it's more likely to be correct.

### Explicit Over Implicit
All behavior is explicitly defined through composition. No hidden magic or implicit conventions. What you see is what happens.

### Immutability as Default
State is never mutated, only transformed. Enables reasoning, debugging, and time-travel. New states are created through transformation.

### Trust Through Verification
Server validates everything, clients trust completely. Clear security boundary enables simple client code and guaranteed consistency.

---

**Next Steps**:
- For design philosophy and patterns → [ARCHITECTURE_PRINCIPLES.md](ARCHITECTURE_PRINCIPLES.md)
- For implementation reference → [CONCEPTS.md](CONCEPTS.md)
- For multiplayer details → [remixed-855ccfd5.md](remixed-855ccfd5.md)
