# Texas 42 - Architecture Orientation

**New to the codebase?** This document provides the mental models needed to navigate the architecture.

## Related Documentation
- **Vision**: [VISION.md](VISION.md) - Strategic direction and north star outcomes
- **Principles**: [ARCHITECTURE_PRINCIPLES.md](ARCHITECTURE_PRINCIPLES.md) - Design philosophy and mental models
- **Reference**: [CONCEPTS.md](CONCEPTS.md) - Complete implementation reference
- **Deep Dives**:
  - [MULTIPLAYER.md](MULTIPLAYER.md) - Multiplayer architecture (simple Socket/GameClient/Room pattern)
  - [archive/pure-layers-threaded-rules.md](archive/pure-layers-threaded-rules.md) - Layer system implementation (historical)

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
│  Svelte Stores (gameStore)      │  Reactive state + derived views
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  GameClient                     │  Fire-and-forget + subscriptions
│  - send(message)                │  No promises, updates via callback
│  - subscribe(callback)          │
└─────────────────────────────────┘
              ↕ Socket interface
┌─────────────────────────────────┐
│  Room (Orchestrator)            │  ★ COMPOSITION POINT ★
│  - Owns state and context       │  Takes send callback in constructor
│  - Manages sessions             │  Broadcasts via callback
│  - Delegates to pure helpers    │  Transport-agnostic
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  Pure Helpers (kernel.ts)       │  Stateless game logic
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

### Dumb Client Pattern
The client is intentionally "dumb" - it receives pre-computed data and displays it without computing game logic.

**Server-owned projection**: `buildKernelView()` computes derived fields (suit analysis, valid plays, UI hints) on the server side. The client receives a complete view ready for display.

**No rule logic on client**: The client never imports from `src/game/core/` or implements rule logic. All trump/suit/follow-suit decisions happen server-side via `GameRules`.

**DerivedViewFields**: Fields computed from state + rules that the client needs for display:
- Which dominoes are valid plays
- Suit analysis for the current player
- Trump indicators and hints

This pattern ensures consistency and prevents the client from diverging from server rules.

---

## Unified Layer System

**Problem**: Special contracts (nello, splash, plunge, sevens) need to change BOTH what's possible AND how execution works.

**Solution**: Unified Layer composition with two orthogonal surfaces:

```
LAYERS (execution rules + action generation) = Game Configuration
```

### Execution Rules Surface

**Purpose**: Define HOW the game executes (who acts, when tricks complete, how winners determined)

**Interface**: `GameRules` - 18 pure methods grouped into:
- **WHO** (3): getTrumpSelector, getFirstLeader, getNextPlayer
- **WHEN** (2): isTrickComplete, checkHandOutcome
- **HOW** (6): getLedSuit, suitsWithTrump, canFollow, rankInTrick, isTrump, calculateTrickWinner
- **VALIDATION** (3): isValidPlay, getValidPlays, isValidBid
- **SCORING** (3): getBidComparisonValue, isValidTrump, calculateScore
- **LIFECYCLE** (1): getPhaseAfterHandComplete

**Composition**: Layers override only what differs from base. Compose via reduce:
```typescript
const rules = composeRules([baseLayer, nelloLayer, splashLayer]);
// nelloLayer.isTrickComplete overrides base (3 plays not 4)
// Other rules pass through unchanged
```

**Example**: Nello partner sits out → override `isTrickComplete` to return true at 3 plays instead of 4.

### When to Add New Rules Methods

The GameRules interface currently has 18 methods, but **this number is not fixed**. Add new rules methods when you need:

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

### Pacing Layers: Consensus vs Speed

Two layers control game pacing with opposite philosophies:

**Consensus Layer** (`consensus.ts`) - Gates progress behind human acknowledgment
- Replaces `complete-trick` and `score-hand` with `agree-trick` / `agree-score` actions
- All human players must acknowledge before the game advances
- AI players don't vote (they're not waiting to see the result)
- Use for: Multiplayer human games where players need "tap to continue" pacing
- Derives state from `actionHistory` (pure function, no mutable state)

**Speed Layer** (`speed.ts`) - Auto-executes forced moves
- Marks single legal actions with `autoExecute: true`
- Sets `authority: 'system'` to bypass capability checks
- Use for: AI-only games, single-player practice, faster gameplay
- Examples: Only one legal play, only one legal bid, consensus actions when no player actions exist

**Contrast**:
| Aspect | Consensus | Speed |
|--------|-----------|-------|
| Philosophy | Wait for humans | Skip trivial decisions |
| Progress | Gated by acknowledgment | Auto-executed immediately |
| Use case | Multiplayer human games | AI games, practice mode |
| Actions | Adds agree-* actions | Annotates with autoExecute |

**Composition Note**: These layers can be composed together, but **order matters**:
- **Consensus → Speed** (current): Consensus replaces `complete-trick` with `agree-trick`, then speed auto-executes it. Result: human sees trick result, game auto-advances.
- **Speed → Consensus** (hypothetical): Speed marks `complete-trick` autoExecute, then consensus replaces it with `agree-trick` (losing the autoExecute flag). Result: human must manually tap to continue.

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

### Design Note: Layer State Inspection Pattern

Layers check `state.trump.type` to determine if their rules apply:

```typescript
// nello.ts
isTrickComplete: (state, prev) =>
  state.trump?.type === 'nello'
    ? state.currentTrick.length === 3
    : prev
```

**This is intentional, not an architectural violation.** Texas 42's game rules require:
1. Layers must be composable at config time (nello layer is always composed when enabled)
2. But layers are *activated* by player choice during trump selection

The state check is necessary because nello is selected as a trump option, not pre-determined by config. This pattern is:
- **Explicit** - Says exactly what it means
- **Local** - All nello logic lives in nello.ts
- **Necessary** - Matches how Texas 42 special contracts work

Executors remain mode-agnostic (calling `rules.method()`); layers handle mode-specific logic by checking trump type. Alternatives that try to hide this complexity (config-time composition, helper lambdas, trump-keyed dispatch) were evaluated and rejected as they move complexity without eliminating it.

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

## The Suit System: 7 Natural Suits + 1 Called Suit

Texas 42 uses an 8-suit system for game logic:

### The 7 Natural Suits (0-6)

Each pip value defines a suit. A domino's natural suit is its high pip:
- **0 (Blanks)**: 0-0
- **1 (Aces)**: 1-0, 1-1
- **2 (Deuces)**: 2-0, 2-1, 2-2
- **...and so on through...**
- **6 (Sixes)**: 6-0, 6-1, 6-2, 6-3, 6-4, 6-5, 6-6

### The Called Suit (Suit 7)

Suit 7 is the 8th suit - its membership is determined by what you **call** (declare as trump):

| Declaration | What goes to suit 7 |
|-------------|---------------------|
| "5s are trump" | All dominoes containing 5 (5-0, 5-1, 5-2, 5-3, 5-4, 5-5, 6-5) |
| "Doubles are trump" | All 7 doubles (0-0, 1-1, 2-2, 3-3, 4-4, 5-5, 6-6) |
| "Nello" | All 7 doubles (but no power to beat other suits) |
| "No-trump" | Nothing (empty) |

### Why "Called"?

The name comes from the game vocabulary: "I called 5s" or "I called doubles." When you declare trump, you **call** certain dominoes into suit 7. This terminology:

- **Works for pip-trump**: "I called 5s" → all 5s are in the called suit
- **Works for doubles-trump**: "I called doubles" → doubles are in the called suit
- **Works for nello**: Doubles are called together (just without power)
- **Is grounded in game language**: Players naturally say "I called trump"

### Code References

```typescript
// src/game/types.ts
export const CALLED = 7 as const;  // The 8th suit

// Usage in strength table keys:
"5-0|trump-blanks|led-called"  // Reading: "5-0 when blanks are trump and the called suit was led"
```

The constant `CALLED` (value 7) is used throughout the codebase. When you see suit 7 or `CALLED`, it refers to the absorbed/called suit whose membership depends on the trump declaration.

---

## File Map

**Core Engine** (pure utilities):
- `src/game/core/gameEngine.ts` - State machine, action generation
- `src/game/core/actions.ts` - Action executors (thread rules through)
- `src/game/core/rules.ts` - Pure game rules (suit following, trump)
- `src/game/types.ts` - Core types (GameState, GameAction, etc)

**Layer System** (unified execution + actions):
- `src/game/layers/types.ts` - GameRules interface (18 methods), Layer
- `src/game/layers/rules-base.ts` - Single source of truth for base rule logic (Crystal Palace)
- `src/game/layers/compose.ts` - composeRules() - reduce pattern
- `src/game/layers/base.ts` - Standard Texas 42
- `src/game/layers/{nello,splash,plunge,sevens,tournament,oneHand,hints,speed,consensus}.ts` - Layer implementations
- `src/game/layers/registry.ts` - LAYER_REGISTRY - name → layer mapping

**Multiplayer** (authorization, visibility):
- `src/multiplayer/types.ts` - MultiplayerGameState, PlayerSession, Capability, GameView
- `src/multiplayer/authorization.ts` - authorizeAndExecute(), filterActionsForSession()
- `src/multiplayer/capabilities.ts` - Capability builders + getVisibleStateForSession()
- `src/multiplayer/protocol.ts` - ClientMessage, ServerMessage types
- `src/multiplayer/local.ts` - createLocalGame(), attachAIBehavior()

**Server** (orchestration + authority):
- `src/server/Room.ts` - Room orchestrator (takes send callback, delegates to kernel)
- `src/server/HeadlessRoom.ts` - Minimal API for tools/scripts
- `src/kernel/kernel.ts` - Pure helper functions (executeKernelAction, buildKernelView)

**Client** (trust boundary):
- `src/multiplayer/Socket.ts` - Transport interface (send, onMessage, close)
- `src/multiplayer/GameClient.ts` - Client class (~43 lines, fire-and-forget)
- `src/stores/gameStore.ts` - Svelte facade over GameClient
- `src/App.svelte` - UI entry point

**Test Helpers**:
- `src/tests/helpers/stateBuilder.ts` - Fluent StateBuilder for test states
- `src/tests/helpers/dealConstraints.ts` - Generate hands with specific constraints
- `src/tests/helpers/executionContext.ts` - createTestContext() for unit tests

**Guardrail Tests** (architectural enforcement):
- `src/tests/guardrails/no-bypass.test.ts` - Prevent imports bypassing GameRules
- `src/tests/guardrails/projection-security.test.ts` - Verify no hidden state leaks
- `src/tests/guardrails/rule-contracts.test.ts` - Base + special contract rule conformance

---

## StateBuilder (Test State Construction)

Primary API for building game states in tests. Located in `src/tests/helpers/stateBuilder.ts`.

```typescript
import { StateBuilder } from '../helpers/stateBuilder';
import { ACES } from '../../game/types';

// Create state at any phase
const state = StateBuilder
  .inPlayingPhase({ type: 'suit', suit: ACES })
  .withSeed(12345)
  .withPlayerHand(0, ['6-6', '6-5', '5-5', '6-4', '3-2', '4-1', '5-0'])
  .withCurrentPlayer(1)
  .build();
```

**Factory methods**: `inBiddingPhase()`, `inTrumpSelection()`, `inPlayingPhase()`, `withTricksPlayed()`, `inScoringPhase()`, `gameEnded()`, plus special contracts (`nelloContract()`, `splashContract()`, `plungeContract()`, `sevensContract()`).

**Chainable modifiers**: `.withDealer()`, `.withCurrentPlayer()`, `.withTrump()`, `.withPlayerHand()`, `.withHands()`, `.withCurrentTrick()`, `.withTricks()`, `.withTeamScores()`, `.withSeed()`, `.withConfig()`.

**Deal constraints**: `.withPlayerDoubles(player, minCount)`, `.withPlayerConstraint(player, { minDoubles, exactDominoes, voidInSuit })`, `.withFillSeed()`.

---

## Essential Components Summary

| Component | Purpose | Key Insight |
|-----------|---------|-------------|
| **GameState** | Immutable game position | Never mutated, only transformed |
| **GameAction** | State transition unit | Source of truth via event sourcing |
| **GameRules** | 18 execution methods | Injected, never inspected |
| **Layer** | Unified execution + actions | Compose via reduce pattern |
| **Capability** | Permission token | Replace identity with permissions |
| **ExecutionContext** | Composed configuration | Bundles layers + rules + actions |
| **Room** | Game orchestrator | Owns state, sessions, AI, transport; delegates to pure helpers |

---

## Key Abstractions

### GameRules (18 Methods)

Interface executors call to determine behavior. Layers override specific methods:

```typescript
interface GameRules {
  // WHO: Player determination (3)
  getTrumpSelector(state, winningBid): number
  getFirstLeader(state, trumpSelector, trump): number
  getNextPlayer(state, currentPlayer): number

  // WHEN: Timing and completion (2)
  isTrickComplete(state): boolean
  checkHandOutcome(state): HandOutcome

  // HOW: Game mechanics (6)
  getLedSuit(state, domino): LedSuit
  suitsWithTrump(state, domino): LedSuit[]
  canFollow(state, led, domino): boolean
  rankInTrick(state, led, domino): number
  isTrump(state, domino): boolean
  calculateTrickWinner(state, trick): number

  // VALIDATION: Legality (3)
  isValidPlay(state, domino, playerId): boolean
  getValidPlays(state, playerId): Domino[]
  isValidBid(state, bid, playerHand?): boolean

  // SCORING: Outcomes (3)
  getBidComparisonValue(bid): number
  isValidTrump(trump): boolean
  calculateScore(state): [number, number]

  // LIFECYCLE: Game flow (1)
  getPhaseAfterHandComplete(state): GamePhase
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

**Standard builders** (in `src/multiplayer/capabilities.ts`):
- `humanCapabilities(playerIndex)` → act + observe own hand
- `aiCapabilities(playerIndex)` → act + observe own hand
- `spectatorCapabilities()` → observe all hands

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
gameStore.executeAction(action)
  ↓
GameClient.send({ type: 'EXECUTE_ACTION', action })  // Fire-and-forget
  ↓
Socket.send() → Room.handleMessage()
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
Room.broadcastState():
  - For each connected client
  - buildKernelView(mpState, clientId, ctx, metadata) → filter state + build view
  - Build GameView { state, validActions, metadata }
  - send(clientId, { type: 'STATE_UPDATE', view })
  ↓
GameClient.handleMessage()
  - Update cached view
  - Notify all subscribers
  ↓
gameStore subscription fires
  - Updates stores from view.state and view.validActions
  ↓
Derived stores recompute
  - gameState (from view)
  - viewProjection (UI metadata)
  ↓
UI components reactively update
```

**Key**: Room filters once per client perspective via pure helpers, broadcasts only GameView (no unfiltered state), client trusts server completely.

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

## AI System

### Strategy Implementations

Located in `src/game/ai/`:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **RandomAIStrategy** | Picks random valid action | Testing, baselines |
| **MonteCarloStrategy** | Monte Carlo simulation with rollouts | Default playable AI |

The AI system uses Monte Carlo simulation with configurable rollout strategies. See `src/game/ai/monte-carlo.ts` for implementation details.

---

## What's Not Here Yet

- **Online Mode**: Cloudflare Workers/Durable Objects (design ready, not implemented)
- **Spectator Mode**: Watch games with commentary (design ready, not implemented)

Current mode: Local in-process (browser main thread, createLocalGame wiring)

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
- For multiplayer details → [MULTIPLAYER.md](MULTIPLAYER.md)
