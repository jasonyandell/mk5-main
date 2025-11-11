# Texas 42 - Architecture Orientation

**New to the codebase?** This document provides the mental models needed to navigate the architecture.

## Related Documentation
- **Vision**: [VISION.md](VISION.md) - Strategic direction and north star outcomes
- **Principles**: [ARCHITECTURE_PRINCIPLES.md](ARCHITECTURE_PRINCIPLES.md) - Design philosophy and mental models
- **Reference**: [CONCEPTS.md](CONCEPTS.md) - Complete implementation reference
- **Deep Dives**:
  - [remixed-855ccfd5.md](remixed-855ccfd5.md) - Multiplayer architecture specification
  - [archive/pure-layers-threaded-rules.md](archive/pure-layers-threaded-rules.md) - RuleSet system implementation (historical)
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
4. **Zero Coupling**: Core engine has no multiplayer/action-transformer awareness
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
│  Room (Orchestrator)            │  ★ COMPOSITION POINT ★
│  - Owns state and context       │  Routes messages, manages lifecycle
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
│  ActionTransformer System       │  Transform actions (what's possible)
└─────────────────────────────────┘
              ↕
┌─────────────────────────────────┐
│  RuleSet System                 │  Provide rules (how execution works)
│  13 GameRules methods           │  Special contracts live here
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

### RuleSets as Lenses
Each ruleset provides a different lens through which to view game rules. Stack lenses to create new game modes. Nello adds a "3-player trick" lens, Plunge adds a "partner leads" lens.

### ActionTransformers as Decorators
ActionTransformers wrap and transform the base game, adding features without modifying core logic. Tournament filters, Speed annotates, OneHand scripts—each wraps the state machine.

### Capabilities as Keys
Each capability unlocks specific functionality. Collect keys to gain more power. `observe-all-hands` unlocks full visibility, `act-as-player` unlocks actions for a seat.

### The Kernel as a Pure Function
Given state and action, always produces same new state. No hidden state or side effects. `newState = f(oldState, action)` always holds.

### A Trust Hierarchy
Server validates, clients display. Clear security boundary: server is authoritative, clients are delegating.

---

## Two-Level Composition (The Key Insight)

**Problem**: Special contracts (nello, splash, plunge, sevens) need to change BOTH what's possible AND how execution works.

**Solution**: Two orthogonal composition surfaces:

```
RULESETS (execution rules) × ACTION_TRANSFORMERS (action transformation) = Game Configuration
```

### Level 1: RuleSets → Execution Rules

**Purpose**: Define HOW the game executes (who acts, when tricks complete, how winners determined)

**Interface**: `GameRules` - 13 pure methods grouped into:
- **WHO** (3): getTrumpSelector, getFirstLeader, getNextPlayer
- **WHEN** (2): isTrickComplete, checkHandOutcome
- **HOW** (2): getLedSuit, calculateTrickWinner
- **VALIDATION** (3): isValidPlay, getValidPlays, isValidBid
- **SCORING** (3): getBidComparisonValue, isValidTrump, calculateScore

**Composition**: RuleSets override only what differs from base. Compose via reduce:
```typescript
const rules = composeRules([baseRuleSet, nelloRuleSet, splashRuleSet]);
// nelloRuleSet.isTrickComplete overrides base (3 plays not 4)
// Other rules pass through unchanged
```

**Example**: Nello partner sits out → override `isTrickComplete` to return true at 3 plays instead of 4.

### Level 2: ActionTransformers → Action Transformation

**Purpose**: Transform WHAT actions are possible (filter, annotate, script, replace)

**Interface**: `ActionTransformer = (StateMachine) → StateMachine` where `StateMachine = (state) => GameAction[]`

**Operations**:
- **Filter**: Remove actions (tournament removes special bids)
- **Annotate**: Add metadata (hints, autoExecute flags)
- **Script**: Inject actions (oneHand scripts bidding)
- **Replace**: Swap action types (oneHand replaces score-hand with end-game)

**Composition**: ActionTransformers wrap the state machine via reduce:
```typescript
const composed = applyActionTransformers(baseStateMachine, [tournament, oneHand]);
// tournament filters, then oneHand scripts
```

### The Composition Point (Room Constructor)

**ONE place** where everything composes:

```typescript
// 1. RuleSets provide rules
const rulesets = [baseRuleSet, ...getEnabledRuleSets(config)];
const rules = composeRules(rulesets);

// 2. Thread rulesets through base state machine
const baseWithRuleSets = (state) => getValidActions(state, rulesets, rules);

// 3. ActionTransformers transform the actions
const getValidActionsComposed = applyActionTransformers(baseWithRuleSets, transformerConfigs);

// 4. Create ExecutionContext (groups the triple together)
this.ctx = Object.freeze({
  rulesets: Object.freeze(rulesets),
  rules,
  getValidActions: getValidActionsComposed
});

// 5. Use context everywhere
// buildKernelView(state, forPlayerId, this.ctx, metadata)
// executeKernelAction(mpState, playerId, action, this.ctx)
```

**Result**: RuleSets change execution semantics, ActionTransformers change action availability. Executors have ZERO conditional logic - they call `ctx.rules.method()` and trust the result. ExecutionContext groups the always-traveling-together parameters (rulesets, rules, getValidActions) into a single immutable object.

---

## File Map

**Core Engine** (pure utilities):
- `src/game/core/gameEngine.ts` - State machine, action generation
- `src/game/core/actions.ts` - Action executors (thread rules through)
- `src/game/core/rules.ts` - Pure game rules (suit following, trump)
- `src/game/types.ts` - Core types (GameState, GameAction, etc)

**RuleSet System** (execution semantics):
- `src/game/rulesets/types.ts` - GameRules interface (13 methods), GameRuleSet
- `src/game/rulesets/compose.ts` - composeRules() - reduce pattern
- `src/game/rulesets/base.ts` - Standard Texas 42
- `src/game/rulesets/{nello,splash,plunge,sevens}.ts` - Special contracts
- `src/game/rulesets/utilities.ts` - getEnabledRuleSets() - config → rulesets

**ActionTransformer System** (action transformation):
- `src/game/action-transformers/registry.ts` - applyActionTransformers() - composition
- `src/game/action-transformers/{tournament,oneHand}.ts` - ActionTransformer implementations

**Multiplayer** (authorization, visibility):
- `src/game/multiplayer/types.ts` - MultiplayerGameState, PlayerSession, Capability
- `src/game/multiplayer/authorization.ts` - authorizeAndExecute(), filterActionsForSession()
- `src/game/multiplayer/capabilityUtils.ts` - getVisibleStateForSession()
- `src/game/multiplayer/capabilities.ts` - Standard capability builders

**Server** (orchestration + authority):
- `src/server/Room.ts` - Room orchestrator (combines server + kernel)
- `src/kernel/kernel.ts` - Pure helper functions (executeKernelAction, buildKernelView)
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
| **GameRuleSet** | Rule overrides | Compose via reduce pattern |
| **ActionTransformer** | Action transformer | Wrap state machine functions |
| **Capability** | Permission token | Replace identity with permissions |
| **ExecutionContext** | Composed configuration | Bundles rulesets + rules + actions |
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

**Key**: Each rule gets `prev` parameter - delegate to previous ruleset or override. Compose via reduce.

### GameRuleSet

RuleSet that overrides specific rules and/or transforms actions:

```typescript
interface GameRuleSet {
  name: string;
  rules?: Partial<GameRules>;  // Override execution behavior
  getValidActions?: (state, prev) => GameAction[];  // Transform actions
}
```

**Pattern**: Check state, return override or pass through `prev`.

### ActionTransformer

Function that transforms the state machine:

```typescript
type StateMachine = (state: GameState) => GameAction[];
type ActionTransformer = (base: StateMachine) => StateMachine;
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
  - Find session by playerId
  - Get all valid actions via composed state machine
  - Filter by session capabilities
  - Execute: executeAction(coreState, action, ctx.rules)
  - Process auto-execute actions
  ↓
Room.notifyListeners():
  - For each subscriber (with perspective)
  - buildKernelView(mpState, playerId, ctx, metadata) → filter state + build view
  - Build GameView { state, transitions, metadata }
  ↓
Room broadcasts GameView via Transport
  ↓
Transport.send() → connection.onMessage(STATE_UPDATE)
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

### Add a New RuleSet (Special Contract)

1. Create `src/game/rulesets/myContract.ts`:
```typescript
export const myContractRuleSet: GameRuleSet = {
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

2. Add to `src/game/rulesets/utilities.ts:getEnabledRuleSets()`:
```typescript
if (config.enableMyContract) rulesets.push(myContractRuleSet);
```

3. Add config flag to `GameConfig` type

**That's it**. No changes to core executors.

### Add a New ActionTransformer

1. Create `src/game/action-transformers/myTransformer.ts`:
```typescript
export const myTransformer: TransformerFactory = (config) => (base) => (state) => {
  const actions = base(state);
  // Filter, annotate, script, or replace
  return modifiedActions;
};
```

2. Register in `src/game/action-transformers/registry.ts`

3. Use in config: `{ actionTransformer: { type: 'my-transformer' } }`

### Enable a Feature

Most features are config flags:
```typescript
const config: GameConfig = {
  enableNello: true,      // Adds nello ruleset
  enableSplash: true,     // Adds splash ruleset
  enablePlunge: true,     // Adds plunge ruleset
  enableSevens: true,     // Adds sevens ruleset
  actionTransformer: { type: 'tournament' }  // Applies tournament transformer
};
```

Flags → getEnabledRuleSets() → compose in Room constructor.

---

## Architectural Invariants

1. **Pure State Storage**: Room stores unfiltered GameState, filters on-demand per request
2. **Server Authority**: Client trusts server's validActions list, never refilters
3. **Capability-Based Access**: Permissions via capability tokens, not identity checks
4. **Single Composition Point**: Room constructor is ONLY place rulesets/action-transformers compose
5. **Zero Coupling**: Core engine has zero knowledge of multiplayer/action-transformers/special contracts
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
| Adding game logic to Room | Put it in pure helpers or RuleSets |
| Filtering state at storage | Store unfiltered, filter on-demand per request |
| Client-side validation | Trust server's validActions list completely |
| Deep coupling between rulesets | Keep rulesets focused on single responsibility |

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
| Wrong game behavior | RuleSet implementation, `GameRules` methods |
| State not updating | `executeAction`, check if action is authorized |
| UI not reactive | Svelte stores, `ViewProjection` computation |
| Message not routing | `Room.handleMessage()`, Transport layer |

### Understanding the Code

1. **Start with types** - The architecture is type-driven. Follow these key types:
   - `GameRules` → Understand execution semantics
   - `GameRuleSet` → See how rules compose
   - `ActionTransformer` → Learn action transformation
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
