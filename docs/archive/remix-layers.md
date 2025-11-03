# Texas 42 - Pure Functional RuleSet Composition Architecture

**⚠️ HISTORICAL DOCUMENT**: This document uses legacy terminology. "Layer" is now "RuleSet", "Variant" is now "ActionTransformer". Content preserved for reference.

**Created**: 2025-10-20
**Branch**: mk8
**Status**: Architecture finalized, ready for implementation (terminology updated 2025)

---

## Executive Summary

We're refactoring the game engine to enable pure functional composition of rule layers, allowing official Texas 42 rules like nello (3-player tricks, partner sits out) that are currently impossible. The key insight: **unified `(state, prev) => result` signature** where every method accepts previous result and returns new result, enabling clean pipeline composition via reduce.

**Critical Decision**: Use **runtime state checks** (`isNello(state)`) rather than dynamic recomposition to preserve event sourcing purity and time-travel debugging.

---

## Context: Why This Refactoring?

### The Fundamental Problem

**Nello is part of official Texas 42 rules** (see `docs/rules.md` §8.1.1), not an optional variant:

```
Nel-O (Nello, Low, Low-boy)
Requirements: Must bid at least 1 mark
Objective: Bidder must lose every trick
Rules:
- Partner sits out with dominoes face-down
- No trump suit declared
- Various doubles treatments:
  - Standard: Doubles form own suit
```

**Current architecture cannot support nello** because:
- Trick size hardcoded to 4 players
- Functions like `getTrickWinner()`, `getNextPlayer()` imported directly
- No composition point for structural changes
- Would require duplicating entire execution logic

### What Nello Requires

```
✓ Partner sits out (shouldSkipPlayer)
✓ 3-player tricks, not 4 (getTrickSize, isTrickComplete)
✓ No trump declared (getTrumpForTrick)
✓ Doubles form own suit (getSuitOfDomino)
✓ Custom trick winner logic (calculateTrickWinner)
```

**None of these are possible with current architecture.**

### Secondary Problem: Consensus Pollutes Core State

**Current consensus system**:
```typescript
// GameState has consensus field
consensus: {
  completeTrick: Set<number>,  // Who agreed to complete trick
  scoreHand: Set<number>        // Who agreed to score hand
}

// 4 consensus action types pollute history
'agree-complete-trick'
'agree-score-hand'
'complete-trick'
'score-hand'
```

**Why this is wrong**:
1. **Not game rules** - tricks complete when 4th domino played, not when players agree
2. **Breaks deterministic replay** - action history includes UI coordination
3. **Hardcoded player counts** - `if (consensus.size === 4)` breaks nello
4. **Conflates concerns** - mixes game rules with multiplayer UX

**Decision**: Remove consensus from core entirely. Make trick/hand completion automatic based on game rules. (Multiplayer coordination is explicitly out of scope for this phase.)

---

## The Solution: Pure Functional Layer Composition

### Core Insight: Unified Signature

Every method accepts `(state, previousResult) => result`:

```typescript
// Universal pattern
type LayerMethod<T> = (state: GameState, prev: T) => T

// Examples
getValidActions: (state: GameState, prev: GameAction[]) => GameAction[]
getTrickSize: (state: GameState, prev: number) => number
shouldSkipPlayer: (state: GameState, id: number, prev: boolean) => boolean
getSuitOfDomino: (state: GameState, domino: Domino, prev: number) => number
```

### Why This Works: Three Layer Patterns

**1. Base generates** (ignores prev):
```typescript
baseGetValidActions = (state, prev) => {
  // prev is [], generate fresh from state
  return generateActionsFromRules(state)
}

baseGetTrickSize = (state, prev) => 4  // Always 4
```

**2. Layers transform** (use prev):
```typescript
// Tournament filters actions
tournamentGetValidActions = (state, prev) =>
  prev.filter(a => a.type !== 'bid' || !isSpecialBid(a))

// Speed annotates actions
speedGetValidActions = (state, prev) =>
  prev.map(a => prev.length === 1 ? {...a, autoExecute: true} : a)
```

**3. Layers override** (ignore prev):
```typescript
// Nello overrides trick size
nelloGetTrickSize = (state, prev) =>
  state.winningBid?.bid === 'nello' ? 3 : prev

// Nello overrides trump
nelloGetTrumpForTrick = (state, prev) =>
  state.winningBid?.bid === 'nello' ? { type: 'no-trump' } : prev
```

### Composition is Pure Pipeline

```typescript
const composed = (state) => {
  return [base, nello, tournament, speed].reduce(
    (result, layer) => layer.method(state, result),
    initialValue
  )
}

// For getValidActions:
[] → base → [actions] → nello → [actions] → tournament → [filtered] → speed → [annotated]

// For getTrickSize:
null → base → 4 → nello → 3 (last non-null wins)
```

**This is monoid composition** - associative, with identity element.

---

## Architecture Design

### GameRules Interface

```typescript
interface GameRules {
  // Structural queries (enables nello!)
  getTrickSize(state: GameState, prev: number): number
  getActivePlayerCount(state: GameState, prev: number): number
  shouldSkipPlayer(state: GameState, playerId: number, prev: boolean): boolean
  isTrickComplete(state: GameState, prev: boolean): boolean

  // Game logic
  getTrumpForTrick(state: GameState, prev: TrumpSelection): TrumpSelection
  getSuitOfDomino(state: GameState, domino: Domino, prev: number): number
  calculateTrickWinner(state: GameState, trick: Play[], prev: number): number
  getNextPlayer(state: GameState, current: number, prev: number): number

  // Domino comparison (needed for "doubles low" variation)
  compareDominoes(state: GameState, d1: Domino, d2: Domino, suit: number, prev: number): number
}
```

### GameEngine Interface

```typescript
interface GameEngine {
  rules: GameRules
  getValidActions(state: GameState, prev: GameAction[]): GameAction[]
  executeAction(state: GameState, action: GameAction, prev: GameState): GameState
}
```

### Layer Definition

```typescript
interface GameLayer {
  name: string  // 'nello', 'tournament', 'speed', etc.

  // Any method can be overridden with unified signature
  getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[]
  executeAction?: (state: GameState, action: GameAction, prev: GameState) => GameState

  rules?: {
    getTrickSize?: (state: GameState, prev: number) => number
    shouldSkipPlayer?: (state: GameState, id: number, prev: boolean) => boolean
    isTrickComplete?: (state: GameState, prev: boolean) => boolean
    getTrumpForTrick?: (state: GameState, prev: TrumpSelection) => TrumpSelection
    getSuitOfDomino?: (state: GameState, domino: Domino, prev: number) => number
    calculateTrickWinner?: (state: GameState, trick: Play[], prev: number) => number
    // ... all rules
  }
}
```

**Terminology**: We use "Layer" instead of "Variant" because:
- Vision uses "variant" for action transforms only
- Texas 42 has "special contracts" (nello, plunge, splash) that modify rules
- "Layer" covers both: rule modifiers AND action transforms
- Simpler, more accurate terminology

---

## Critical Architectural Decision: Runtime State Checks vs. Dynamic Recomposition

### The Question

Should we:
1. **Compose once at init, layers check state** (`isNello(state)` conditionals)
2. **Recompose engine when bid wins** (nello bid → recompose with nello layer)

### The Analysis

**Option 1: Runtime Conditionals**
```typescript
// Compose once with ALL layers
const engine = compose([base, nello, plunge, tournament])

// Layers check state
nelloLayer.getTrickSize = (state, prev) =>
  state.winningBid?.bid === 'nello' ? 3 : prev
```

**Event Sourcing:**
```typescript
// Perfect!
function replayActions(config, actions) {
  const engine = composeEngine(config)  // Once
  let state = initialState

  for (const action of actions) {
    state = executeAction(engine, state, action)
  }

  return state  // Pure data, serializable
}

// Time-travel: one line
const stateAt50 = replayActions(config, actions.slice(0, 50))

// Serialization: works perfectly
JSON.stringify(state)  // ✅ No function references
```

**Analysis:**
- ✅ Engine composed once, never changes
- ✅ State is pure data (no functions)
- ✅ Replay is trivial
- ✅ Time-travel is one line
- ✅ Serialization works
- ❌ Runtime checks (performance cost?)

**Option 2: Dynamic Recomposition**
```typescript
// Recompose when bid wins
executeAction(state, { type: 'declare-trump' }) {
  const bidLayers = getBidLayers(state.winningBid)
  const newEngine = compose([configEngine, ...bidLayers])

  // Problem: Where does newEngine go?
  // Can't store in state (functions not serializable)
  // Store in GameHost? How to replay? Need memoization?
}
```

**Problems:**
1. **Storage**: Functions don't serialize → breaks event sourcing
2. **Replay**: Must recompose during replay → need "when to recompose" logic
3. **Performance**: Need memoization → WeakMap cache, complexity
4. **Conceptual**: State-dependent functions → impure

**Analysis:**
- ❌ Engine storage unclear
- ❌ Replay needs recomposition logic
- ❌ Performance requires memoization
- ❌ State-dependent functions (impure)
- ✅ No runtime conditionals

### The Decision: Runtime Conditionals (Option 1)

**Why:**

1. **Event Sourcing Purity**: State remains pure data with zero function references
2. **Replay Simplicity**: `engine = compose(config); replay(engine, actions)` - one line
3. **Time-Travel**: `replayActions(config, actions.slice(0, N))` - trivial
4. **Serialization**: `JSON.stringify(state)` - works perfectly
5. **Determinism**: Same config + same actions = same state, guaranteed
6. **Conceptual Simplicity**: Engine doesn't change, state changes

**Performance Reality Check:**

```typescript
isNello(state) { return state.winningBid?.bid === 'nello' }
```

Cost per check: ~3 nanoseconds (property access + string comparison)
Checks per game: ~3,000 (300 actions × 10 rules)
**Total cost: 9 microseconds per game** ← Negligible!

**The price of runtime checks is VASTLY lower than the conceptual/implementation weight of recomposition.**

### Key Architectural Insight

**Nello is STATE-dependent, not CONFIG-dependent.**

```typescript
// Configuration (set at game start, never changes)
const gameConfig = {
  tournament: true,           // Immutable
  doublesAsOwnSuit: false,    // Immutable
  targetScore: 7              // Immutable
}

// State (changes during game)
const gameState = {
  winningBid: { bid: 'nello' },  // Changes every hand!
  phase: 'playing',              // Changes constantly
  currentPlayer: 2               // Changes constantly
}
```

**Nello rules activate based on `state.winningBid`**, which **changes every hand**. If we recomposed on bid changes, we'd recompose 7+ times per game, track WHEN to recompose during replay, and manage engine lifecycle. Runtime checks eliminate all this complexity.

---

## Layer Examples

### Base Layer (Default Texas 42)

```typescript
const baseLayer: GameLayer = {
  name: 'base',

  getValidActions: (state, prev) => {
    // Ignore prev (empty array), generate from rules
    return generateValidActionsFromState(state)
  },

  executeAction: (state, action, prev) => {
    // Ignore prev, execute pure state transition
    return coreExecuteAction(state, action)
  },

  rules: {
    getTrickSize: (state, prev) => 4,
    getActivePlayerCount: (state, prev) => 4,
    shouldSkipPlayer: (state, id, prev) => false,
    isTrickComplete: (state, prev) => state.currentTrick.length === 4,
    getTrumpForTrick: (state, prev) => state.trump,
    getSuitOfDomino: (state, domino, prev) => domino.high,  // Higher end
    calculateTrickWinner: (state, trick, prev) => {
      // Standard 4-player trick logic
      return standardTrickWinner(state, trick)
    },
    getNextPlayer: (state, current, prev) => (current + 1) % 4
  }
}
```

### Nello Layer (3-player, partner sits out)

```typescript
const nelloLayer: GameLayer = {
  name: 'nello',

  rules: {
    // All rules check: is this a nello hand?
    getTrickSize: (state, prev) =>
      state.winningBid?.bid === 'nello' ? 3 : prev,

    getActivePlayerCount: (state, prev) =>
      state.winningBid?.bid === 'nello' ? 3 : prev,

    shouldSkipPlayer: (state, id, prev) => {
      if (state.winningBid?.bid === 'nello') {
        const partner = (state.winningBidder + 2) % 4
        return id === partner
      }
      return prev
    },

    isTrickComplete: (state, prev) =>
      state.winningBid?.bid === 'nello' && state.currentTrick.length === 3
        ? true
        : prev,

    getTrumpForTrick: (state, prev) =>
      state.winningBid?.bid === 'nello'
        ? { type: 'no-trump' }
        : prev,

    getSuitOfDomino: (state, domino, prev) => {
      if (state.winningBid?.bid === 'nello') {
        // Doubles form own suit in nello
        return domino.high === domino.low ? 7 : domino.high
      }
      return prev
    },

    calculateTrickWinner: (state, trick, prev) => {
      if (state.winningBid?.bid === 'nello') {
        // Custom nello logic (no trump, doubles as suit 7)
        return calculateNelloTrickWinner(state, trick)
      }
      return prev
    }
  }
}
```

**Note**: All nello rules check `state.winningBid?.bid === 'nello'`. This is the **runtime state check** that makes nello work without recomposition.

### Tournament Layer (Filter special contracts)

```typescript
const tournamentLayer: GameLayer = {
  name: 'tournament',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev

    // Filter out special bids
    return prev.filter(action =>
      action.type !== 'bid' ||
      !['nello', 'splash', 'plunge', 'sevens'].includes(action.bid)
    )
  }
}
```

### Speed Layer (Auto-execute single option)

```typescript
const speedLayer: GameLayer = {
  name: 'speed',

  getValidActions: (state, prev) => {
    if (state.phase !== 'playing') return prev

    const playActions = prev.filter(a => a.type === 'play')

    if (playActions.length === 1) {
      return [{
        ...playActions[0],
        autoExecute: true,
        delay: 300
      }]
    }

    return prev
  }
}
```

### Plunge Layer (4+ doubles, partner declares trump)

```typescript
const plungeLayer: GameLayer = {
  name: 'plunge',

  getValidActions: (state, prev) => {
    if (state.phase === 'bidding') {
      const currentPlayer = state.players[state.currentPlayer]
      const doubleCount = currentPlayer.hand.filter(isDouble).length

      if (doubleCount >= 4) {
        // Add plunge bid option
        return [...prev, { type: 'bid', bid: 'plunge', value: 4 }]
      }
    }
    return prev
  },

  executeAction: (state, action, prev) => {
    if (action.type === 'bid' && action.bid === 'plunge') {
      // Partner must declare trump
      return {
        ...prev,
        phase: 'trump-declaration',
        trumpDeclarer: (action.player + 2) % 4  // Partner
      }
    }
    return prev
  }
}
```

### Sevens Layer (Distance from 7 wins)

```typescript
const sevensLayer: GameLayer = {
  name: 'sevens',

  rules: {
    calculateTrickWinner: (state, trick, prev) => {
      if (state.winningBid?.bid === 'sevens') {
        // Winner = domino closest to 7 total pips
        const distances = trick.map(play =>
          Math.abs(7 - (play.domino.high + play.domino.low))
        )
        const minDistance = Math.min(...distances)
        return trick.findIndex(play =>
          Math.abs(7 - (play.domino.high + play.domino.low)) === minDistance
        )
      }
      return prev
    }
  }
}
```

---

## Composition Function

```typescript
function composeEngine(
  baseLayers: GameLayer[],
  config: GameConfig
): GameEngine {

  // Determine active layers based on config
  const layers = [
    baseLayer,  // Always first

    // Config-based layers
    config.tournament ? tournamentLayer : null,
    config.speed ? speedLayer : null,

    // Bid-based layers (always included, check state)
    nelloLayer,
    plungeLayer,
    splashLayer,
    sevensLayer,

  ].filter(Boolean)

  // Compose getValidActions
  const getValidActions = (state: GameState) => {
    return layers.reduce(
      (actions, layer) =>
        layer.getValidActions?.(state, actions) ?? actions,
      [] as GameAction[]
    )
  }

  // Compose executeAction
  const executeAction = (state: GameState, action: GameAction) => {
    return layers.reduce(
      (prevState, layer) =>
        layer.executeAction?.(state, action, prevState) ?? prevState,
      state
    )
  }

  // Compose each rule
  const rules = {} as GameRules
  const ruleNames: (keyof GameRules)[] = [
    'getTrickSize',
    'getActivePlayerCount',
    'shouldSkipPlayer',
    'isTrickComplete',
    'getTrumpForTrick',
    'getSuitOfDomino',
    'calculateTrickWinner',
    'getNextPlayer',
    'compareDominoes'
  ]

  for (const ruleName of ruleNames) {
    rules[ruleName] = (state, ...args) => {
      const initialValue = getInitialValue(ruleName)  // null, 0, false, etc.

      return layers.reduce(
        (prev, layer) =>
          layer.rules?.[ruleName]?.(state, ...args, prev) ?? prev,
        initialValue
      )
    } as any  // Type gymnastics needed
  }

  return { rules, getValidActions, executeAction }
}
```

### Initial Values by Rule Type

```typescript
function getInitialValue(ruleName: keyof GameRules): any {
  const defaults: Record<string, any> = {
    getTrickSize: null,
    getActivePlayerCount: null,
    shouldSkipPlayer: false,
    isTrickComplete: false,
    getTrumpForTrick: null,
    getSuitOfDomino: null,
    calculateTrickWinner: null,
    getNextPlayer: null,
    compareDominoes: 0
  }

  return defaults[ruleName]
}
```

---

## Implementation Plan

### Phase 1: Remove Consensus ⚠️ BREAKING CHANGE
**Files**: ~10 in `src/game/core/`

1. Remove `consensus` field from GameState type
2. Delete 4 consensus action types:
   - `agree-complete-trick`
   - `agree-score-hand`
   - `complete-trick`
   - `score-hand`
3. Make trick completion automatic in `executeAction` for `play` actions:
   - Check `isTrickComplete(state)` after each play
   - If true, immediately calculate winner and award trick
4. Make hand progression automatic after 7th trick:
   - Check `state.tricks.length === 7` after trick completion
   - If true, immediately transition to scoring phase

**Why this is necessary**: Consensus hardcodes 4-player assumptions and pollutes action history.

### Phase 2: Define GameEngine Interface
**New files**:
- `src/game/core/engine/types.ts` - GameRules, GameEngine, GameLayer interfaces
- `src/game/core/engine/baseLayer.ts` - Base layer with default Texas 42 rules
- `src/game/core/engine/compose.ts` - composeEngine function

1. Define `GameRules` interface with all rule methods
2. Define `GameEngine` interface (`rules`, `getValidActions`, `executeAction`)
3. Define `GameLayer` interface (partial overrides with unified signature)
4. Create `baseLayer` with default 4-player Texas 42 rules
5. Create `composeEngine` function using reduce pipeline

### Phase 3: Update Core to Use Engine
**Files**: ~5-10 in `src/game/core/`

1. Refactor `executeAction` to accept composed engine:
   ```typescript
   executeAction(engine: GameEngine, state: GameState, action: GameAction)
   ```
2. Replace hardcoded logic with engine rule calls:
   ```typescript
   // OLD
   const trickSize = 4
   const winner = calculateTrickWinner(trick, trump)

   // NEW
   const trickSize = engine.rules.getTrickSize(state, null)
   const winner = engine.rules.calculateTrickWinner(state, trick, null)
   ```
3. Thread engine through all core functions
4. Update `getValidActions` to use engine composition

### Phase 4: Create Layer Implementations
**New files**: `src/game/layers/`

1. `nello.ts` - Nello layer (3-player, partner sits out, no trump)
2. `plunge.ts` - Plunge layer (4+ doubles, partner declares trump)
3. `splash.ts` - Splash layer (3+ doubles, similar to plunge)
4. `sevens.ts` - Sevens layer (distance from 7 wins)
5. `tournament.ts` - Tournament layer (filter special contracts)
6. `speed.ts` - Speed layer (auto-execute single option)
7. `registry.ts` - Layer registry mapping names to implementations

### Phase 5: Update GameHost Composition
**Files**: `src/server/game/GameHost.ts`, `src/server/game/createGameAuthority.ts`

1. Compose engine at GameHost initialization:
   ```typescript
   constructor(gameId: string, config: GameConfig, sessions: PlayerSession[]) {
     this.engine = composeEngine([baseLayer], config)
     this.mpState = createMultiplayerGame(gameId, initialState, sessions, config)
   }
   ```
2. Use `this.engine` for all operations:
   ```typescript
   const actions = this.engine.getValidActions(coreState, [])
   const newState = this.engine.executeAction(coreState, action, coreState)
   ```
3. Remove old variant composition logic (replaced by layer composition)

### Phase 6: Update Tests
**Files**: ~20-30 test files

1. Remove consensus-related tests
2. Add layer composition tests:
   - Base layer produces correct defaults
   - Nello layer overrides for nello hands
   - Tournament layer filters correctly
   - Multiple layers compose correctly
3. Add integration tests for special contracts:
   - Nello hand (3-player tricks, partner sits out)
   - Plunge hand (partner declares trump, must win all)
   - Sevens hand (distance from 7 wins)
4. Update existing tests for automatic trick/hand completion

---

## Verification Against Full Official Rules

We verified this architecture against all rules in `docs/rules.md`:

### ✅ Fully Supported

- **Standard Gameplay** (§5) - Base layer
- **Tournament Mode** (§10) - Tournament layer filters special contracts
- **Doubles Variations** (§5.2.2) - `getSuitOfDomino` overrides
- **Nel-O** (§8.1.1) - Nello layer with multi-rule override
- **Plunge** (§8.1.2) - Plunge layer with bid validation + execution override
- **Splash** (§8.1.4) - Similar to plunge
- **Sevens** (§8.1.4) - Custom `calculateTrickWinner`
- **Forced Bidding** (§4.4.2) - Filter/add actions based on state

### 🟡 Needs Minor Extension

**"Doubles Low" variation**: Requires new `compareDominoes` rule method (doubles rank lowest in suit). Added to `GameRules` interface.

### ⏭️ Out of Scope (Multiplayer/UI)

- Tournament time limits (§10.1.2)
- Communication rules (§9)
- Player conduct (§9)

**Conclusion**: The `(state, prev) => result` layer composition architecture handles **all documented Texas 42 game rules**.

---

## Key Architectural Invariants

1. **Pure Functions Everywhere**: All layers are pure functions - no side effects
2. **Engine Composed Once**: Composition happens at GameHost init, never changes
3. **State is Pure Data**: No function references in state - perfect serialization
4. **Deterministic Replay**: `replayActions(config, actions)` is trivial
5. **Layer Order Matters**: Layers apply left-to-right via reduce
6. **Last Non-Null Wins**: For overrides, rightmost layer providing value wins
7. **Transforms Use Prev**: Layers can use or ignore previous result as needed

---

## Migration Path

### Breaking Changes

1. **Consensus removal**: Existing games with consensus actions in history won't replay correctly
2. **Action history format**: Old URLs with consensus actions will break

### Migration Strategy

1. **Version game state**: Add `version: 2` to new GameState format
2. **Replay adapter**: Detect old URLs, strip consensus actions, replay with v2
3. **Documentation**: Clear communication about URL format change

### Backward Compatibility

- Core game logic remains identical (same rules, scoring, bidding)
- Only internal state structure changes
- UI components mostly unchanged (consensus UI can be removed)

---

## Success Criteria

✅ Consensus removed from core state and action types
✅ GameEngine interface with composable rules defined
✅ All layers use unified `(state, prev) => result` signature
✅ Nello layer works (3-player tricks, no trump, partner sits out, doubles as own suit)
✅ Tournament/speed layers still work (action filtering/annotation)
✅ Plunge/Splash/Sevens layers work (bid validation + custom logic)
✅ Action history is clean (no consensus actions)
✅ Composition is pure functional pipeline via reduce
✅ Event sourcing unchanged (replay, time-travel, serialization all work)
✅ All tests pass

---

## Open Questions for Implementation

1. **Layer registration**: How to map layer names to implementations for config?
   - Suggestion: Registry pattern `layerRegistry['nello'] = nelloLayer`

2. **GameState type changes**: Should we add `version: number` field now?
   - Suggestion: Yes, start at `version: 2` for new format

3. **Helper functions**: Should `isNello(state)`, `isPlunge(state)` be centralized?
   - Suggestion: Yes, `src/game/layers/helpers.ts` with all bid checks

4. **Error handling**: What if layer throws? Should composition catch?
   - Suggestion: Let errors propagate, catch at GameHost boundary

5. **Performance monitoring**: Should we measure layer composition cost?
   - Suggestion: Add optional timing logs in dev mode only

---

## References

- **Vision document**: `docs/remixed-855ccfd5.md` (§5 on variants)
- **Onboarding**: `docs/GAME_ONBOARDING.md` (current architecture)
- **Official rules**: `docs/rules.md` (complete Texas 42 specification)
- **Current game state**: `src/game/types.ts:150-183`
- **Current variant system**: `src/game/variants/` (to be replaced)
- **GameHost composition**: `src/server/game/GameHost.ts:81-91` (to be updated)

---

## Summary

This architecture achieves:

- ✅ **Pure functional composition** via `(state, prev) => result` unified signature
- ✅ **Official rules support** including nello, plunge, splash, sevens
- ✅ **Event sourcing preservation** with trivial replay and time-travel
- ✅ **Deterministic behavior** same config + same actions = same state
- ✅ **Clean action history** no consensus pollution
- ✅ **Extensibility** easy to add new layers without touching core
- ✅ **Performance** negligible runtime cost (~microseconds per game)
- ✅ **Simplicity** simpler than dynamic recomposition alternative

**The path forward is clear. The math is sound (monoid composition). Time to implement.** 🎯
