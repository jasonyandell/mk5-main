# Texas 42 - Pure Functional ActionTransformer Composition Handoff

**⚠️ HISTORICAL DOCUMENT**: This document uses legacy terminology. "Layer" is now "RuleSet", "Variant" is now "ActionTransformer". Content preserved for reference.

**Created**: 2025-10-20
**Branch**: mk8
**Status**: Architecture designed, ready for implementation (terminology updated 2025)

---

## Executive Summary

We're refactoring the game engine to enable pure functional composition of variants, allowing structural variants like nello (3-player tricks) that are currently impossible. The key insight: **unified signature** where every method accepts `(state, previousResult) => result`, enabling clean pipeline composition.

---

## The Core Problems

### Problem 1: Consensus Pollutes Core Game State

**Current**: Consensus system uses player agreement for trick completion:
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

**Why it's wrong**:
1. **Not game rules** - trick completes when 4th domino played, not when players agree
2. **Pollutes action history** - breaks deterministic replay
3. **Hardcoded player counts** - `if (consensus.size === 4)` breaks structural variants
4. **Conflates concerns** - mixes game rules (when trick ends) with UI sync (pause for discussion)

**Solution**:
- Remove consensus from core entirely
- Make trick/hand completion automatic
- Add separate CoordinationManager for UI pausing (outside core state)

### Problem 2: Variants Can't Modify Structural Behavior

**Current**: Variants transform action lists only:
```typescript
type Variant = (base: StateMachine) => StateMachine
type StateMachine = (state: GameState) => GameAction[]
```

**Why it's limiting**:
- Functions like `getTrickWinner()`, `getNextPlayer()` imported directly
- No composition point for structural changes
- Nello needs 3-player tricks, but tricks are hardcoded to 4
- Would require duplicating entire execution logic

**Example - nello requirements**:
```
- Partner sits out (needs shouldSkipPlayer)
- 3-player tricks, not 4 (needs getTrickSize, isTrickComplete)
- No trump declared (needs getTrumpForTrick)
- Doubles form own suit (needs getSuitOfDomino)
```

None of these are possible with current architecture!

---

## The Solution: Pure Functional Composition

### Core Insight: Unified Signature

Every method accepts `(state, previousResult) => result`:

```typescript
// Universal pattern
type Transform<T> = (state: GameState, prev: T) => T

// Examples
getValidActions: (state: GameState, prev: GameAction[]) => GameAction[]
getTrickSize: (state: GameState, prev: number) => number
shouldSkipPlayer: (state: GameState, id: number, prev: boolean) => boolean
```

### Why This Works

**Base generates** (ignores prev):
```typescript
baseGetValidActions = (state, prev) => {
  // prev is [], generate fresh from state
  return generateActionsFromRules(state)
}

baseGetTrickSize = (state, prev) => 4  // Always 4
```

**Variants transform** (use or ignore prev):
```typescript
// Nello overrides
nelloGetTrickSize = (state, prev) => 3  // Ignore prev, always 3

// Tournament filters
tournamentGetValidActions = (state, prev) => {
  // Use prev, filter it
  return prev.filter(a => !isSpecialBid(a))
}

// Speed annotates
speedGetValidActions = (state, prev) => {
  // Use prev, add metadata
  return prev.map(a => prev.length === 1 ? {...a, autoExecute: true} : a)
}
```

### Composition is Pure Pipeline

```typescript
const composed = (state) => {
  return [base, nello, tournament, speed].reduce(
    (result, variant) => variant.method(state, result),
    initialValue
  )
}

// For getValidActions:
[] → base → [actions] → nello → [actions] → tournament → [filtered] → speed → [annotated]

// For getTrickSize:
null → base → 4 → nello → 3 (last non-null wins)
```

---

## Architecture Design

### GameEngine Interface

```typescript
interface GameRules {
  // Structural queries (enables nello!)
  getTrickSize(state: GameState, prev: number): number
  getActivePlayerCount(state: GameState, prev: number): number
  shouldSkipPlayer(state: GameState, id: number, prev: boolean): boolean
  isTrickComplete(state: GameState, prev: boolean): boolean

  // Game logic
  getTrumpForTrick(state: GameState, prev: TrumpSelection): TrumpSelection
  getSuitOfDomino(state: GameState, domino: Domino, prev: number): number
  calculateTrickWinner(state: GameState, trick: Play[], prev: number): number
  getNextPlayer(state: GameState, current: number, prev: number): number
}

interface GameEngine {
  rules: GameRules
  getValidActions(state: GameState, prev: GameAction[]): GameAction[]
  executeAction(state: GameState, action: GameAction, prev: GameState): GameState
}
```

### Variant Definition

```typescript
interface Variant {
  // Any method can be overridden with unified signature
  getValidActions?: (state: GameState, prev: GameAction[]) => GameAction[]
  executeAction?: (state: GameState, action: GameAction, prev: GameState) => GameState

  rules?: {
    getTrickSize?: (state: GameState, prev: number) => number
    shouldSkipPlayer?: (state: GameState, id: number, prev: boolean) => boolean
    isTrickComplete?: (state: GameState, prev: boolean) => boolean
    getTrumpForTrick?: (state: GameState, prev: TrumpSelection) => TrumpSelection
    getSuitOfDomino?: (state: GameState, domino: Domino, prev: number) => number
    // ... all rules
  }
}
```

### Nello Example

```typescript
const nello: Variant = {
  rules: {
    // Override: always 3 (ignore prev)
    getTrickSize: (state, prev) => 3,
    getActivePlayerCount: (state, prev) => 3,

    // Override: partner sits out
    shouldSkipPlayer: (state, id, prev) => {
      const partner = (state.winningBidder + 2) % 4
      return id === partner
    },

    // Override: complete when 3 played
    isTrickComplete: (state, prev) =>
      state.currentTrick.length === 3,

    // Override: no trump in nello
    getTrumpForTrick: (state, prev) =>
      ({ type: 'no-trump' }),

    // Override: doubles form own suit (standard nello)
    getSuitOfDomino: (state, domino, prev) =>
      domino.high === domino.low ? 7 : domino.high
  }
}
```

### Tournament Example

```typescript
const tournament: Variant = {
  // Transform: filter special bids
  getValidActions: (state, prev) =>
    prev.filter(a =>
      a.type !== 'bid' || !['nello', 'splash', 'plunge'].includes(a.bid)
    )
}
```

### Speed Example

```typescript
const speed: Variant = {
  // Transform: add autoExecute metadata
  getValidActions: (state, prev) =>
    prev.map(a =>
      prev.length === 1 ? {...a, autoExecute: true} : a
    )
}
```

### Composition Function

```typescript
function composeVariants(
  base: GameEngine,
  variants: Variant[]
): GameEngine {

  // Compose getValidActions
  const getValidActions = (state: GameState) => {
    return [base, ...variants].reduce(
      (actions, variant) =>
        variant.getValidActions?.(state, actions) ?? actions,
      [] as GameAction[]
    )
  }

  // Compose executeAction
  const executeAction = (state: GameState, action: GameAction) => {
    return [base, ...variants].reduce(
      (prevState, variant) =>
        variant.executeAction?.(state, action, prevState) ?? prevState,
      state
    )
  }

  // Compose each rule
  const rules = {} as GameRules
  for (const ruleName of Object.keys(base.rules)) {
    rules[ruleName] = (state, ...args) => {
      const initialValue = getInitialValue(ruleName)  // null, 0, false, etc.

      return [base, ...variants].reduce(
        (prev, variant) =>
          variant.rules?.[ruleName]?.(state, ...args, prev) ?? prev,
        initialValue
      )
    }
  }

  return { rules, getValidActions, executeAction }
}
```

---

## Implementation Plan

### Phase 1: Remove Consensus (Breaking Change)
**Files**: ~10 in src/game/core/

1. Remove `consensus` field from GameState
2. Delete 4 consensus action types
3. Make trick completion automatic in executePlay
4. Make hand progression automatic after 7th trick

### Phase 2: Define GameEngine Interface
**New files**:
- `src/game/core/engine/types.ts`
- `src/game/core/engine/baseEngine.ts`

1. Define GameRules interface with all rule methods
2. Define GameEngine interface
3. Create baseEngine factory with default implementations

### Phase 3: Update Core to Use Rules
**Files**: ~5-10 in src/game/core/

1. Update executeAction to call rules methods instead of hardcoded logic
2. Update getValidActions to call rules methods
3. Thread rules through all core functions

### Phase 4: Rewrite Variant System
**Files**: src/game/variants/

1. Update types.ts with new Variant interface
2. Rewrite registry.ts with composition function
3. Rewrite all existing variants (tournament, speed, hints, oneHand)
4. Implement nello variant as proof

### Phase 5: Update GameHost
**Files**: src/server/game/GameHost.ts, multiplayer layer

1. Compose engine at initialization
2. Use composed engine for all operations
3. Remove old variant composition logic

### Phase 6: Add Coordination Layer (Optional)
**New files**: src/game/multiplayer/coordination/

1. Create CoordinationManager class
2. Track player readiness separate from core state
3. Detect pause points (trick complete, hand complete)
4. Add playerReady() to GameClient interface

### Phase 7: Update Tests
**Files**: ~20-30 test files

1. Remove consensus-related tests
2. Add variant composition tests
3. Add nello integration tests
4. Update existing tests for automatic completion

---

## Key Technical Decisions

### Decision 1: Unified `(state, prev) => result` Signature

**Rationale**:
- Eliminates replace/transform bifurcation
- Simple composition via reduce
- Variants can choose to use or ignore prev
- Consistent pattern everywhere

**Trade-off**: Some methods ignore prev (but that's acceptable)

### Decision 2: Remove Consensus from Core

**Rationale**:
- Tricks complete automatically (when 4th/3rd domino played)
- Action history is pure game events
- No hardcoded player counts
- UI pausing is separate concern (coordination layer)

**Trade-off**: Need new coordination mechanism for multiplayer pausing

### Decision 3: Make All Rules Composable

**Rationale**:
- Enables structural variants (nello, custom player counts, etc.)
- Each variant can override or passthrough
- Composition happens at init time, not runtime

**Trade-off**: Slightly more complex interface

---

## Success Criteria

✅ Consensus removed from core
✅ GameEngine interface with composable rules
✅ All variants use unified signature
✅ Nello variant works (3-player tricks, no trump, partner sits out)
✅ Tournament/speed/hints variants still work
✅ Action history is clean (no consensus actions)
✅ All tests pass
✅ Composition is pure functional pipeline

---

## Open Questions for Next Session

1. Should coordination layer pause after every trick or be configurable?
2. Do AI clients auto-signal ready immediately or add delay?
3. Should getNextStates also get the `(state, prev)` signature?
4. How to handle variant serialization (storing which variants are active)?

---

## References

- Vision document: docs/remixed-855ccfd5.md (§5 on variants)
- Current game state: src/game/types.ts:150-183
- Current variant system: src/game/variants/
- GameHost composition: src/server/game/GameHost.ts:81-91

---

**The architecture is elegant, the math checks out (monoid composition!), and the path forward is clear. Time to implement!** 🎯
