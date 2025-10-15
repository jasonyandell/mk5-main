# Handoff: Pure Functional Architecture Migration

**Date:** 2025-10-14
**Branch:** mk8
**Status:** ✅ Player-based architecture complete, 🎯 Ready for functional combinator migration

---

## Executive Summary

The player-based architecture migration (h-session2.md) is **COMPLETE**:
- ✅ No compilation errors (typecheck passes)
- ✅ All 647 unit tests passing
- ✅ Clean `PlayerSession` type with string playerIds
- ✅ Triple indirection eliminated

**Next critical task:** Migrate variant system from imperative hooks to **pure functional combinators**. This is essential for a game implementation - everything must flow from the functional core.

---

## Current Architecture State

### What We Have (Good ✅)

#### 1. Pure Core Functions
```typescript
// src/game/core/actions.ts
executeAction: (GameState, GameAction) => GameState  // ✅ Pure!

// src/game/core/gameEngine.ts
getValidActions: (GameState) => GameAction[]         // ✅ Pure!
```

These are the foundation. They work correctly and are already pure functions.

#### 2. Clean Player Model
```typescript
// src/game/multiplayer/types.ts
interface PlayerSession {
  playerId: string              // "player-0", "ai-1" - stable identity
  playerIndex: 0 | 1 | 2 | 3   // Seat position
  controlType: 'human' | 'ai'   // Who controls this player
}
```

No more `clientId ↔ sessionId ↔ playerId` triple indirection. Clean and direct.

#### 3. Authorization Layer
```typescript
// src/game/multiplayer/authorization.ts
authorizeAndExecute: (
  MultiplayerGameState,
  ActionRequest
) => Result<MultiplayerGameState>
```

Works exactly as specified in vision document.

#### 4. AI as External Actor
```typescript
// src/game/multiplayer/AIClient.ts
class AIClient {
  // Connects via protocol, no special server privileges
  // Uses same IGameAdapter as human clients
}
```

AI is not baked into the server - it's an independent client.

### What We Have (Wrong ❌)

#### 1. Imperative Variant System
```typescript
// src/server/game/VariantRegistry.ts
class VariantRegistry {
  static initialize(state, variant)      // Hook 1
  static afterAction(state, ...)         // Hook 2
  static checkGameEnd(state, ...)        // Hook 3
  static updateVariantState(...)         // Hook 4
}
```

**Problem:** This is imperative lifecycle hooks scattered across codebase. NOT compositional.

**Location:** `src/server/game/VariantRegistry.ts` - 418 lines of imperative hooks

#### 2. Variant Config External to State
```typescript
// Current: Variant stored in GameHost, not in GameState
class GameHost {
  private variant: GameVariant | undefined
  private variantState: VariantState = {}
}
```

**Problem:** Can't replay games from action history alone. State is incomplete.

**Breaking issue:** Event sourcing doesn't work - need external variant config to reconstruct state.

#### 3. No Capability System
Vision document Section 4 defines composable capability tokens for:
- What players can do (action authorization)
- What players can see (visibility filtering)
- Standard sets: human, AI, spectator, coach, tutorial

**Current state:** Only basic `controlType: 'human' | 'ai'` exists. No fine-grained capabilities.

---

## The Vision: Pure Functional Composition

### Core Principle (from vision document)

> **Pure Function Composition**: All game logic expressed as pure functions that compose naturally:
> ```
> GameState → Pure Function → GameState → Pure Function → GameState
> ```

> **Variants as Transformers**: Game rule modifications are functions that transform the state machine, stored in game state, and applied at runtime.

### How It Should Work

```typescript
// 1. A state machine is a FUNCTION
type StateMachine = (state: GameState) => GameAction[]

// 2. A variant is a FUNCTION TRANSFORMER
type Variant = (baseMachine: StateMachine) => StateMachine

// 3. Composition is just function composition
const rules = variant3(variant2(variant1(baseRules)))
```

**Why this matters for games:**
- ✅ **Compositional** - Variants stack naturally via function composition
- ✅ **Testable** - Each variant is an isolated pure function
- ✅ **Event Sourceable** - Replay from actions = perfect reconstruction
- ✅ **Correct by Construction** - Type system enforces correctness
- ✅ **No Side Effects** - Can't "forget" to apply a variant
- ✅ **Parallel Universes** - Can test different rule combinations trivially

---

## The Architecture Gap

| Component | Vision | Current | Status |
|-----------|--------|---------|--------|
| Core executeAction | Pure function | ✅ Pure function | ✅ **ALIGNED** |
| Core getValidActions | Pure function | ✅ Pure function | ✅ **ALIGNED** |
| Variant Pattern | Function transformers | ❌ Imperative hooks | ❌ **MISALIGNED** |
| Variant Storage | In GameState | ❌ External in GameHost | ❌ **MISALIGNED** |
| Variant Composition | pipe/compose | ❌ N/A (no composition) | ❌ **MISALIGNED** |
| Capability System | Composable tokens | ❌ Not implemented | ❌ **MISSING** |
| Visibility Filtering | Pure functions | ❌ Not implemented | ❌ **MISSING** |
| Player Management | Dynamic add/remove | ❌ Fixed at creation | ⚠️ **PARTIAL** |

**Critical issues:**
1. **Variants are imperative** - Can't compose, hard to test, scattered logic
2. **State is incomplete** - Can't replay from actions (event sourcing broken)
3. **No visibility control** - Everyone sees everything (no spectators/coaches)

---

## The Migration Plan

### Phase 1: Foundation (No Breaking Changes)

**Create GameRules Interface**
- File: `src/game/core/rules.ts`
- Define `GameRules` type with `getValidActions`, `executeAction`, `checkGameEnd`
- Wrap existing pure functions in `standardRules` object
- Zero changes to existing game logic

```typescript
// NEW: src/game/core/rules.ts
export interface GameRules {
  getValidActions: (state: GameState) => GameAction[]
  executeAction: (state: GameState, action: GameAction) => GameState
  checkGameEnd?: (state: GameState) => GameEndResult | null
}

export const standardRules: GameRules = {
  getValidActions,  // Import from gameEngine.ts
  executeAction,    // Import from actions.ts
  checkGameEnd: (state) =>
    state.phase === 'game_end'
      ? { winner: state.winner, scores: state.teamScores }
      : null
}
```

**Create Variant Type**
- File: `src/game/variants/types.ts`
- Define `Variant = (GameRules) => GameRules`
- Add `composeVariants` helper for function composition

```typescript
// NEW: src/game/variants/types.ts
export type Variant = (baseRules: GameRules) => GameRules

export const composeVariants = (...variants: Variant[]): Variant =>
  (baseRules) => variants.reduce(
    (rules, variant) => variant(rules),
    baseRules
  )
```

### Phase 2: Implement Functional Variants

**One-Hand Variant**
- File: `src/game/variants/oneHand.ts`
- Transform `checkGameEnd` to end after one hand
- Test in isolation

```typescript
// NEW: src/game/variants/oneHand.ts
export const oneHandVariant: Variant = (baseRules) => ({
  getValidActions: baseRules.getValidActions,
  executeAction: baseRules.executeAction,

  checkGameEnd: (state) => {
    if (state.phase === 'scoring' &&
        state.consensus.scoreHand.size === 4) {
      return {
        winner: state.teamScores[0] > state.teamScores[1] ? 0 : 1,
        reason: 'one-hand-complete',
        scores: state.teamScores
      }
    }
    return null
  }
})
```

**Minimum Bid Variant**
- File: `src/game/variants/minBid.ts`
- Transform `getValidActions` to filter low bids
- Parameterized by minimum value

```typescript
// NEW: src/game/variants/minBid.ts
export const minBidVariant = (minBid: number): Variant =>
  (baseRules) => ({
    getValidActions: (state) => {
      const actions = baseRules.getValidActions(state)

      if (state.phase !== 'bidding') return actions

      return actions.filter(action => {
        if (action.type === 'bid' && action.bid === 'points') {
          return action.value >= minBid
        }
        return true
      })
    },

    executeAction: baseRules.executeAction,
    checkGameEnd: baseRules.checkGameEnd
  })
```

**Tournament Variant (Composition Example)**
- File: `src/game/variants/tournament.ts`
- Compose multiple transformations
- Demonstrates pure function composition

```typescript
// NEW: src/game/variants/tournament.ts
export const tournamentVariant: Variant = composeVariants(
  // Remove special contracts (nello, plunge)
  (baseRules) => ({
    getValidActions: (state) => {
      const actions = baseRules.getValidActions(state)
      return actions.filter(action =>
        action.type !== 'bid' ||
        ['points', 'marks'].includes(action.bid)
      )
    },
    executeAction: baseRules.executeAction,
    checkGameEnd: baseRules.checkGameEnd
  }),

  // Force minimum bid of 30
  minBidVariant(30)
)
```

### Phase 3: Move Variant Config to State

**Add variant field to GameState**
- File: `src/game/types.ts`
- Add `variant: { type, config }` to GameState interface
- Update `createInitialState` to include variant config

```typescript
// MODIFY: src/game/types.ts
export interface GameState {
  // ... existing fields ...

  // NEW: Variant configuration (part of state for event sourcing!)
  variant: {
    type: 'standard' | 'one-hand' | 'tournament' | 'min-bid' | 'custom'
    config?: {
      minBid?: number
      targetHand?: number
      originalSeed?: number
      composedVariants?: string[]  // Stack of variant types
      [key: string]: unknown
    }
  }
}
```

**Create variant resolver**
- File: `src/game/variants/registry.ts`
- `getVariantForState(state)` maps state.variant to Variant function
- `getRulesForState(state)` composes and returns active rules

```typescript
// NEW: src/game/variants/registry.ts
export function getVariantForState(state: GameState): Variant {
  switch (state.variant.type) {
    case 'standard':
      return (rules) => rules // Identity function

    case 'one-hand':
      return oneHandVariant

    case 'tournament':
      return tournamentVariant

    case 'min-bid':
      return minBidVariant(state.variant.config?.minBid || 30)

    case 'custom':
      // Build variant chain from config
      const variants: Variant[] = []
      if (state.variant.config?.minBid) {
        variants.push(minBidVariant(state.variant.config.minBid))
      }
      if (state.variant.config?.targetHand) {
        variants.push(oneHandVariant)
      }
      return composeVariants(...variants)

    default:
      return (rules) => rules
  }
}

// Get active rules for a state
export function getRulesForState(state: GameState): GameRules {
  const variant = getVariantForState(state)
  return variant(standardRules)
}
```

### Phase 4: Integration

**Update getValidActions**
- File: `src/game/core/gameEngine.ts`
- Check for variant in state, apply rules transformation
- Fall back to standard behavior

```typescript
// MODIFY: src/game/core/gameEngine.ts
export function getValidActions(state: GameState): GameAction[] {
  // Apply variant transformation if present
  if (state.variant && state.variant.type !== 'standard') {
    const rules = getRulesForState(state)
    return rules.getValidActions(state)
  }

  // Standard logic (unchanged)
  switch (state.phase) {
    case 'bidding': return getBiddingActions(state)
    case 'trumpSelection': return getTrumpSelectionActions(state)
    case 'playing': return getPlayingActions(state)
    case 'scoring': return getScoringActions(state)
    default: return []
  }
}
```

**Update GameHost**
- File: `src/server/game/GameHost.ts`
- Remove VariantRegistry calls
- Add variant game-end checking after action execution

```typescript
// MODIFY: src/server/game/GameHost.ts
executeAction(playerId: string, action: GameAction): { ok: boolean; error?: string } {
  const result = authorizeAndExecute(this.mpState, { playerId, action })

  if (!result.ok) {
    return { ok: false, error: result.error }
  }

  let newState = result.value.state

  // NEW: Check variant-specific game end
  const rules = getRulesForState(newState)
  if (rules.checkGameEnd) {
    const gameEnd = rules.checkGameEnd(newState)
    if (gameEnd) {
      newState = {
        ...newState,
        phase: 'game_end',
        winner: gameEnd.winner
      }
    }
  }

  this.mpState = { state: newState, sessions: result.value.sessions }
  this.notifyListeners()

  return { ok: true }
}
```

### Phase 5: Capability System (Functional)

**Implement visibility filters**
- File: `src/game/multiplayer/capabilities.ts`
- `VisibilityFilter = (GameState, playerId) => GameState`
- Implement standard filters

```typescript
// NEW: src/game/multiplayer/capabilities.ts
export type VisibilityFilter = (
  state: GameState,
  playerId: string
) => GameState

export const humanVisibility: VisibilityFilter = (state, playerId) => {
  const session = findSession(state, playerId)
  if (!session) return state

  // Hide other players' hands
  return {
    ...state,
    players: state.players.map((p, i) =>
      i === session.playerIndex
        ? p
        : { ...p, hand: [] }  // Hide hand
    )
  }
}

export const spectatorVisibility: VisibilityFilter = (state, _) => ({
  ...state,
  players: state.players.map(p => ({ ...p, hand: [] }))  // Hide all hands
})

export const coachVisibility = (studentIndex: number): VisibilityFilter =>
  (state, _) => ({
    ...state,
    players: state.players.map((p, i) =>
      i === studentIndex ? p : { ...p, hand: [] }  // Show only student's hand
    )
  })

export const debugVisibility: VisibilityFilter = (state, _) => state  // Show everything
```

**Apply in GameHost.getView()**
- Modify: `src/server/game/GameHost.ts`
- Use capability-based filtering for personalized views

```typescript
// MODIFY: src/server/game/GameHost.ts
getView(forPlayerId?: string): GameView {
  let visibleState = this.mpState.state

  if (forPlayerId) {
    const session = this.players.get(forPlayerId)
    const filter = getVisibilityFilter(session?.capabilities || ['human'])
    visibleState = filter(visibleState, forPlayerId)
  }

  const validActions = this.getValidActionsForView(visibleState, forPlayerId)

  return {
    state: visibleState,
    validActions,
    players: this.getPlayerInfo(),
    metadata: { ... }
  }
}
```

### Phase 6: Cleanup

**Delete VariantRegistry**
- Remove: `src/server/game/VariantRegistry.ts` (418 lines)
- All imperative hook code replaced with pure function composition

**Update tests**
- Test variants as pure functions (trivial!)
- Example:

```typescript
// Test variant in isolation
test('oneHandVariant ends after one hand', () => {
  const rules = oneHandVariant(standardRules)
  const stateAfterHand = { ...state, phase: 'scoring' }
  const result = rules.checkGameEnd(stateAfterHand)
  expect(result?.winner).toBeDefined()
})

// Test composition
test('tournamentVariant composes correctly', () => {
  const rules = tournamentVariant(standardRules)
  const actions = rules.getValidActions(biddingState)
  expect(actions.every(a => a.type !== 'bid' || a.value >= 30)).toBe(true)
})
```

**Verify event sourcing**
- Ensure replay from actions works perfectly

```typescript
// Replay test
test('can replay game from action history', () => {
  const finalState = gameState.actionHistory.reduce(
    (state, action) => {
      const rules = getRulesForState(state)
      return rules.executeAction(state, action)
    },
    initialState
  )
  expect(finalState).toEqual(gameState)
})
```

---

## Why This Architecture Is Correct

### 1. Compositional
```typescript
const rules = pipe(
  standardRules,
  oneHandVariant,
  tournamentVariant,
  minBidVariant(30)
)
```

Variants stack naturally. No hooks to wire up.

### 2. Testable
```typescript
// Each variant is a pure function - trivial to test
test('minBidVariant filters low bids', () => {
  const rules = minBidVariant(30)(standardRules)
  const actions = rules.getValidActions(state)
  expect(actions.filter(a => a.value < 30)).toHaveLength(0)
})
```

### 3. Event Sourceable
```typescript
// Replay from action history - perfect reconstruction
const finalState = actions.reduce(
  (state, action) => getRulesForState(state).executeAction(state, action),
  initialState
)
```

Variant config is IN state, so replay works perfectly.

### 4. Correct by Construction
- Type system enforces `Variant` signature
- Can't "forget" to apply a variant
- No way to break composition
- Compiler guarantees correctness

### 5. Parallel Universes
```typescript
// Test different rule combinations trivially
const standardResult = standardRules.executeAction(state, action)
const tournamentResult = tournamentVariant(standardRules).executeAction(state, action)
```

Can explore "what if?" scenarios without side effects.

---

## File Inventory

### Existing Files (Will Modify)

| File | Current State | Changes Needed |
|------|--------------|----------------|
| `src/game/types.ts` | GameState without variant | Add `variant` field |
| `src/game/core/gameEngine.ts` | Pure getValidActions | Add variant dispatch |
| `src/game/core/actions.ts` | Pure executeAction | No changes (already perfect!) |
| `src/game/core/state.ts` | createInitialState | Add variant parameter |
| `src/server/game/GameHost.ts` | Uses VariantRegistry | Replace with getRulesForState |
| `src/game/multiplayer/types.ts` | Has PlayerSession | Add capabilities field |

### New Files (Will Create)

| File | Purpose |
|------|---------|
| `src/game/core/rules.ts` | GameRules interface, standardRules wrapper |
| `src/game/variants/types.ts` | Variant type, composeVariants helper |
| `src/game/variants/oneHand.ts` | One-hand variant transformer |
| `src/game/variants/minBid.ts` | Minimum bid variant transformer |
| `src/game/variants/tournament.ts` | Tournament variant (composition example) |
| `src/game/variants/registry.ts` | getVariantForState, getRulesForState |
| `src/game/multiplayer/capabilities.ts` | Visibility filters, capability types |

### Files to Delete

| File | Reason |
|------|--------|
| `src/server/game/VariantRegistry.ts` | Replaced by functional composition |

---

## Key Insights

### Why Functional Composition Matters for Games

**Games are state machines.** The natural representation is:
```
State → [Valid Actions] → Pick Action → New State → ...
```

**Variants modify the machine.** They change what actions are valid, how state transitions work, when the game ends. The natural representation is:
```
Variant: StateMachine → StateMachine
```

**Function composition is the right tool:**
- Each variant is a pure function (easy to reason about)
- Variants compose via standard function composition (no special wiring)
- Type system ensures correctness (can't break composition)
- Testing is trivial (each variant tested in isolation)
- No side effects (can run "what if?" scenarios)

**Event sourcing becomes trivial:**
- Variant config is IN the state
- Replay is just: `actions.reduce(executeAction, initialState)`
- Perfect time travel debugging

**This is how Elm, Redux, and other functional architectures work.** We should follow their lead.

### The Imperative Trap

The current `VariantRegistry` falls into the imperative trap:
- Hooks scattered across codebase (`initialize`, `afterAction`, `checkGameEnd`, etc.)
- No composition story (can't easily combine variants)
- Hard to test (need to set up full game state machine)
- Side effects possible (mutable VariantState)
- Event sourcing broken (variant config external to state)

**This is how OOP game engines fail.** Don't repeat their mistakes.

### The Pure Functional Way

Define the interface:
```typescript
type GameRules = {
  getValidActions: (state) => actions
  executeAction: (state, action) => newState
  checkGameEnd: (state) => result | null
}
```

Variants transform the interface:
```typescript
type Variant = (rules: GameRules) => GameRules
```

Composition is just function composition:
```typescript
const myRules = variant3(variant2(variant1(standardRules)))
```

**This is correct by construction.** The type system enforces it.

---

## Testing Strategy

### Unit Tests (Pure Functions)

Each variant gets isolated tests:

```typescript
// src/game/variants/oneHand.test.ts
describe('oneHandVariant', () => {
  it('should end game after one hand completes', () => {
    const rules = oneHandVariant(standardRules)
    const state = { ...baseState, phase: 'scoring' as const }
    const result = rules.checkGameEnd(state)
    expect(result?.reason).toBe('one-hand-complete')
  })
})
```

### Composition Tests

Test that variants compose correctly:

```typescript
// src/game/variants/composition.test.ts
describe('variant composition', () => {
  it('should stack tournament rules', () => {
    const rules = tournamentVariant(standardRules)
    const actions = rules.getValidActions(biddingState)

    // Should enforce minimum bid
    expect(actions.filter(a => a.type === 'bid' && a.value < 30)).toHaveLength(0)

    // Should remove special contracts
    expect(actions.filter(a => a.bid === 'nello')).toHaveLength(0)
  })
})
```

### Event Sourcing Tests

Verify replay works:

```typescript
// src/tests/integration/event-sourcing.test.ts
describe('event sourcing', () => {
  it('should perfectly replay game from action history', () => {
    // Play a full game
    const finalState = playFullGame()

    // Replay from actions
    const replayedState = finalState.actionHistory.reduce(
      (state, action) => {
        const rules = getRulesForState(state)
        return rules.executeAction(state, action)
      },
      initialState
    )

    expect(replayedState).toEqual(finalState)
  })
})
```

---

## Migration Timeline

**Estimated effort:** ~3-4 hours of focused work

### Phase 1: Foundation (~30 min)
- Create GameRules interface
- Create Variant type
- No breaking changes

### Phase 2: Functional Variants (~45 min)
- Implement oneHandVariant
- Implement minBidVariant
- Implement tournamentVariant
- Write unit tests

### Phase 3: State Integration (~30 min)
- Add variant field to GameState
- Create variant resolver
- Update createInitialState

### Phase 4: Integration (~45 min)
- Update getValidActions
- Update GameHost
- Remove VariantRegistry calls

### Phase 5: Capabilities (~45 min)
- Implement visibility filters
- Update GameHost.getView()
- Add capability field to PlayerSession

### Phase 6: Cleanup (~15 min)
- Delete VariantRegistry.ts
- Update tests
- Verify event sourcing

---

## Critical Success Factors

### 1. Don't Break Existing Tests
Current tests use `executeAction` and `getValidActions` directly. These are already pure and should continue working unchanged.

### 2. Make Variant Config Part of State
This is CRITICAL for event sourcing. Without variant config in state, replay from action history is impossible.

### 3. Keep Functions Pure
No side effects, no mutation. Variants should be referentially transparent.

### 4. Test Composition
Ensure that `composeVariants(v1, v2, v3)` works correctly and produces expected behavior.

### 5. Verify Event Sourcing
Write explicit tests that replay games from action history and verify state matches.

---

## Questions to Answer During Implementation

### Q: How do we handle variant-specific state?
**A:** We don't! Variants only transform behavior, not state shape. All state goes in GameState. If a variant needs to track something (like "which hand are we on?"), it goes in `GameState.variant.config` or we derive it from existing state.

### Q: What if two variants conflict?
**A:** Function composition order matters. Last transformer wins. This is explicit and predictable. Can add validation in `composeVariants` if needed.

### Q: How do we handle time-based variants (speed mode)?
**A:** Timestamps go in GameState. Variant transforms `getValidActions` to filter out actions that violate time limits. Time checking is external (in GameHost or client).

### Q: How do we handle progressive complexity (tutorial mode)?
**A:** Tutorial is a variant! It transforms `getValidActions` to show only tutorial-allowed actions. Tutorial state (current step, hints, etc.) goes in `GameState.variant.config`.

---

## References

### Vision Document
- **File:** `docs/remixed-855ccfd5.md`
- **Section 1.2:** Architecture Principles (pure function composition)
- **Section 5:** Variant System (function transformers)
- **Section 4:** Capability System (composable tokens)

### Current Implementation
- **Pure functions:** `src/game/core/actions.ts`, `src/game/core/gameEngine.ts`
- **Imperative variants:** `src/server/game/VariantRegistry.ts` (to be deleted)
- **GameHost:** `src/server/game/GameHost.ts` (to be updated)

### Inspirations
- **Elm Architecture:** Model-Update-View with pure state transitions
- **Redux:** Reducers as pure functions, middleware as transformers
- **React Hooks:** Composition via function composition
- **Functional Core, Imperative Shell:** Gary Bernhardt's pattern

---

## Next Steps

1. **Read this document thoroughly** - Understand the vision and the gap
2. **Review vision document** - `docs/remixed-855ccfd5.md` sections 1.2, 4, 5
3. **Examine current code** - See what's pure (✅) and what's imperative (❌)
4. **Start Phase 1** - Create GameRules interface (no breaking changes!)
5. **Test each phase** - Ensure tests pass after each phase
6. **Delete VariantRegistry** - Final cleanup when functional system works

---

## Status Check Commands

```bash
# Typecheck
npm run typecheck

# Unit tests
npm test

# Verify current state
git status

# See what's been modified
git diff src/game/types.ts
git diff src/server/game/GameHost.ts
```

---

## Commit Strategy

Atomic commits per phase:
1. `feat: add GameRules interface and Variant type`
2. `feat: implement oneHand, minBid, tournament variants`
3. `feat: move variant config into GameState`
4. `refactor: integrate functional variants in gameEngine and GameHost`
5. `feat: add capability system with visibility filters`
6. `refactor: delete VariantRegistry, use functional composition`

Each commit should pass tests. Perfect for code review.

---

**End of handoff. Ready to implement pure functional composition.**
