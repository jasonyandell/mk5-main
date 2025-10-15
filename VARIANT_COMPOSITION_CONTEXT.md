# Variant Composition Refactor - Context & Grounding

**Purpose:** This document provides essential context to prevent confusion between "what exists" and "what we're building." Read this FIRST before touching any code.

---

## Critical Mental Model

### ❌ DO NOT Trust Current Implementation
The current variant system is **fundamentally wrong** and being completely replaced. Do not learn from it, copy from it, or try to preserve any of its patterns.

### ✅ DO Trust These Documents
1. **Vision document** (`docs/remixed-855ccfd5.md`) - The north star
2. **Refactor plan** (`VARIANT_COMPOSITION_REFACTOR.md`) - What we're building
3. **This context doc** - What to ignore vs what to preserve

---

## What Currently Exists (DO NOT COPY THESE PATTERNS)

### ❌ WRONG: Imperative VariantRegistry (418 lines)
**Location:** `src/server/game/VariantRegistry.ts`

**Pattern (DO NOT USE):**
```typescript
VariantRegistry.register({
  type: 'one-hand',
  initialize: (state) => { /* mutate */ },
  afterAction: (state, action) => { /* check and mutate */ },
  checkGameEnd: (state) => { /* inspect */ },
  updateVariantState: (state) => { /* track separate state */ }
});
```

**Problems:**
- Imperative lifecycle hooks (not composable)
- Separate `VariantState` tracking (breaks event sourcing)
- 418 lines of ceremony
- Server knows about variants (coupling)

**Status:** WILL BE DELETED COMPLETELY

### ❌ WRONG: Variant Flags in GameState
**Location:** `src/game/types.ts` lines 169-170

**Pattern (DO NOT USE):**
```typescript
interface GameState {
  tournamentMode: boolean;  // ❌ Variant shouldn't be in core state
  gameTarget: number;       // ❌ Unclear purpose, variant-specific
}
```

**Status:** WILL BE REMOVED (affects 49 files)

### ❌ WRONG: Core Engine Variant Awareness
**Location:** `src/game/core/rules.ts`, `src/game/core/gameEngine.ts`

**Pattern (DO NOT USE):**
```typescript
// Core engine checking variant flags
if (!state.tournamentMode) {
  // Generate special bids...
}

function isValidBid(state, bid, hand, tournamentMode) {
  if (tournamentMode) {
    // Filter special bids...
  }
}
```

**Status:** WILL BE REMOVED - base engine will be maximally permissive

---

## What to Preserve (KEEP THESE)

### ✅ GOOD: Core Game Engine
**Locations:**
- `src/game/core/actions.ts` - `executeAction` (pure state transitions)
- `src/game/core/gameEngine.ts` - `getValidActions` (action generation)
- `src/game/core/state.ts` - `createInitialState` (initialization)
- `src/game/core/rules.ts` - Game rules (dominoes, scoring, etc.)

**These are SOUND. They work. We're just removing variant awareness from them.**

**What we'll change:**
- Remove `tournamentMode` parameters
- Make them maximally permissive (generate ALL possible actions)
- Let variants filter/transform at composition time

**What we'll keep:**
- All the game logic (bidding rules, trick taking, scoring)
- Pure function signatures
- Zero-coupling architecture

### ✅ GOOD: Protocol Layer
**Locations:**
- `src/shared/multiplayer/protocol.ts` - Message types
- `src/server/game/GameHost.ts` - Game authority
- `src/game/multiplayer/authorization.ts` - Permission checking

**These are architecturally CORRECT. We're just adding variant composition to them.**

**What we'll change:**
- GameHost composes variants at construction
- Protocol adds `variants` array to GameConfig
- Remove VariantRegistry calls from GameHost

**What we'll keep:**
- Pure state management
- Authorization layer
- Protocol message flow

### ✅ GOOD: Event Sourcing Foundation
**Location:** `src/game/types.ts` line 179

```typescript
interface GameState {
  actionHistory: GameAction[];  // ✅ Already exists and is good!
}
```

**Status:** This is PERFECT. We're building on it.

We'll add:
- `initialConfig: GameConfig` to state
- `replayActions(config, actions)` utility
- Make action history the source of truth

### ✅ GOOD: Seed System
**Locations:**
- `src/game/core/dominoes.ts` - `dealDominoesWithSeed`
- `src/game/core/state.ts` - `shuffleSeed` in config

**Status:** Deterministic shuffling works perfectly. Keep it.

### ✅ GOOD: AI System (for later)
**Locations:**
- `src/game/core/ai-scheduler.ts` - `selectAIAction`
- `src/game/core/gameSimulator.ts` - Game simulation

**Status:** Sound logic. Currently deterministic. We're NOT integrating this in Phase 1 (one-hand uses hardcoded bid). But it's there for future use.

---

## Architecture Comparison

### Current (WRONG) - What Exists Now

```
GameHost
  ↓
Creates state with variant flags
  ↓
VariantRegistry.initialize(state, variant) → mutates state
  ↓
Core engine checks variant flags
  ↓
VariantRegistry.afterAction(state, action) → mutates state
  ↓
VariantRegistry.checkGameEnd(state) → inspects state
```

**Problems:**
- Imperative mutations
- Variants have privileged access to state
- Core engine knows about variants
- Can't compose variants
- No event sourcing

### Target (CORRECT) - What We're Building

```
GameHost
  ↓
Composes variant transforms
  createState = v1.transform(v2.transform(base))
  getActions = v1.transform(v2.transform(base))
  executeAction = v1.transform(v2.transform(base))
  ↓
Uses composed functions
  state = createState(config)  → Event sourcing
  actions = getActions(state)   → Pure composition
  state = executeAction(state, action)  → Pure transition
```

**Benefits:**
- Pure function composition
- Variants are external transforms
- Core engine has zero variant awareness
- Variants compose naturally
- Perfect event sourcing

---

## Key Principles (Memorize These)

### 1. Event Sourcing is Foundation
```typescript
// This is THE fundamental truth
GameState = replayActions(initialConfig, actionHistory)

// Everything else is derived
const currentPhase = state.phase  // Computed from history
const validActions = getActions(state)  // Computed from current state
```

### 2. Variants Transform, They Don't Mutate
```typescript
// ❌ WRONG (current implementation)
variant.afterAction(state, action) {
  state.someField = newValue  // Mutation!
}

// ✅ CORRECT (what we're building)
variant.transformExecuteAction = (base) => (state, action) => {
  const newState = base(state, action)  // Pure
  return { ...newState, someField: newValue }  // New object
}
```

### 3. Base Functions are Maximally Permissive
```typescript
// ❌ WRONG (current)
function getValidActions(state) {
  const actions = [...];
  if (!state.tournamentMode) {
    actions.push({ bid: 'nello' })  // Core knows about variants
  }
  return actions;
}

// ✅ CORRECT (what we're building)
function baseGetValidActions(state) {
  return [
    { bid: 'points' },
    { bid: 'nello' },  // Always generate ALL actions
    { bid: 'splash' },
    { bid: 'plunge' }
  ];
}

// Variants filter
tournamentVariant.transformGetValidActions = (base) => (state) => {
  return base(state).filter(a => a.bid !== 'nello')  // Variant filters
}
```

### 4. Optional Transforms, Not Required Hooks
```typescript
// ❌ WRONG (current)
// Variant must implement ALL hooks, even if empty
{
  initialize: (state) => state,  // Empty
  afterAction: (state) => state,  // Empty
  checkGameEnd: () => false  // Empty
}

// ✅ CORRECT (what we're building)
// Variant specifies ONLY what it needs
{
  type: 'tournament',
  transformGetValidActions: (base) => (state) => { /* only this */ }
  // No other transforms needed!
}
```

### 5. Composition Order Matters
```typescript
// Variants compose left-to-right, last one wins on conflicts
variants: [
  { type: 'tournament' },  // First: filters special bids
  { type: 'one-hand' }     // Last: if it changes phase, that's final
]

// Becomes:
getActions = oneHand.transform(tournament.transform(base))
//           ^^^^^^^ outermost (last)
//                   ^^^^^^^^^^ inner
//                              ^^^^ base
```

---

## One-Hand Variant: Critical Clarifications

### What One-Hand IS
- **Skip bidding for new players** who don't understand hand strength
- **Create shareable challenges** for experts (via seed)
- **Start at 'playing' phase** (bidding already done)
- **End after one hand** (not continue to new hands)

### What One-Hand is NOT
- ❌ Not about "counting hands" (it's always exactly 1 hand)
- ❌ Not about "playing N hands then stopping"
- ❌ Not about giving players bidding practice

### Implementation (Phase 1 - Simple)
```typescript
// Hardcoded bid: Player 3 bids 30 points, suit 4 trump
// No AI integration yet (too complex for Phase 1)

transformInitialState: (base) => (config) => {
  let state = base(config)  // Deals hands from seed

  // Fast-forward through bidding (hardcoded)
  state = executeAction(state, { type: 'bid', player: 3, bid: 'points', value: 30 })
  state = executeAction(state, { type: 'pass', player: 0 })
  state = executeAction(state, { type: 'pass', player: 1 })
  state = executeAction(state, { type: 'pass', player: 2 })
  state = executeAction(state, { type: 'select-trump', player: 3, trump: { type: 'suit', suit: 4 } })

  return state  // Now at 'playing' phase
}

transformExecuteAction: (base) => (state, action) => {
  const newState = base(state, action)

  // After scoring, end instead of new hand
  if (action.type === 'score-hand' && newState.phase === 'bidding') {
    return { ...newState, phase: 'game_end' }
  }

  return newState
}
```

**Simple. Deterministic. No AI complexity. ~40 lines total.**

### Future Enhancement (Phase 2+)
- Use AI to determine bid/trump from dealt hand
- Make bid/trump configurable
- Add seed finder to find competitive scenarios

**But NOT in Phase 1!** Keep it simple.

---

## URL System: Critical Constraint

### Current State
- Beautiful URL compression exists (`src/game/core/url-compression.ts`)
- Currently broken (doesn't replay actions)
- Architecturally misaligned (mixes client/server state)

### What We Need
**URL isomorphism:** State can be PERFECTLY reconstructed from URL.

```typescript
// URL format: /?s=42069&h=one_hand&a=play-0-2,play-1-5,...
//              └─seed   └─mode      └─action history

// Perfect reconstruction
const config = { shuffleSeed: 42069, variants: [{ type: 'one-hand' }] }
const actions = parseActions('play-0-2,play-1-5,...')
const state = replayActions(config, actions)  // Exact state!
```

### Status for This Refactor
**OUT OF SCOPE** - Fix later in separate effort.

**Why?** Event sourcing architecture supports this perfectly (config + actions = state). But URL encoding/decoding is a large migration that's independent of variant composition.

**What we do now:** Ensure architecture supports it (via `replayActions`). Implement encoding later.

---

## Testing Strategy

### ❌ WRONG: Test with variant flags
```typescript
// Old pattern - DO NOT USE
const state = createInitialState({ tournamentMode: true })
const actions = getValidActions(state)
expect(actions.some(a => a.bid === 'nello')).toBe(false)
```

### ✅ CORRECT: Test with composition
```typescript
// New pattern - USE THIS
const baseState = createInitialState({ shuffleSeed: 123 })
const baseActions = getValidActions(baseState)

// Test base is permissive
expect(baseActions.some(a => a.bid === 'nello')).toBe(true)

// Test variant filters
const variant = getVariant('tournament')
const composed = variant.transformGetValidActions!(getValidActions)
const filtered = composed(baseState)
expect(filtered.some(a => a.bid === 'nello')).toBe(false)
```

### ✅ CORRECT: Test with event sourcing
```typescript
// Create state at arbitrary point via action replay
const state = replayActions(
  { shuffleSeed: 42069, variants: [{ type: 'one-hand' }] },
  [
    { type: 'play', player: 0, dominoId: '0-2' },
    { type: 'play', player: 1, dominoId: '1-5' }
  ]
)

expect(state.currentTrick.length).toBe(2)
```

---

## Common Pitfalls & How to Avoid

### Pitfall 1: Referencing Old VariantRegistry
**Symptom:** Import statement `import { VariantRegistry } from '...'`

**Fix:** Don't reference it. It's being deleted. Use `getVariant()` from new registry.

### Pitfall 2: Checking tournamentMode in Core Engine
**Symptom:** `if (state.tournamentMode)` in core files

**Fix:** Remove the check. Base functions should be maximally permissive. Let variants filter.

### Pitfall 3: Mutating State in Variants
**Symptom:** `state.someField = newValue`

**Fix:** Return new object: `return { ...state, someField: newValue }`

### Pitfall 4: Trying to Test Variants via GameState Flags
**Symptom:** Creating state with `{ tournamentMode: true }` in tests

**Fix:** Use `replayActions` with variant config, or compose variants manually in tests.

### Pitfall 5: Implementing All Three Transforms for Every Variant
**Symptom:** Empty `transformInitialState` that just returns state

**Fix:** Only implement transforms you need. They're optional!

### Pitfall 6: Coupling to Seed Finder
**Symptom:** Trying to integrate seed finding into one-hand variant

**Fix:** Seed finder is OUT OF SCOPE. One-hand works with any seed (even random). Add seed finder later.

---

## Scope Boundaries (What's IN vs OUT)

### ✅ IN SCOPE - Do These Now

1. **Create variant system** (`src/game/variants/`)
   - Type definitions
   - Registry (pure lookup)
   - Tournament variant
   - One-hand variant (hardcoded bid)
   - Speed-mode variant

2. **Create event sourcing utilities** (`src/game/core/replay.ts`)
   - `replayActions(config, actions)`
   - `createInitialStateWithVariants(config)`

3. **Update GameHost**
   - Compose variants at construction
   - Use composed functions
   - Remove VariantRegistry calls

4. **Delete old system**
   - Delete `VariantRegistry.ts` (418 lines)
   - Remove `tournamentMode` from GameState
   - Fix all 49 files

5. **Update tests**
   - Use composition patterns
   - Use event sourcing patterns
   - Verify all pass

### ❌ OUT OF SCOPE - Defer These

1. **Seed finder** - Entire system deferred
2. **E2E test refactoring** - Separate effort
3. **URL encoding/compression** - Separate effort
4. **Capability system** - Separate effort
5. **AI integration for one-hand** - Future enhancement

---

## File-by-File Guidance

### Files to CREATE (new code)
- `src/game/variants/types.ts` - ✅ Pure types, safe to create
- `src/game/variants/registry.ts` - ✅ Pure lookup, safe to create
- `src/game/variants/tournament.ts` - ✅ New variant, safe to create
- `src/game/variants/oneHand.ts` - ✅ New variant, safe to create
- `src/game/variants/speedMode.ts` - ✅ New variant, safe to create
- `src/game/core/replay.ts` - ✅ Event sourcing utility, safe to create

### Files to UPDATE (modify carefully)
- `src/game/types.ts` - ⚠️ Remove tournamentMode, add initialConfig
- `src/shared/multiplayer/protocol.ts` - ⚠️ Add variants array to GameConfig
- `src/server/game/GameHost.ts` - ⚠️ Add variant composition, remove VariantRegistry
- `src/game/core/rules.ts` - ⚠️ Remove tournamentMode parameters
- `src/game/core/state.ts` - ⚠️ Remove tournamentMode from options
- `src/game/core/gameEngine.ts` - ⚠️ Remove variant checks
- `src/stores/gameStore.ts` - ⚠️ Update variant creation

### Files to DELETE (remove completely)
- `src/server/game/VariantRegistry.ts` - ❌ Delete entirely (418 lines)

### Files to IGNORE (leave alone)
- `src/game/core/ai-scheduler.ts` - ✅ Keep as-is (future use)
- `src/game/core/gameSimulator.ts` - ✅ Keep as-is (future use)
- `src/game/core/seedFinder.ts` - ✅ Keep as-is (future migration)
- `src/game/core/url-compression.ts` - ✅ Keep as-is (future migration)
- All E2E test files - ✅ Keep as-is (separate effort)

---

## Success Signals (You're On Track If...)

### ✅ Good Signs
- Creating new files in `src/game/variants/`
- Implementing pure function transformers
- Using `replayActions` in tests
- Composing variants manually to test behavior
- Removing `tournamentMode` parameters from functions
- Making base functions more permissive (not less)

### 🚩 Warning Signs
- Importing `VariantRegistry` in new code
- Adding variant flags to GameState
- Checking variant state in core engine
- Mutating state in variants
- Implementing empty transform functions
- Trying to integrate seed finder
- Trying to fix E2E tests

### ❌ Red Flags (STOP and Re-Read This Doc)
- Copying patterns from `VariantRegistry.ts`
- Using lifecycle hooks (initialize, afterAction, etc.)
- Creating separate variant state tracking
- Making core engine check for specific variants
- Adding complexity to one-hand (should be ~40 lines)
- Worrying about URL encoding (out of scope)

---

## Quick Reference Card

**What am I building?**
Pure functional variant composition with event sourcing.

**What's the core insight?**
`GameState = replayActions(config, actionHistory)`

**What are variants?**
Optional function transformers over createState/getActions/executeAction.

**What's in scope?**
Variant system, event sourcing, tournament/one-hand/speed-mode, delete old code.

**What's out of scope?**
Seed finder, E2E tests, URL encoding, capabilities, AI integration.

**What's the success criteria?**
TypeScript compiles, unit tests pass, 418 lines deleted, tournamentMode gone.

**When do I reference old code?**
Only to understand what to DELETE, never to copy patterns.

**When do I reference new docs?**
Always. Vision doc + Refactor plan + This context doc = complete truth.

---

## Onboarding Checklist

Before writing any code:
- [ ] Read this entire context document
- [ ] Read `VARIANT_COMPOSITION_REFACTOR.md` (the plan)
- [ ] Skim `docs/remixed-855ccfd5.md` (the vision)
- [ ] Understand event sourcing: state = replay(config, actions)
- [ ] Understand composition: variant wraps base function
- [ ] Understand scope: what's IN (composition) vs OUT (seed finder, URLs)
- [ ] Understand one-hand: hardcoded bid, ~40 lines, no AI

When you start coding:
- [ ] Create new variant files first (safe, no breaking changes)
- [ ] Test variants in isolation (manual composition in tests)
- [ ] Wire into GameHost (compose at construction)
- [ ] Delete old code (big bang, fix all 49 files)
- [ ] Verify tests pass

When you get confused:
- [ ] Re-read relevant section of this doc
- [ ] Check: am I copying old patterns? (DON'T)
- [ ] Check: am I building pure functions? (DO)
- [ ] Check: am I in scope? (refer to boundaries section)

---

**This document is your grounding. Refer back whenever you're unsure whether something is "what exists" or "what we're building."**

**Remember: The current implementation is WRONG. Don't learn from it. Build the new system from the vision.**
