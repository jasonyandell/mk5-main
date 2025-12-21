# Suit System Unification Plan

## Executive Summary

**Goal**: Create a single elegant model for suits/trump/following-suit, clean up UI/game boundary, and delete unused features.

**Scope**:
1. Unify suit/trump logic into single authoritative module (`suit-context.ts`)
2. Delete `suitAnalysis` from GameState (stale cache anti-pattern)
3. Document and delete "Perfects" feature (unused, causes game logic in UI)

**Expected Outcome**: ~300+ lines deleted, single source of truth for all suit operations.

---

## Current Architecture

### The Unified Core (GameRules Interface)

The codebase has a well-designed `GameRules` interface in `src/game/layers/types.ts` with 17 methods. The suit-related methods are:

| Method | Purpose | Defined In |
|--------|---------|------------|
| `getLedSuit(state, domino)` | What suit does a domino lead? | base.ts, nello.ts override |
| `suitsWithTrump(state, domino)` | What suits does domino belong to? | compose.ts base, nello.ts override |
| `canFollow(state, led, domino)` | Can domino follow the led suit? | compose.ts base, nello.ts override |
| `rankInTrick(state, led, domino)` | Domino's rank for trick-taking | compose.ts base, nello.ts override |
| `calculateTrickWinner(state, trick)` | Who won the trick? | compose.ts, sevens.ts override |

**This is the correct abstraction** - it allows layers to override specific behaviors for special contracts (nello, sevens).

### The Problem: Parallel Implementations

The same logic exists in **THREE places**:

#### 1. Core Functions (`src/game/core/dominoes.ts`)
```typescript
getLedSuit(domino, trump): LedSuit          // lines 168-198
getDominoValue(domino, trump): number       // lines 203-234
isTrump(domino, trump): boolean             // lines 246-258
```

#### 2. Compose Base Functions (`src/game/layers/compose.ts`)
```typescript
suitsWithTrumpBase(state, domino): LedSuit[]   // lines 56-84
canFollowBase(state, led, domino): boolean     // lines 89-110
rankInTrickBase(state, led, domino): number    // lines 115-142
```

#### 3. Layer Rule Implementations (`src/game/layers/base.ts`)
```typescript
getLedSuit(state, domino): LedSuit          // lines 345-363
suitsWithTrump(state, domino): LedSuit[]    // lines 373-405
canFollow(state, led, domino): boolean      // lines 412-434
```

**And then nello.ts has its own complete set of overrides** (lines 104-169)!

---

## Pattern Analysis: The Same Logic Repeated

### Pattern 1: "Is this domino trump?"

This check appears in **10+ places** with slight variations:

```typescript
// In dominoes.ts:isTrump()
if (isDoublesTrump(trumpSuit)) {
  return domino.high === domino.low;
}
if (isRegularSuitTrump(trumpSuit)) {
  return dominoHasSuit(domino, trumpSuit);
}

// In compose.ts:rankInTrickBase()
const isTrump = isDoublesTrump(trumpSuit)
  ? isDouble
  : isRegularSuitTrump(trumpSuit) && dominoHasSuit(domino, trumpSuit);

// In base.ts:suitsWithTrump()
if (isDoublesTrump(trumpSuit)) { ... }
if (isRegularSuitTrump(trumpSuit)) { ... }
```

### Pattern 2: "What suits does this domino belong to?"

Duplicated in:
- `compose.ts:suitsWithTrumpBase()` (lines 56-84)
- `base.ts:suitsWithTrump()` (lines 373-405) - **identical logic**
- `nello.ts:suitsWithTrump()` (lines 111-120) - nello-specific

### Pattern 3: "Can this domino follow the led suit?"

Duplicated in:
- `compose.ts:canFollowBase()` (lines 89-110)
- `base.ts:canFollow()` (lines 412-434) - **identical logic**
- `nello.ts:canFollow()` (lines 123-140) - nello-specific

### Pattern 4: "Is doubles the led suit?"

The check `led === 7` appears scattered:
- `compose.ts:canFollowBase()` line 94
- `compose.ts:rankInTrickBase()` line 131
- `base.ts:canFollow()` line 417
- `nello.ts:canFollow()` line 129
- `nello.ts:rankInTrick()` line 151

---

## The Fundamental Issue

**The layer system requires base implementations in compose.ts, but base.ts duplicates them.**

Looking at the code flow:
1. `composeRules(layers)` in compose.ts creates composed rules
2. For `canFollow`, it starts with `canFollowBase()` and threads through layers
3. But `base.ts:canFollow()` **just passes through** (it returns `prev`)!
4. So `base.ts:canFollow()` is dead code - the logic is actually in `canFollowBase()`

Same for:
- `base.ts:suitsWithTrump()` - duplicates `suitsWithTrumpBase()` but never called
- `base.ts:rankInTrick()` - just returns `prev`

**base.ts has ~60 lines of dead/duplicate suit logic.**

---

## Nello's Necessary Duplication

Nello genuinely needs different logic:
- **Doubles form their own suit (7)** even without doubles-trump
- **No trump hierarchy** - only suit following matters
- **Doubles can't follow non-double leads**

This is correctly handled via layer overrides. But the overrides duplicate the entire algorithm structure, not just the differing parts.

---

## UI Boundary Violation (The Deeper Problem)

### What's Happening

`src/lib/utils/dominoHelpers.ts` (UI layer) re-implements game logic:

```typescript
// UI is computing game logic!
function isTrump(domino: Domino, trump: TrumpSelection): boolean {
  if (trump.type === 'no-trump') return false;
  if (trump.type === 'doubles') return domino.high === domino.low;
  return dominoHasSuit(domino, trump.suit!);  // imports from game core
}

function getPlayContexts(domino: Domino, trump: TrumpSelection): LedSuitOrNone[] { ... }
function computeExternalBeaters(leftoverDominoes: string[], bestTrumpStr: string): string[] { ... }
```

This is used only by **PerfectHandDisplay** (seed finder). The UI shouldn't need to understand trump rules.

### The Clean Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  UI Layer                                                        │
│  - Receives pre-computed data                                    │
│  - Knows NOTHING about suit rules, trump, following              │
│  - Just renders what server tells it                             │
└─────────────────────────────────────────────────────────────────┘
                               ↑
                      GameView / ViewProjection
                               ↑
┌─────────────────────────────────────────────────────────────────┐
│  Server (Room + Kernel)                                          │
│  - Owns all game logic                                           │
│  - Computes suit analysis, strength, beaters                     │
│  - Sends UI-ready data in GameView                               │
└─────────────────────────────────────────────────────────────────┘
                               ↑
                      Unified Suit Model (suit-context.ts)
```

---

## Proposed Solution: Two-Part Refactor

### Part 1: Unify Game-Side Suit Logic (suit-context.ts)

#### The Key Insight

All suit operations depend on exactly **three inputs**:
1. The domino being analyzed
2. The trump selection
3. (For following/ranking) The led suit

#### Proposal: Create `src/game/core/suit-context.ts`

A single module that provides all suit operations with a consistent API:

```typescript
// suit-context.ts

/** What suits does this domino belong to given the trump? */
export function getSuitsOfDomino(domino: Domino, trump: TrumpSelection): LedSuit[] {
  // Single authoritative implementation
}

/** Is this domino trump? */
export function isDominoTrump(domino: Domino, trump: TrumpSelection): boolean {
  // Delegates to getSuitsOfDomino check
}

/** What suit does this domino lead? */
export function getLedSuit(domino: Domino, trump: TrumpSelection): LedSuit {
  // Single authoritative implementation
}

/** Can this domino follow the led suit? */
export function canDominoFollow(
  domino: Domino,
  trump: TrumpSelection,
  ledSuit: LedSuit
): boolean {
  // Delegates to getSuitsOfDomino
}

/** Get domino's rank in a trick */
export function getDominoRank(
  domino: Domino,
  trump: TrumpSelection,
  ledSuit: LedSuit
): number {
  // Single authoritative implementation
}
```

#### How Layers Would Use It

The base layer becomes thin:

```typescript
// base.ts
rules: {
  getLedSuit: (state, domino) =>
    suitContext.getLedSuit(domino, state.trump),

  suitsWithTrump: (state, domino) =>
    suitContext.getSuitsOfDomino(domino, state.trump),

  canFollow: (state, led, domino) =>
    suitContext.canDominoFollow(domino, state.trump, led),

  rankInTrick: (state, led, domino) =>
    suitContext.getDominoRank(domino, state.trump, led),
}
```

Nello layer overrides would provide **alternative implementations** of the same interface:

```typescript
// nello.ts
const nelloSuitContext = {
  getSuitsOfDomino: (domino, trump) => {
    if (trump.type !== 'nello') return undefined; // fall through
    return isDouble(domino) ? [7] : [domino.high, domino.low];
  },
  // ... other overrides
};
```

### Part 2: Document and Delete Perfects Feature

**What It Is**: "Gold Perfect Hands" - a tool to find domino hands that guarantee winning all 7 tricks. Used for finding interesting seed values and analyzing hand strength.

**Files to Document and Delete**:
- `src/PerfectsApp.svelte` - Main Svelte app for displaying perfect hands
- `src/lib/components/PerfectHandDisplay.svelte` - Display component
- `src/lib/utils/dominoHelpers.ts` - Game logic re-implemented for UI
- `src/lib/utils/domino-sort.ts` - Sorting with trump awareness
- `scripts/find-perfect-hands.ts` - CLI script to find perfect hands
- `scripts/find-perfect-partition.ts` - Partition finder
- `scripts/find-3hand-leftover.ts` - 3-hand leftover analysis
- `data/3hand-partitions.json` - Pre-computed partition data
- `docs/archive/perfect-hand-plan2.md` - Design documentation (keep for history)

**Git History**: Last modified in commits c7afb0e through b57ff61 (various "green" and "interim - perfect hands" commits)

**Documentation Location**: Create `docs/archive/perfects-feature.md` before deletion

---

## Implementation Steps

### Step 1: Document and Delete Perfects Feature
1. Create `docs/archive/perfects-feature.md` documenting what it was
2. Delete all Perfects files (see list above)
3. Remove any vite/svelte config for PerfectsApp
4. Run tests to verify nothing breaks

### Step 2: Create suit-context.ts (Unify Game Logic)
1. Create `src/game/core/suit-context.ts` with authoritative functions:
   - `getSuitsOf(domino, trump)` - What suits does it belong to?
   - `isTrump(domino, trump)` - Is it trump?
   - `getLedSuit(domino, trump)` - What suit does it lead?
   - `canFollow(domino, trump, led)` - Can it follow suit?
   - `getRank(domino, trump, led)` - Its rank for trick-taking
2. All existing code keeps working (additive)

### Step 3: Thin out base.ts + compose.ts
1. Delete `suitsWithTrumpBase`, `canFollowBase`, `rankInTrickBase` from compose.ts
2. Update base.ts rules to delegate to suit-context
3. ~60 lines deleted

### Step 4: Simplify nello.ts
1. Create nello-specific suit-context overrides
2. Replace 40+ lines of duplicated algorithm structure

### Step 5: Delete suitAnalysis from GameState
1. Remove from Player interface in types.ts
2. Remove computation in actions.ts (executePlay, executeTrumpSelection)
3. Remove from state.ts initialization
4. Remove from multiplayer filtering
5. Update any tests that relied on it

---

## Files to Modify

### Step 1: Delete Perfects Feature

| File | Action |
|------|--------|
| `docs/archive/perfects-feature.md` | **Create** - document before deletion |
| `src/PerfectsApp.svelte` | **Delete** |
| `src/lib/components/PerfectHandDisplay.svelte` | **Delete** |
| `src/lib/utils/dominoHelpers.ts` | **Delete** |
| `src/lib/utils/domino-sort.ts` | **Delete** |
| `scripts/find-perfect-hands.ts` | **Delete** |
| `scripts/find-perfect-partition.ts` | **Delete** |
| `scripts/find-3hand-leftover.ts` | **Delete** |
| `data/3hand-partitions.json` | **Delete** |
| `vite.config.ts` (if applicable) | Remove PerfectsApp entry |

### Step 2-4: Unify Suit Logic

| File | Action |
|------|--------|
| `src/game/core/suit-context.ts` | **Create** - authoritative suit/trump model |
| `src/game/core/dominoes.ts` | Delete `getLedSuit`, `getDominoValue`, `isTrump` (moved) |
| `src/game/layers/compose.ts` | Delete `*Base` functions |
| `src/game/layers/base.ts` | Thin out to delegate to suit-context |
| `src/game/layers/nello.ts` | Simplify overrides |
| `src/game/ai/*.ts` | Update imports |

### Step 5: Delete suitAnalysis

| File | Action |
|------|--------|
| `src/game/types.ts` | Remove `suitAnalysis` from Player, SuitAnalysis types |
| `src/game/core/actions.ts` | Remove computation in executePlay/executeTrumpSelection |
| `src/game/core/state.ts` | Remove initialization |
| `src/game/core/suit-analysis.ts` | Keep `analyzeSuits()` for on-demand use, delete caching |
| `src/multiplayer/capabilities.ts` | Remove from filtering |
| Tests | Update any that relied on suitAnalysis |

---

## Expected Outcomes

1. **Single source of truth**: All suit logic in `suit-context.ts`
2. **~300+ lines deleted**: Perfects feature + duplicate implementations + suitAnalysis caching
3. **Clear mental model**: "Suit questions → suit-context.ts"
4. **Clean UI boundary**: UI receives data, never computes game rules
5. **No stale cache bugs**: suitAnalysis computed on-demand only

---

## Related Issues

- **t42-e92**: Epic: Suit system consolidation + simulation performance
- **t42-ofy**: Crystal Palace: Suit System Consolidation (CLOSED)
- **t42-v17**: Make suit analysis lazy or derivation-based
- **t42-cjw**: Simulation hot path optimizations (quick wins)
