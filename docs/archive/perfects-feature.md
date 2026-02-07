# Perfects Feature (Archived)

> **Status**: Deleted as part of Trump Unification refactor
> **Reason**: UI layer was computing game logic, violating client/server boundary
> **Date Archived**: December 2025

## What It Was

The "Perfects" feature was a tool for finding and displaying **gold perfect hands** - domino hands that guarantee winning all 7 tricks when played optimally.

### Definition of a Gold Perfect Hand

A gold perfect hand requires:
1. **4+ trumps** to guarantee trump control
2. **Trumps must be the highest available** (no external dominoes can beat them)
3. **Non-trumps can only be beaten by trumps or dominoes already in the hand**

The key insight: in a perfect hand, you always lead (never follow suit), so you only need to analyze contexts where you lead dominoes.

## Why It Was Deleted

The Perfects feature violated the architectural principle that **the client should be dumb** - it should receive pre-computed data from the server, not compute game logic itself.

Specifically, `src/lib/utils/dominoHelpers.ts` re-implemented trump and suit logic:
- `isTrump()` - duplicated game rule
- `getPlayContexts()` - duplicated led suit logic
- `computeExternalBeaters()` - AI-level analysis in UI

This created a parallel implementation of game rules outside the `GameRules` interface, violating the "single source of truth" principle.

## Files That Were Deleted

### UI Components (249 lines)
- `src/PerfectsApp.svelte` (203 lines) - Main Svelte app displaying partitions
- `src/lib/components/PerfectHandDisplay.svelte` (46 lines) - Hand display component

### UI Utilities (152 lines)
- `src/lib/utils/dominoHelpers.ts` (99 lines) - Game logic re-implemented for UI
- `src/lib/utils/domino-sort.ts` (53 lines) - Trump-aware domino sorting

### Scripts (1,167 lines)
- `scripts/find-perfect-hands.ts` (415 lines) - Find perfect hands for a trump
- `scripts/find-perfect-partition.ts` (295 lines) - Find 3-way partitions where all 3 hands are perfect
- `scripts/find-3hand-leftover.ts` (457 lines) - Analyze the 4th player's leftover hand

### Data Files (~1.6 MB)
- `data/perfect-hands.json` (88 KB) - Pre-computed perfect hands
- `data/3hand-partitions.json` (1.5 MB) - Pre-computed 3-hand partitions

### Tests
- `src/tests/e2e/perfects-page.spec.ts` - E2E tests for the Perfects page

### Documentation (kept for history)
- `docs/archive/perfect-hand-plan2.md` - Original design document

**Total: ~1,568 lines of code + 1.6 MB of data**

## Git History

### First Introduced
```
c7afb0e interim - perfect hands (Sep 27, 2025)
  - docs/perfect-hand-plan.md (334 lines)
  - docs/perfect-hand-plan2.md (303 lines)
  - scripts/find-perfect-hands.ts (358 lines)
```

### Key Commits
```
c7afb0e interim - perfect hands
d20cbde interim - perfect hands
ff3def3 Refactor PerfectsApp to display one hand per page
```

### Last Modified
Various "green" commits through `b57ff61`

## The Algorithm (For Reference)

The perfect hand detection worked as follows:

```typescript
function isGoldPerfectHand(hand: Domino[], trump: TrumpSelection): boolean {
  // 1. Must have 4+ trumps
  const trumpCount = hand.filter(d => isTrump(d, trump)).length;
  if (trumpCount < 4) return false;

  // 2. For each domino, check what can beat it when LED
  for (const domino of hand) {
    const strength = getDominoStrength(domino, trump, getLedSuit(domino));

    // If any external domino can beat this, not perfect
    const externalBeaters = strength.beatenBy.filter(id => !handIds.has(id));
    if (externalBeaters.length > 0) {
      // Exception: if all beaters are in hand, we control play order
      // (play beaters first, then this domino is safe)
    }
  }

  return true;
}
```

## If This Feature Is Needed Again

If perfect hand analysis is needed in the future, it should be implemented as:

1. **A server-side tool** - Scripts in `scripts/` are fine
2. **A separate repository** - Not in the game UI
3. **An API endpoint** - Server computes, client displays

The client should never import from `src/game/core/` or implement rule logic.
