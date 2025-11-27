# Perfect Hands System

A "perfect hand" in Texas 42 is a 7-domino combination that guarantees winning all 7 tricks when played optimally.

## Types

- **Platinum**: No external domino can beat any domino in the hand
- **Gold**: Has 4+ highest trumps; non-trumps can only be beaten by trumps (which get exhausted)

## Data Files

Two generated JSON files power the `/perfects` page:

- `data/perfect-hands.json` - All platinum/gold hands for each trump type
- `data/3hand-partitions.json` - Combinations of 3 perfect hands (21 dominoes) plus leftover analysis

These are **not committed to git** - they're deterministic outputs (~1.6MB total).

## Regeneration

```bash
npm run generate:perfects
```

This runs two scripts:
1. `scripts/find-perfect-hands.ts` - Searches all C(28,7) = 1.18M combinations
2. `scripts/find-3hand-leftover.ts` - Finds non-overlapping 3-hand partitions

Takes a few minutes to complete.

## Implementation

- `src/PerfectsApp.svelte` - Visualization page at `/perfects`
- `scripts/find-perfect-hands.ts` - Core detection algorithm using strength tables
