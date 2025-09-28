# Perfect Hand Finder - Problem Statement and Plan

## Problem Statement

In the game of Texas 42, we need to find "perfect hands" - combinations of 7 dominoes that guarantee winning all 7 tricks when played optimally.

### Key Definitions
- **Hand**: A combination of exactly 7 dominoes from the 28 total dominoes
- **Platinum Perfect Hand**: A hand where no domino can be beaten by dominoes not in the hand (wins regardless of play order)
- **Gold Perfect Hand**: A hand with 4+ trumps where each non-trump domino can only be beaten by trumps or other dominoes in the hand (wins with optimal sequencing)
- **Trump Types**: no-trump, doubles, blanks (0), aces (1), deuces (2), tres (3), fours (4), fives (5), sixes (6)
- **Total Dominoes**: 28 dominoes in a double-six set

### Domino Set
The 28 dominoes are:
- 6-6, 6-5, 6-4, 6-3, 6-2, 6-1, 6-0
- 5-5, 5-4, 5-3, 5-2, 5-1, 5-0
- 4-4, 4-3, 4-2, 4-1, 4-0
- 3-3, 3-2, 3-1, 3-0
- 2-2, 2-1, 2-0
- 1-1, 1-0
- 0-0

## Code References

### Core Types
```typescript
// From src/game/types.ts
export interface Domino {
  high: number;
  low: number;
  id: string | number;
  points?: number;
}

export interface TrumpSelection {
  type: 'not-selected' | 'suit' | 'doubles' | 'no-trump';
  suit?: RegularSuit;  // 0-6 when type === 'suit'
}
```

### Strength Table Structure
```typescript
// From src/game/ai/strength-table.generated.ts
export interface StrengthEntry {
  beatenBy: string[];      // Domino IDs that can follow AND beat this
  beats: string[];         // Domino IDs that can follow but lose to this
  cannotFollow: string[];  // Domino IDs that cannot follow suit
}

// Example entry:
"6-6|trump-sixes|led-sixes": {
  beatenBy: [],  // Nothing beats 6-6 when sixes are trump
  beats: ["6-5", "6-4", "6-3", "6-2", "6-1", "6-0"],
  cannotFollow: [/* all non-six dominoes */]
}
```

### Domino Creation Helper
```typescript
function createDomino(high: number, low: number): Domino {
  return {
    id: `${high}-${low}`,
    high,
    low
  };
}
```

## Algorithm Design

### 1. Data Preparation Phase

```typescript
// Generate all 28 dominoes
const allDominoes: Domino[] = [];
for (let high = 6; high >= 0; high--) {
  for (let low = high; low >= 0; low--) {
    allDominoes.push(createDomino(high, low));
  }
}

// Convert strength table to use Domino objects
// Create Map<string, Set<string>> for O(1) lookups
const strengthLookup = new Map<string, {
  beatenBySet: Set<string>,
  original: StrengthEntry
}>();
```

### 2. Platinum Perfect Hand Detection

```typescript
function isPlatinumPerfectHand(
  hand: Domino[],
  trump: TrumpSelection,
  strengthLookup: Map
): boolean {
  const handIds = new Set(hand.map(d => d.id));

  // Collect ALL dominoes that can beat ANY domino in this hand
  const allBeaters = new Set<string>();

  // For each domino in the hand
  for (const domino of hand) {
    // Check all possible play contexts
    const contexts = getPlayContexts(domino, trump);

    for (const context of contexts) {
      const key = `${domino.id}|${trumpKey}|${context}`;
      const strength = strengthLookup.get(key);

      if (strength) {
        // Add all beaters to our union set
        for (const beater of strength.beatenBySet) {
          allBeaters.add(beater);
        }
      }
    }
  }

  // Perfect hand if ALL beaters are within the hand itself
  // i.e., no external domino can beat any domino in this hand
  for (const beater of allBeaters) {
    if (!handIds.has(beater)) {
      return false; // Found external domino that can beat
    }
  }

  return true;
}
```

### 3. Gold Perfect Hand Detection

```typescript
function isGoldPerfectHand(
  hand: Domino[],
  trump: TrumpSelection,
  strengthLookup: Map
): boolean {
  // Gold perfect only applies to suit trump
  if (trump.type !== 'suit') {
    return false;
  }

  const handIds = new Set(hand.map(d => d.id));
  const trumpDominoes = hand.filter(d => isTrump(d, trump));

  // Must have 4+ trumps to guarantee trump control
  // With 7 total trumps in the game and 4 in our hand,
  // opponents have at most 3 trumps combined
  if (trumpDominoes.length < 4) {
    return false;
  }

  // Collect all beaters efficiently using Sets
  const allBeaters = new Set<string>();

  // Check 1: Our trumps cannot be beaten by external dominoes
  // We must have the highest trumps (e.g., 0-0 for blanks, 1-1 for aces)
  for (const ourTrump of trumpDominoes) {
    const contexts = getPlayContexts(ourTrump, trump);

    for (const context of contexts) {
      const strength = strengthLookup.get(`${ourTrump.id}|${trumpKey}|${context}`);

      if (strength) {
        for (const beater of strength.beatenBySet) {
          allBeaters.add(beater);
        }
      }
    }
  }

  // Any external beater for our trumps = not gold perfect
  for (const beater of allBeaters) {
    if (!handIds.has(beater)) {
      return false;  // External domino can beat our trump
    }
  }

  // Check 2: Non-trumps can only be beaten by trumps or dominoes in hand
  const nonTrumpDominoes = hand.filter(d => !isTrump(d, trump));

  for (const nonTrump of nonTrumpDominoes) {
    const contexts = getPlayContexts(nonTrump, trump);

    for (const context of contexts) {
      const strength = strengthLookup.get(`${nonTrump.id}|${trumpKey}|${context}`);

      if (strength) {
        for (const beaterId of strength.beatenBySet) {
          if (!handIds.has(beaterId)) {
            // External beater exists - must be a trump
            const beaterDomino = parseDominoId(beaterId);
            if (!isTrump(beaterDomino, trump)) {
              // External non-trump can beat our non-trump
              return false;
            }
            // External trump is acceptable (will be exhausted)
          }
          // Beater in our hand is acceptable (we control sequencing)
        }
      }
    }
  }

  return true;
}
```

### 4. Combination Generation

```typescript
// Generate combinations using iterative approach for memory efficiency
function* generateCombinations(arr: Domino[], k: number) {
  const n = arr.length;
  const indices = Array.from({length: k}, (_, i) => i);

  yield indices.map(i => arr[i]);

  while (true) {
    let i = k - 1;
    while (i >= 0 && indices[i] === n - k + i) i--;

    if (i < 0) break;

    indices[i]++;
    for (let j = i + 1; j < k; j++) {
      indices[j] = indices[j - 1] + 1;
    }

    yield indices.map(i => arr[i]);
  }
}
```

### 5. Main Search Loop

```typescript
const trumpTypes: TrumpSelection[] = [
  { type: 'no-trump' },
  { type: 'doubles' },
  { type: 'suit', suit: 0 }, // blanks
  { type: 'suit', suit: 1 }, // aces
  { type: 'suit', suit: 2 }, // deuces
  { type: 'suit', suit: 3 }, // tres
  { type: 'suit', suit: 4 }, // fours
  { type: 'suit', suit: 5 }, // fives
  { type: 'suit', suit: 6 }, // sixes
];

for (const trump of trumpTypes) {
  console.log(`\nSearching for ${getTrumpName(trump)} perfect hands...`);
  let platinumCount = 0;
  let goldCount = 0;

  for (const hand of generateCombinations(allDominoes, 7)) {
    if (isPlatinumPerfectHand(hand, trump, strengthLookup)) {
      console.log(`Platinum: ${getTrumpName(trump)}, Hand: ${hand.map(d => d.id).join(', ')}`);
      platinumCount++;
    } else if (isGoldPerfectHand(hand, trump, strengthLookup)) {
      console.log(`Gold: ${getTrumpName(trump)}, Hand: ${hand.map(d => d.id).join(', ')}`);
      goldCount++;
    }
  }

  console.log(`Found ${platinumCount} platinum and ${goldCount} gold perfect hands for ${getTrumpName(trump)}`);
}
```

## Performance Considerations

1. **Total Combinations**: C(28, 7) = 1,184,040 hands to check
2. **Contexts per Domino**: Up to 3 contexts (played as trump, led as high suit, led as low suit)
3. **Total Operations**: ~1.2M hands × union building × membership checks

### Key Optimizations
- **Union-based approach**: Collect all beaters into a single Set, then check if any are external
- Use Set for O(1) membership testing
- Pre-compute all strength data before search
- Early termination on first external beater found
- Generator functions to avoid memory overhead
- Cache trump-specific keys to avoid string concatenation

## Expected Output Format

```
Platinum: sixes, Hand: "6-6", "6-5", "6-4", "6-3", "6-2", "6-1", "6-0"
Platinum: doubles, Hand: "6-6", "5-5", "4-4", "3-3", "2-2", "1-1", "0-0"
Gold: blanks, Hand: "0-0", "6-0", "5-0", "4-0", "3-3", "2-2", "1-1"
Gold: blanks, Hand: "0-0", "6-0", "5-0", "4-0", "6-6", "6-5", "6-4"
```

### Example Analysis

**Gold Perfect Hand**: [0-0, 6-0, 5-0, 4-0, 6-6, 6-5, 6-4] with blanks trump
- 4 blanks (trump control) ✓
- 6-6: beatenBy [] when led as six ✓
- 6-5: beatenBy [6-6] when led as six - in hand ✓ (Note: 6-5 is NEVER led as a five!)
- 6-4: beatenBy [6-6, 6-5] when led as six - both in hand ✓
- Result: Gold perfect (play sequence: trumps, then 6-6, 6-5, 6-4)

**Not Gold Perfect**: [0-0, 6-0, 5-0, 4-0, 6-6, 6-4, 6-2] with blanks trump
- 4 blanks (trump control) ✓
- 6-6: beatenBy [] ✓
- 6-4: beatenBy [6-6, 6-5] - 6-5 is external ✗
- Result: Not gold perfect (6-5 can beat 6-4)

## Implementation Notes

1. The strength table uses string IDs like "6-6", not "[6-6]"
2. Must handle all play contexts (played-as-trump, led-sixes, led-fives, etc.)
3. The `beatenBy` array only includes dominoes that can legally follow AND beat
4. Platinum hands win regardless of play order
5. Gold hands require optimal play (exhaust trump first)
6. Gold hands only apply to suit trump and require 4+ trumps

## Game Mechanics Justifications

1. **Perfect Hand = Always Leading**: In a perfect hand, we win all 7 tricks, meaning we always lead and never follow suit. If we ever have to follow, we've already lost a trick.
2. **Leading Rules**: When leading, non-double dominoes ALWAYS play as their highest pip value. A 6-2 can only be led as a six, never as a two.
3. **Trump Distribution**: Each suit has exactly 7 dominoes. With 4 players getting 7 dominoes each, if we hold 4+ trumps, opponents collectively hold at most 3 trumps.
4. **Trump Exhaustion**: After playing 4 trump tricks, no opponent can have any trumps remaining.
5. **Sequential Safety**: A non-trump domino that can only be beaten by dominoes in our hand is safe because we control the play order. For example, if 6-5 can only be beaten by 6-6 and we have 6-6, we play 6-6 first.

## Next Steps

1. Create `scripts/find-perfect-hands.ts`
2. Implement both platinum and gold perfect hand detection
3. Run search across all trump types
4. Analyze distribution of platinum vs gold perfect hands
5. Validate gold perfect hands with game simulation if needed