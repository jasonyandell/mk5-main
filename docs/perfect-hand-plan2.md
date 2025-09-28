# Gold Perfect Hands - Problem Statement and Implementation

## Problem Statement

In Texas 42, a **gold perfect hand** is a combination of 7 dominoes that guarantees winning all 7 tricks when played optimally. Gold perfect hands require strategic sequencing and trump exhaustion to achieve victory.

## Key Requirements

1. **Must have 4+ trumps** to guarantee trump control
2. **Trumps must be the highest available** (no external dominoes can beat them)
3. **Non-trumps can only be beaten by trumps or dominoes in the hand**

## Game Mechanics and Justifications

### Perfect Hand Principle
- **In a perfect hand, we win all 7 tricks, meaning we always lead**
- If we ever have to follow suit, we've already lost a trick
- Therefore, we only need to check contexts where we LEAD dominoes

### Leading Rules
- **Non-double dominoes are ALWAYS led as their highest pip value**
  - 6-2 can only be led as a six, NEVER as a two
  - 5-3 can only be led as a five, NEVER as a three
  - This is a fundamental rule of Texas 42
- Doubles are led as their natural suit
- When doubles are trump, they lead as "doubles" (suit 7)

### Trump Distribution
- Each suit contains exactly 7 dominoes
- With 4 players receiving 7 dominoes each (28 total)
- If we hold 4+ trumps, opponents collectively hold at most 3 trumps
- After playing 4 trump tricks with opponents following suit, no opponent can have trumps remaining

### Sequential Safety
- If a domino can only be beaten by dominoes in our hand, we control the play order
- Example: If 6-5 can only be beaten by 6-6 and we have 6-6, we play 6-6 first, then 6-5 is safe

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

export type RegularSuit = 0 | 1 | 2 | 3 | 4 | 5 | 6;
export const PLAYED_AS_TRUMP = -1;
```

### Strength Table Structure
```typescript
// From src/game/ai/strength-table.generated.ts
export interface StrengthEntry {
  beatenBy: string[];      // Domino IDs that can follow AND beat this
  beats: string[];         // Domino IDs that can follow but lose to this
  cannotFollow: string[];  // Domino IDs that cannot follow suit
}

function getDominoStrength(
  domino: Domino,
  trump: TrumpSelection,
  playedAsSuit: LedSuitOrNone
): StrengthEntry | undefined
```

### Helper Functions
```typescript
// From src/game/core/dominoes.ts
function isTrump(domino: Domino, trump: TrumpSelection): boolean

// Create a domino
function createDomino(high: number, low: number): Domino {
  return { id: `${high}-${low}`, high, low };
}
```

## The Algorithm

### Gold Perfect Hand Detection

```typescript
function isGoldPerfectHand(
  hand: Domino[],
  trump: TrumpSelection
): boolean {
  // Gold perfect only applies to suit trump
  if (trump.type !== 'suit') {
    return false;
  }

  const handIds = new Set(hand.map(d => d.id));
  const trumpDominoes = hand.filter(d => isTrump(d, trump));

  // Must have 4+ trumps
  if (trumpDominoes.length < 4) {
    return false;
  }

  // Check 1: Our trumps cannot be beaten by external dominoes
  // Collect all beaters for our trumps
  const trumpBeaters = new Set<string>();

  for (const ourTrump of trumpDominoes) {
    const contexts = getPlayContexts(ourTrump, trump);

    for (const context of contexts) {
      const strength = getDominoStrength(ourTrump, trump, context);

      if (strength) {
        for (const beater of strength.beatenBy) {
          trumpBeaters.add(beater);
        }
      }
    }
  }

  // Any external beater for our trumps = not gold perfect
  for (const beater of trumpBeaters) {
    if (!handIds.has(beater)) {
      return false;  // External domino can beat our trump
    }
  }

  // Check 2: Non-trumps can only be beaten by trumps or dominoes in hand
  const nonTrumpDominoes = hand.filter(d => !isTrump(d, trump));

  for (const nonTrump of nonTrumpDominoes) {
    const contexts = getPlayContexts(nonTrump, trump);

    for (const context of contexts) {
      const strength = getDominoStrength(nonTrump, trump, context);

      if (strength) {
        for (const beaterId of strength.beatenBy) {
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

### Critical: The getPlayContexts Function

**This is the most important part of the algorithm!** The `getPlayContexts` function must ONLY return contexts where we LEAD, not where we might follow suit.

```typescript
// CORRECT implementation - only checks LEAD contexts
function getPlayContexts(domino: Domino, trump: TrumpSelection): LedSuitOrNone[] {
  const contexts: LedSuitOrNone[] = [];

  // If it's trump, it can be played as trump
  if (isTrump(domino, trump)) {
    contexts.push(PLAYED_AS_TRUMP);
  }

  // When we LEAD (which is all that matters for perfect hands):
  if (domino.high === domino.low) {
    // Double leads as its suit
    if (trump.type === 'doubles') {
      contexts.push(7 as LedSuitOrNone);
    } else {
      contexts.push(domino.high as LedSuitOrNone);
    }
  } else {
    // Non-double ALWAYS leads as its HIGH pip (never as low!)
    contexts.push(domino.high as LedSuitOrNone);
    // Do NOT add domino.low here - we never lead as the low pip!
  }

  return contexts;
}
```

**Why this matters**: In a perfect hand, we win all 7 tricks, meaning we ALWAYS lead. If we ever have to follow suit, we've already lost a trick. Therefore, we only need to check contexts where we lead dominoes.

**Common Bug**: Adding both `domino.high` and `domino.low` for non-doubles. This is wrong because:
- When leading, a non-double is ALWAYS played as its high pip
- 6-5 can only be led as a six, NEVER as a five
- Checking it as a five would incorrectly flag it as vulnerable to 5-5

## Critical Pitfalls and Nuances

### 1. Must Have Highest Trumps
**Wrong**: Any 4 trumps work
**Right**: Must have the highest 4+ trumps

Example with blanks trump:
- ❌ [3-0, 2-0, 1-0, 0-0, ...] - Missing 6-0, 5-0, 4-0 which can beat our trumps
- ✅ [0-0, 6-0, 5-0, 4-0, ...] - These are the highest blanks

### 2. The Double is Always Highest
The double of the trump suit is always the highest trump:
- Blanks: 0-0 is highest (not 6-0)
- Aces: 1-1 is highest (not 6-1)
- Sixes: 6-6 is highest (already natural)

### 3. Sequential Play Matters
Non-trumps that can be beaten by dominoes in our hand are safe:
- [6-6, 6-5, 6-4] - Play 6-6 first, then 6-5 (now highest), then 6-4
- [6-6, 6-4, 6-2] - NOT safe! 6-5 externally can beat 6-4

### 4. Non-Doubles ONLY Play High When Leading
Since perfect hands always lead (never follow):
- A 6-2 is ONLY checked as a six (its high pip)
- We NEVER check it as a two (it's never led as a two)
- Example: 6-5 is only vulnerable to dominoes that beat sixes, NOT to 5-5

### 5. External Trumps Are OK for Non-Trumps Only
- **Trumps**: Cannot be beaten by ANY external domino
- **Non-Trumps**: Can be beaten by external trumps (they'll be exhausted) but NOT external non-trumps

## Examples

### Valid Gold Perfect Hands

**Blanks Trump**: [0-0, 6-0, 5-0, 4-0, 6-6, 6-5, 6-4]
- Has top 4 blanks ✅
- 6-6: beatenBy [] when led as six ✅
- 6-5: beatenBy [6-6] when led as six - in hand ✅ (NOT checked as five!)
- 6-4: beatenBy [6-6, 6-5] when led as six - both in hand ✅

**Aces Trump**: [1-1, 6-1, 5-1, 4-1, 6-6, 5-5, 4-4]
- Has top 4 aces ✅
- Doubles cannot be beaten when led as their suit ✅

### Invalid Gold Perfect Hands

**Missing Highest Trump**: [6-0, 5-0, 4-0, 3-0, 6-6, 6-5, 6-4] with blanks
- Missing 0-0 (highest blank) ❌
- 0-0 can beat all our trumps

**Gap in Sequence**: [0-0, 6-0, 5-0, 4-0, 6-6, 6-4, 6-2] with blanks
- 6-4: beatenBy includes 6-5 which is external ❌

**Wrong Trump Count**: [0-0, 6-0, 5-0, 6-6, 5-5, 4-4, 3-3] with blanks
- Only 3 trumps ❌ (need 4+ for control)

## Performance Optimizations

1. **Use Sets for O(1) lookups** instead of repeated array searches
2. **Union all beaters first**, then check membership once
3. **Early termination** on first external non-trump beater found
4. **Cache trump checks** to avoid repeated isTrump() calls

## Search Strategy

1. **Generate all C(28,7) = 1,184,040 combinations**
2. **Filter by trump count** (must have 4+)
3. **Check trump safety** (no external beaters)
4. **Check non-trump safety** (only beaten by trumps or hand dominoes)
5. **Track and report** all gold perfect hands found

## Output Formatting Recommendations

When displaying perfect hands, format them for clarity:

1. **Sort the hand**:
   - Display trumps first (sorted high to low)
   - Then display non-trumps (sorted high to low)

2. **Format dominoes with trump pip first**:
   - If blanks are trump: write `0-6` not `6-0`
   - If aces are trump: write `1-6` not `6-1`
   - This makes it immediately clear which dominoes are trumps

3. **Example formatting**:
   ```
   Original: [6-6, 6-5, 6-0, 5-5, 5-0, 4-0, 0-0]
   Formatted: [0-0, 0-6, 0-5, 0-4, 6-6, 5-5, 6-5] (with blanks trump)
   ```

## Expected Results

Gold perfect hands are extremely rare because they require:
- The highest 4+ trumps (no higher trump can exist externally)
- Non-trumps that form safe sequences or are protected by hand dominoes
- Typical patterns include:
  - Top 4 trumps + 3 consecutive high cards (e.g., [0-0, 0-6, 0-5, 0-4, 6-6, 6-5, 6-4] with blanks)
  - Top 4 trumps + 3 high doubles
  - Top 5-7 trumps + remaining high cards

With the corrected algorithm (only checking lead contexts), approximately 42-45 gold perfect hands exist per trump suit.