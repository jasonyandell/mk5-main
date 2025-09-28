/**
 * Gold Perfect Hand Finder for Texas 42
 *
 * Implementation based on docs/perfect-hand-plan2.md
 *
 * A gold perfect hand is a combination of 7 dominoes that guarantees winning all 7 tricks
 * when played optimally. Requirements:
 * - Must have 4+ trumps to guarantee trump control
 * - Trumps must be the highest available (no external dominoes can beat them)
 * - Non-trumps can only be beaten by trumps or dominoes in the hand
 */

import type { Domino, TrumpSelection, LedSuitOrNone } from '../src/game/types.js';
import { getDominoStrength } from '../src/game/ai/strength-table.generated.js';
import { isTrump } from '../src/game/core/dominoes.js';
import { PLAYED_AS_TRUMP } from '../src/game/types.js';

// Helper to create a domino
function createDomino(high: number, low: number): Domino {
  return {
    id: `${high}-${low}`,
    high,
    low
  };
}

// Generate all 28 dominoes
function generateAllDominoes(): Domino[] {
  const dominoes: Domino[] = [];
  for (let high = 6; high >= 0; high--) {
    for (let low = high; low >= 0; low--) {
      dominoes.push(createDomino(high, low));
    }
  }
  return dominoes;
}

// Generator for combinations
function* generateCombinations(arr: Domino[], k: number): Generator<Domino[]> {
  const n = arr.length;
  const indices = Array.from({ length: k }, (_, i) => i);

  yield indices.map(i => arr[i]!);

  while (true) {
    let i = k - 1;
    while (i >= 0 && indices[i] === n - k + i) i--;

    if (i < 0) break;

    indices[i]++;
    for (let j = i + 1; j < k; j++) {
      indices[j] = indices[j - 1] + 1;
    }

    yield indices.map(i => arr[i]!);
  }
}

// Get all possible play contexts for a domino when LEADING
// In a perfect hand, we always lead (since we win every trick)
function getPlayContexts(domino: Domino, trump: TrumpSelection): LedSuitOrNone[] {
  const contexts: LedSuitOrNone[] = [];

  // If it's trump, it can be played as trump
  if (isTrump(domino, trump)) {
    contexts.push(PLAYED_AS_TRUMP);
  }

  // When we LEAD (which is all that matters for perfect hands):
  if (domino.high === domino.low) {
    // Double leads as its suit
    // Exception: when doubles are trump, they lead as doubles (7)
    if (trump.type === 'doubles') {
      contexts.push(7 as LedSuitOrNone);
    } else {
      contexts.push(domino.high as LedSuitOrNone);
    }
  } else {
    // Non-double ALWAYS leads as its HIGH pip
    // We never lead as the low pip!
    contexts.push(domino.high as LedSuitOrNone);
  }

  return contexts;
}

// Parse domino ID to get domino object
function parseDominoId(id: string): Domino {
  const [high, low] = id.split('-').map(Number);
  return createDomino(high!, low!);
}

/**
 * Check if a hand is a gold perfect hand
 *
 * Per plan2.md, the algorithm must:
 * 1. Check we have 4+ trumps (for trump control)
 * 2. Check our trumps cannot be beaten by external dominoes
 * 3. Check non-trumps can only be beaten by trumps or dominoes in hand
 */
function isGoldPerfectHand(
  hand: Domino[],
  trump: TrumpSelection
): boolean {
  // Gold perfect applies to both suit trump and doubles trump
  // Only exclude no-trump games
  if (trump.type === 'no-trump') {
    return false;
  }

  const handIds = new Set(hand.map(d => d.id));
  const trumpDominoes = hand.filter(d => isTrump(d, trump));

  // Must have 4+ trumps for control
  if (trumpDominoes.length < 4) {
    return false;
  }

  // Check 1: Our trumps cannot be beaten by external dominoes
  // This is critical - we must have the HIGHEST trumps
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
      return false; // External domino can beat our trump
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

// Check if a hand is platinum perfect (no external beaters at all)
function isPlatinumPerfectHand(
  hand: Domino[],
  trump: TrumpSelection
): boolean {
  const handIds = new Set(hand.map(d => d.id));

  // Collect ALL dominoes that can beat ANY domino in this hand
  const allBeaters = new Set<string>();

  for (const domino of hand) {
    const contexts = getPlayContexts(domino, trump);

    for (const context of contexts) {
      const strength = getDominoStrength(domino, trump, context);

      if (strength) {
        for (const beater of strength.beatenBy) {
          allBeaters.add(beater);
        }
      }
    }
  }

  // Perfect hand if ALL beaters are within the hand itself
  for (const beater of allBeaters) {
    if (!handIds.has(beater)) {
      return false; // Found external domino that can beat
    }
  }

  return true;
}

// Get trump name for display
function getTrumpName(trump: TrumpSelection): string {
  switch (trump.type) {
    case 'no-trump': return 'no-trump';
    case 'doubles': return 'doubles';
    case 'suit':
      const suitNames = ['blanks', 'aces', 'deuces', 'tres', 'fours', 'fives', 'sixes'];
      return suitNames[trump.suit!] || `suit-${trump.suit}`;
    default: return 'unknown';
  }
}

// Format a domino with trump pip first (if it contains trump)
function formatDomino(domino: Domino, trump: TrumpSelection): string {
  if (trump.type === 'suit' && (domino.high === trump.suit || domino.low === trump.suit)) {
    // This domino contains trump - show trump pip first
    if (domino.high === trump.suit) {
      return `${domino.high}-${domino.low}`;
    } else {
      return `${domino.low}-${domino.high}`;
    }
  }
  // Non-trump or no-trump/doubles - show in normal high-low format
  return `${domino.high}-${domino.low}`;
}

// Format and sort a hand: trumps first (high to low), then non-trumps (high to low)
function formatHand(hand: Domino[], trump: TrumpSelection): string {
  const trumpDominoes = hand.filter(d => isTrump(d, trump));
  const nonTrumpDominoes = hand.filter(d => !isTrump(d, trump));

  // Sort trumps by value (highest first)
  // For suit trump, doubles are highest, then by pip total
  trumpDominoes.sort((a, b) => {
    if (trump.type === 'suit') {
      const aIsDouble = a.high === a.low;
      const bIsDouble = b.high === b.low;
      if (aIsDouble && !bIsDouble) return -1;
      if (!aIsDouble && bIsDouble) return 1;
      // Both doubles or both non-doubles - sort by total
      return (b.high + b.low) - (a.high + a.low);
    } else if (trump.type === 'doubles') {
      // For doubles trump, sort by pip value
      return b.high - a.high;
    }
    return 0;
  });

  // Sort non-trumps by value (highest first)
  nonTrumpDominoes.sort((a, b) => {
    // Doubles first, then by total
    const aIsDouble = a.high === a.low;
    const bIsDouble = b.high === b.low;
    if (aIsDouble && !bIsDouble) return -1;
    if (!aIsDouble && bIsDouble) return 1;
    return (b.high + b.low) - (a.high + a.low);
  });

  // Combine and format
  const sortedHand = [...trumpDominoes, ...nonTrumpDominoes];
  return sortedHand.map(d => formatDomino(d, trump)).join(', ');
}

// Main search function
async function findPerfectHands() {
  console.log('Gold Perfect Hand Finder for Texas 42');
  console.log('=====================================');
  console.log('Based on docs/perfect-hand-plan2.md\n');

  const allDominoes = generateAllDominoes();
  console.log(`Generated ${allDominoes.length} dominoes`);

  const totalCombinations = 1184040; // C(28, 7)
  console.log(`Searching ${totalCombinations.toLocaleString()} possible hands...\n`);

  // Define all trump types to test (focus on suit trumps for gold perfect)
  const trumpTypes: TrumpSelection[] = [
    { type: 'suit', suit: 0 }, // blanks
    { type: 'suit', suit: 1 }, // aces
    { type: 'suit', suit: 2 }, // deuces
    { type: 'suit', suit: 3 }, // tres
    { type: 'suit', suit: 4 }, // fours
    { type: 'suit', suit: 5 }, // fives
    { type: 'suit', suit: 6 }, // sixes
    { type: 'doubles' },        // doubles (for platinum only)
    { type: 'no-trump' },       // no-trump (for platinum only)
  ];

  const allPlatinum: { trump: string; hand: string }[] = [];
  const allGold: { trump: string; hand: string }[] = [];

  for (const trump of trumpTypes) {
    const trumpName = getTrumpName(trump);
    console.log(`\nSearching for ${trumpName} perfect hands...`);

    let platinumCount = 0;
    let goldCount = 0;
    let handsChecked = 0;

    for (const hand of generateCombinations(allDominoes, 7)) {
      handsChecked++;

      // Show progress every 100,000 hands
      if (handsChecked % 100000 === 0) {
        process.stdout.write(`\r  Checked ${handsChecked.toLocaleString()} hands...`);
      }

      // Check for platinum first (stricter condition)
      if (isPlatinumPerfectHand(hand, trump)) {
        const handStr = formatHand(hand, trump);
        console.log(`\n  💎 Platinum: ${handStr}`);
        platinumCount++;
        allPlatinum.push({ trump: trumpName, hand: handStr });
      }
      // Gold perfect applies to suit trump and doubles trump
      else if (trump.type !== 'no-trump' && isGoldPerfectHand(hand, trump)) {
        const handStr = formatHand(hand, trump);
        console.log(`\n  🥇 Gold: ${handStr}`);
        goldCount++;
        allGold.push({ trump: trumpName, hand: handStr });
      }
    }

    console.log(`\n  ✅ Found ${platinumCount} platinum and ${goldCount} gold perfect hands for ${trumpName}`);
  }

  // Final summary
  console.log('\n========================================');
  console.log('FINAL SUMMARY');
  console.log('========================================');
  console.log(`Total Platinum Perfect Hands: ${allPlatinum.length}`);
  console.log(`Total Gold Perfect Hands: ${allGold.length}`);

  if (allPlatinum.length > 0) {
    console.log('\nAll Platinum Perfect Hands:');
    for (const { trump, hand } of allPlatinum) {
      console.log(`  ${trump}: ${hand}`);
    }
  }

  if (allGold.length > 0) {
    console.log('\nAll Gold Perfect Hands:');
    console.log('(Hands with 4+ highest trumps where non-trumps are safe)\n');
    for (const { trump, hand } of allGold) {
      console.log(`  ${trump}: ${hand}`);
    }
  }
}

// Run the search
findPerfectHands().catch(console.error);