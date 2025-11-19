import type {
  Domino,
  TrumpSelection,
  RegularSuit,
  LedSuit,
  TrumpSuitOrNone
} from '../types';
import { DOUBLES_AS_TRUMP, NO_TRUMP, TRUMP_NOT_SELECTED } from '../types';
import { DOMINO_VALUES } from '../constants';
import { createSeededRandom } from './random';

/**
 * Creates a complete set of 28 dominoes
 */
export function createDominoes(): Domino[] {
  return DOMINO_VALUES.map(([a, b]) => ({
    high: Math.max(a, b),
    low: Math.min(a, b),
    id: `${Math.max(a, b)}-${Math.min(a, b)}`
  }));
}

/**
 * Shuffles an array using Fisher-Yates algorithm with seeded RNG
 */
function shuffleWithSeed<T>(array: T[], seed: number): T[] {
  const rng = createSeededRandom(seed);
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = rng.nextInt(0, i + 1);
    const temp = shuffled[i]!;
    shuffled[i] = shuffled[j]!;
    shuffled[j] = temp;
  }
  return shuffled;
}



/**
 * Shuffles and returns all 28 dominoes with a seed for deterministic results
 */
export function shuffleDominoesWithSeed(seed: number): Domino[] {
  return shuffleWithSeed(createDominoes(), seed);
}


/**
 * Deals dominoes to 4 players (7 each) with a seed for deterministic results
 */
export function dealDominoesWithSeed(seed: number): [Domino[], Domino[], Domino[], Domino[]] {
  const dominoes = shuffleWithSeed(createDominoes(), seed);

  return [
    dominoes.slice(0, 7),
    dominoes.slice(7, 14),
    dominoes.slice(14, 21),
    dominoes.slice(21, 28)
  ];
}

/**
 * Converts TrumpSelection to TrumpSuit
 */
export function getTrumpSuit(trump: TrumpSelection): TrumpSuitOrNone {
  switch (trump.type) {
    case 'not-selected': return TRUMP_NOT_SELECTED;
    case 'suit': return trump.suit!;
    case 'doubles': return DOUBLES_AS_TRUMP;
    case 'no-trump': return NO_TRUMP;
    case 'nello': return TRUMP_NOT_SELECTED; // Nello has no trump
    case 'sevens': return TRUMP_NOT_SELECTED; // Sevens has no trump hierarchy
  }
}

/**
 * Legacy function for backward compatibility - will be removed
 * @deprecated Use getTrumpSuit instead
 */
export function trumpToNumber(trump: TrumpSelection): number | null {
  const suit = getTrumpSuit(trump);
  return suit === TRUMP_NOT_SELECTED ? null : suit;
}

/**
 * Gets what suit a domino leads when played
 */
export function getLedSuit(domino: Domino, trump: TrumpSelection): LedSuit {
  const trumpSuit = getTrumpSuit(trump);

  // When doubles are trump, doubles lead suit 7
  if (trumpSuit === DOUBLES_AS_TRUMP) {
    if (domino.high === domino.low) {
      return 7;  // Doubles lead the doubles suit
    }
    // Non-doubles lead their higher pip
    return Math.max(domino.high, domino.low) as RegularSuit;
  }

  // When a regular suit is trump (0-6)
  if (trumpSuit >= 0 && trumpSuit <= 6) {
    // Trump dominoes lead trump
    if (domino.high === trumpSuit || domino.low === trumpSuit) {
      return trumpSuit as RegularSuit;
    }
    // Non-trump dominoes lead their natural suit
    if (domino.high === domino.low) {
      return domino.high as RegularSuit;  // Double leads its pip
    }
    return Math.max(domino.high, domino.low) as RegularSuit;
  }

  // No-trump or no trump selected: everything leads naturally
  if (domino.high === domino.low) {
    return domino.high as RegularSuit;  // Doubles lead their pip
  }
  return Math.max(domino.high, domino.low) as RegularSuit;
}

/**
 * Legacy function for backward compatibility - will be removed
 * @deprecated Use getLedSuit instead
 */
export function getDominoSuit(domino: Domino, trump: TrumpSelection): LedSuit {
  return getLedSuit(domino, trump);
}

/**
 * Checks if a domino follows the led suit, handling special cases for Doubles as Trump/Nello.
 * 
 * This is the centralized logic for "following suit".
 * 
 * Rules:
 * 1. If ledSuit is 7 (Doubles), must play a double.
 * 2. If Doubles are their own suit (Nello or Doubles Trump):
 *    - Doubles belong ONLY to suit 7.
 *    - They do NOT follow their natural suit (e.g. 5-5 does not follow 5).
 * 3. If Standard play:
 *    - Doubles belong to their natural suit (e.g. 5-5 follows 5).
 * 
 * @param domino The domino being played
 * @param ledSuit The suit that was led (0-6 or 7 for doubles)
 * @param trump The current trump selection (determines if doubles are their own suit)
 */
export function doesDominoFollowSuit(domino: Domino, ledSuit: number, trump: TrumpSelection): boolean {
  const trumpSuit = getTrumpSuit(trump);
  const doublesAreSuits = trumpSuit === DOUBLES_AS_TRUMP || trump.type === 'nello';

  // Case 1: Doubles led (Suit 7)
  if (ledSuit === 7) {
    // Only doubles follow suit 7
    return domino.high === domino.low;
  }

  // Case 2: Regular suit led (0-6)

  // If doubles are their own suit, they do NOT follow regular suits
  if (doublesAreSuits && domino.high === domino.low) {
    return false;
  }

  // Otherwise, check if domino contains the suit
  return domino.high === ledSuit || domino.low === ledSuit;
}

/**
 * Checks if a domino can follow a specific suit (contains that number)
 * @deprecated Use doesDominoFollowSuit instead for game logic, or check pips directly for UI
 */
export function canDominoFollowSuit(domino: Domino, ledSuit: number, trump: TrumpSelection): boolean {
  return doesDominoFollowSuit(domino, ledSuit, trump);
}

/**
 * Gets the value of a domino for trick-taking purposes
 */
export function getDominoValue(domino: Domino, trump: TrumpSelection): number {
  const numericTrump = trumpToNumber(trump);

  // Special case: doubles trump (when numericTrump === 7)
  if (numericTrump === 7 && domino.high === domino.low) {
    // All doubles are trump, ranked from 6-6 down to 0-0
    return 200 + domino.high;
  }

  // Check if this domino is trump
  if (numericTrump !== null && numericTrump >= 0 && numericTrump <= 6) {
    // Regular suit trump (when numericTrump 0-6)
    if (domino.high === numericTrump || domino.low === numericTrump) {
      // Dominoes containing the trump suit are trump
      // Doubles of the trump suit rank highest
      if (domino.high === domino.low) {
        return 200 + domino.high; // Trump double
      } else {
        return 100 + domino.high + domino.low; // Trump non-double
      }
    }
  }

  // Non-trump dominoes
  if (domino.high === domino.low) {
    // Non-trump doubles are highest in their natural suit
    return 50 + domino.high;
  } else {
    // Non-trump non-doubles - just pip count
    return domino.high + domino.low;
  }
}

/**
 * Determines if a domino is a double
 */
export function isDouble(domino: Domino): boolean {
  return domino.high === domino.low;
}

/**
 * Check if a domino is trump
 */
export function isTrump(domino: Domino, trump: TrumpSelection): boolean {
  const trumpSuit = getTrumpSuit(trump);

  if (trumpSuit === DOUBLES_AS_TRUMP) {
    return domino.high === domino.low;  // All doubles are trump
  }

  if (trumpSuit >= 0 && trumpSuit <= 6) {
    return domino.high === trumpSuit || domino.low === trumpSuit;
  }

  return false;  // No trump or no-trump game
}

/**
 * Gets point value of a domino for scoring
 * Authoritative Texas 42 scoring from mk4
 */
export function getDominoPoints(domino: Domino): number {
  const total = domino.high + domino.low;
  if (domino.high === 5 && domino.low === 5) return 10; // 5-5 = 10 points
  if ((domino.high === 6 && domino.low === 4) || (domino.high === 4 && domino.low === 6)) return 10; // 6-4 = 10 points
  if (total === 5) return 5; // 5-0, 4-1, 3-2 = 5 points each
  return 0;
}

/**
 * Counts the number of doubles in a hand
 */
export function countDoubles(hand: Domino[]): number {
  return hand.filter(isDouble).length;
}