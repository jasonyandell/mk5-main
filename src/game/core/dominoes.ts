import type {
  Domino,
  TrumpSelection,
  RegularSuit,
  TrumpSuitOrNone
} from '../types';
import { CALLED, NO_TRUMP, TRUMP_NOT_SELECTED } from '../types';
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
    case 'doubles': return CALLED;
    case 'no-trump': return NO_TRUMP;
    case 'nello': return TRUMP_NOT_SELECTED; // Nello has no trump
    case 'sevens': return TRUMP_NOT_SELECTED; // Sevens has no trump hierarchy
  }
}

// ============================================================================
// Trump Validation Utilities
// ============================================================================

/**
 * Checks if a trump suit value represents a regular suit (0-6)
 *
 * Type predicate allows TypeScript to narrow the type to RegularSuit
 *
 * @example
 * if (isRegularSuitTrump(trumpSuit)) {
 *   // TypeScript knows trumpSuit is RegularSuit (0-6)
 *   const ledSuit = trumpSuit as LedSuit;
 * }
 */
export function isRegularSuitTrump(trumpSuit: TrumpSuitOrNone): trumpSuit is RegularSuit {
  return trumpSuit >= 0 && trumpSuit <= 6;
}

/**
 * Checks if trump is the special "doubles as trump" value (7)
 */
export function isDoublesTrump(trumpSuit: TrumpSuitOrNone): boolean {
  return trumpSuit === CALLED;
}

/**
 * Checks if trump is the special "no trump" value (8)
 */
export function isNoTrump(trumpSuit: TrumpSuitOrNone): boolean {
  return trumpSuit === NO_TRUMP;
}

/**
 * Checks if trump has been selected (not TRUMP_NOT_SELECTED/-1)
 */
export function isTrumpSelected(trumpSuit: TrumpSuitOrNone): boolean {
  return trumpSuit !== TRUMP_NOT_SELECTED;
}

// ============================================================================
// Domino Suit Utilities
// ============================================================================

/**
 * Checks if a domino contains a specific suit (pip value)
 *
 * This is a simple pip check - does NOT account for trump or special rules.
 * For game logic that respects trump/layers, use GameRules.canFollow instead.
 *
 * @example
 * dominoHasSuit({ high: 6, low: 4 }, 6) // true
 * dominoHasSuit({ high: 6, low: 4 }, 5) // false
 */
export function dominoHasSuit(domino: Domino, suit: number): boolean {
  return domino.high === suit || domino.low === suit;
}

/**
 * Checks if a domino does NOT contain a specific suit
 *
 * Useful for filtering out trump dominoes from suit collections.
 * Inverse of dominoHasSuit.
 *
 * @example
 * const nonTrumpDominoes = suitDominoes.filter(d => dominoLacksSuit(d, trumpSuit));
 */
export function dominoLacksSuit(domino: Domino, suit: number): boolean {
  return domino.high !== suit && domino.low !== suit;
}

/**
 * Gets the non-suit pip value when domino has suit on only one pip
 *
 * Returns null if domino doesn't have the suit or has it on both pips.
 * Useful for strength calculations where the "other" pip matters.
 *
 * @example
 * getNonSuitPip({ high: 6, low: 4 }, 6) // 4 (high is 6, so return low)
 * getNonSuitPip({ high: 6, low: 4 }, 4) // 6 (low is 4, so return high)
 * getNonSuitPip({ high: 6, low: 6 }, 6) // null (both pips are 6)
 * getNonSuitPip({ high: 6, low: 4 }, 5) // null (doesn't have suit 5)
 */
export function getNonSuitPip(domino: Domino, suit: number): number | null {
  if (domino.high === suit && domino.low !== suit) return domino.low;
  if (domino.low === suit && domino.high !== suit) return domino.high;
  return null;
}

/**
 * Determines if a domino is a double
 */
export function isDouble(domino: Domino): boolean {
  return domino.high === domino.low;
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