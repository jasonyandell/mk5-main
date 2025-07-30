import type { Domino, Trump } from '../types';
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
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Shuffles an array using Fisher-Yates algorithm (legacy - uses Math.random)
 */
function shuffle<T>(array: T[]): T[] {
  const shuffled = [...array];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  return shuffled;
}

/**
 * Shuffles and returns all 28 dominoes (legacy - uses Math.random)
 */
export function shuffleDominoes(): Domino[] {
  return shuffle(createDominoes());
}

/**
 * Shuffles and returns all 28 dominoes with a seed for deterministic results
 */
export function shuffleDominoesWithSeed(seed: number): Domino[] {
  return shuffleWithSeed(createDominoes(), seed);
}

/**
 * Deals dominoes to 4 players (7 each) - legacy version using Math.random
 */
export function dealDominoes(): [Domino[], Domino[], Domino[], Domino[]] {
  const dominoes = shuffle(createDominoes());
  
  return [
    dominoes.slice(0, 7),
    dominoes.slice(7, 14),
    dominoes.slice(14, 21),
    dominoes.slice(21, 28)
  ];
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
 * Converts Trump type to numeric value
 */
function trumpToNumber(trump: Trump | number | null): number | null {
  if (trump === null) return null;
  if (typeof trump === 'number') return trump;
  if (typeof trump === 'object' && 'suit' in trump) {
    // If suit is already a number, return it
    if (typeof trump.suit === 'number') {
      return trump.suit;
    }
    // Convert string suit to number
    const suitMap: Record<string, number | null> = {
      'blanks': 0, 'ones': 1, 'twos': 2, 'threes': 3, 
      'fours': 4, 'fives': 5, 'sixes': 6, 'no-trump': 8, 'doubles': 7
    };
    const result = suitMap[trump.suit];
    return result !== undefined ? result : 0;
  }
  return trump as number;
}

/**
 * Gets the suit of a domino based on trump
 */
export function getDominoSuit(domino: Domino, trump: Trump | number | null): number {
  const numericTrump = trumpToNumber(trump);
  
  // Special case: doubles trump (numericTrump = 7)
  if (numericTrump === 7) {
    if (domino.high === domino.low) {
      return 7; // All doubles are trump
    }
    // Non-doubles just use their higher value
    return Math.max(domino.high, domino.low);
  }
  
  // Doubles belong to their natural suit (unless doubles are trump)
  if (domino.high === domino.low) {
    return domino.high; // Natural suit for doubles
  }
  
  // Non-doubles: if either end matches trump, it's trump suit
  if (numericTrump !== null && (domino.high === numericTrump || domino.low === numericTrump)) {
    return numericTrump;
  }
  
  // Otherwise, suit is determined by higher value
  return Math.max(domino.high, domino.low);
}

/**
 * Checks if a domino can follow a specific suit (contains that number)
 */
export function canDominoFollowSuit(domino: Domino, ledSuit: number, trump: Trump | number | null): boolean {
  const numericTrump = trumpToNumber(trump);
  
  // If the domino is trump, it can follow any suit (but is still trump)
  if (numericTrump !== null && (domino.high === numericTrump || domino.low === numericTrump || 
                              (domino.high === domino.low && domino.high === numericTrump))) {
    return true; // Trump can follow any suit
  }
  
  // Check if the domino contains the led suit number
  return domino.high === ledSuit || domino.low === ledSuit;
}

/**
 * Gets the value of a domino for trick-taking purposes
 */
export function getDominoValue(domino: Domino, trump: Trump | number | null): number {
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
 * Gets point value of a domino for scoring
 * Authoritative Texas 42 scoring from mk4
 */
export function getDominoPoints(domino: Domino): number {
  const total = domino.high + domino.low;
  if (domino.high === 5 && domino.low === 5) return 10; // 5-5 = 10 points
  if ((domino.high === 6 && domino.low === 4) || (domino.high === 4 && domino.low === 6)) return 10; // 6-4 = 10 points
  if (total === 5) return 5; // Any domino totaling 5 pips = 5 points
  return 0;
}

/**
 * Counts the number of doubles in a hand
 */
export function countDoubles(hand: Domino[]): number {
  return hand.filter(isDouble).length;
}