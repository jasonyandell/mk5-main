/**
 * Domino Tables - Precomputed lookup tables for game logic
 *
 * This module implements the "Crystal Palace" - the single source of truth
 * for base game rules via table lookups. See bead t42-9xy3 for the theory.
 *
 * Key insight: Trump conflates two independent operations:
 * 1. Absorption - restructures which dominoes belong to which suit
 * 2. Power - determines which dominoes beat others
 *
 * These are factored into separate table dimensions for clean composition.
 */

import type { Domino, TrumpSelection, LedSuit } from '../types';
import { DOUBLES_AS_TRUMP } from '../types';

// ============= TYPE DEFINITIONS =============

/** Domino index 0-27 (triangular number encoding) */
export type DominoId = number;

/** Pip value 0-6 */
export type Pip = 0 | 1 | 2 | 3 | 4 | 5 | 6;

/**
 * Absorption configuration - determines suit structure
 * 0-6: pip absorption (dominoes containing that pip form absorbed suit)
 * 7: doubles absorption (doubles form their own suit)
 * 8: no absorption (theoretical, used for no-trump)
 */
export type AbsorptionId = number;

/**
 * Power configuration - determines which dominoes beat others
 * 0-6: dominoes containing that pip have power
 * 7: doubles have power
 * 8: nothing has power (nello, no-trump)
 */
export type PowerId = number;

/** Suit index 0-7 (0-6 = pip suits, 7 = absorbed/trump suit) */
export type SuitId = number;

// ============= CONSTANTS =============

/** The absorbed suit index (trump suit, or doubles when doubles are trump) */
export const ABSORBED_SUIT = 7;

/** Pip values for each domino index [low, high] */
export const DOMINO_PIPS: readonly [Pip, Pip][] = (() => {
  const result: [Pip, Pip][] = [];
  for (let hi = 0; hi <= 6; hi++) {
    for (let lo = 0; lo <= hi; lo++) {
      result.push([lo as Pip, hi as Pip]);
    }
  }
  return result;
})();

// ============= CONVERSION FUNCTIONS =============

/**
 * Convert Domino object to table index (0-27)
 * Uses triangular number formula: index = hi*(hi+1)/2 + lo
 */
export function dominoToId(d: Domino): DominoId {
  const lo = Math.min(d.high, d.low);
  const hi = Math.max(d.high, d.low);
  return (hi * (hi + 1)) / 2 + lo;
}

/**
 * Extract absorption configuration from game state's trump selection
 */
export function getAbsorptionId(trump: TrumpSelection): AbsorptionId {
  switch (trump.type) {
    case 'suit':
      return trump.suit!;
    case 'doubles':
    case 'nello':
      return 7; // doubles form separate suit
    case 'no-trump':
    case 'sevens':
    case 'not-selected':
    default:
      return 8; // no absorption
  }
}

/**
 * Extract power configuration from game state's trump selection
 */
export function getPowerId(trump: TrumpSelection): PowerId {
  switch (trump.type) {
    case 'suit':
      return trump.suit!;
    case 'doubles':
      return 7; // doubles have power
    case 'nello':
    case 'no-trump':
    case 'sevens':
    case 'not-selected':
    default:
      return 8; // nothing has power
  }
}

// ============= PRECOMPUTED TABLES =============

/**
 * EFFECTIVE_SUIT[d][absorptionId] -> SuitId (0-7)
 *
 * Determines what suit a domino belongs to given the absorption config.
 * - When absorbed: returns 7 (the absorbed suit)
 * - Otherwise: returns high pip (the domino's natural suit for leading)
 *
 * 28 × 9 = 252 entries
 */
export const EFFECTIVE_SUIT: readonly number[][] = (() => {
  const result: number[][] = [];

  for (let d = 0; d < 28; d++) {
    result[d] = [];
    const [lo, hi] = DOMINO_PIPS[d];

    // Pip absorptions (0-6)
    for (let pip = 0; pip <= 6; pip++) {
      if (lo === pip || hi === pip) {
        result[d][pip] = ABSORBED_SUIT;
      } else {
        result[d][pip] = hi;
      }
    }

    // Doubles absorption (7)
    if (lo === hi) {
      result[d][7] = ABSORBED_SUIT;
    } else {
      result[d][7] = hi;
    }

    // No absorption (8)
    result[d][8] = hi;
  }

  return result;
})();

/**
 * SUIT_MASK[absorptionId][suit] -> bitmask of dominoes that can follow that suit
 *
 * A domino can follow a suit if:
 * - For absorbed suit (7): domino must be absorbed
 * - For regular suit: domino must contain that pip AND not be absorbed
 *
 * 9 × 8 = 72 entries (each entry is a 28-bit mask)
 */
export const SUIT_MASK: readonly number[][] = (() => {
  const result: number[][] = [];

  for (let abs = 0; abs < 9; abs++) {
    result[abs] = [];
    for (let suit = 0; suit < 8; suit++) {
      let mask = 0;
      for (let d = 0; d < 28; d++) {
        const [lo, hi] = DOMINO_PIPS[d];
        const effectiveSuit = EFFECTIVE_SUIT[d][abs];
        const isAbsorbed = (effectiveSuit === ABSORBED_SUIT);

        let canFollow: boolean;
        if (suit === ABSORBED_SUIT) {
          // Absorbed suit led: must be absorbed to follow
          canFollow = isAbsorbed;
        } else if (isAbsorbed) {
          // Domino is absorbed: cannot follow non-absorbed suits
          canFollow = false;
        } else {
          // Non-absorbed domino, regular suit led
          // Can follow if either pip matches
          canFollow = (lo === suit || hi === suit);
        }

        if (canFollow) {
          mask |= (1 << d);
        }
      }
      result[abs][suit] = mask;
    }
  }

  return result;
})();

/**
 * RANK[d][powerId] -> number (higher wins)
 *
 * Determines the rank of a domino for trick-winning comparison.
 * - Power dominoes get +50 bonus (trump tier)
 * - Double of power pip gets +100 (highest trump)
 * - Non-power doubles get +20 bonus (highest in their suit)
 * - Base rank is pip sum
 *
 * The +20 double bonus ensures doubles beat non-doubles in the same suit
 * (e.g., 3-3 beats 6-3 when following threes).
 *
 * 28 × 9 = 252 entries
 */
export const RANK: readonly number[][] = (() => {
  const result: number[][] = [];

  for (let d = 0; d < 28; d++) {
    result[d] = [];
    const [lo, hi] = DOMINO_PIPS[d];
    const isDouble = lo === hi;
    const pipSum = lo + hi;

    for (let power = 0; power < 9; power++) {
      if (power <= 6) {
        // Pip power: dominoes containing that pip beat others
        const hasPower = (lo === power || hi === power);
        if (hasPower) {
          if (isDouble && lo === power) {
            result[d][power] = 100; // highest trump (e.g., 5-5 when 5s trump)
          } else {
            result[d][power] = 50 + pipSum;
          }
        } else {
          // Non-power: doubles get +20 bonus (highest in their suit)
          result[d][power] = isDouble ? pipSum + 20 : pipSum;
        }
      } else if (power === 7) {
        // Doubles power
        if (isDouble) {
          result[d][power] = 50 + pipSum; // all doubles are trump
        } else {
          result[d][power] = pipSum;
        }
      } else {
        // No power (8): doubles still highest in suit, non-doubles by pip sum
        result[d][power] = isDouble ? pipSum + 20 : pipSum;
      }
    }
  }

  return result;
})();

/**
 * HAS_POWER[d][powerId] -> boolean
 *
 * Determines if a domino has power (can beat non-power dominoes).
 * Used for trick winner eligibility.
 *
 * 28 × 9 = 252 entries
 */
export const HAS_POWER: readonly boolean[][] = (() => {
  const result: boolean[][] = [];

  for (let d = 0; d < 28; d++) {
    result[d] = [];
    const [lo, hi] = DOMINO_PIPS[d];
    const isDouble = lo === hi;

    for (let power = 0; power < 9; power++) {
      if (power === 8) {
        result[d][power] = false;
      } else if (power === 7) {
        result[d][power] = isDouble;
      } else {
        result[d][power] = (lo === power || hi === power);
      }
    }
  }

  return result;
})();

// ============= GAME LOGIC FUNCTIONS =============

/**
 * Get the suit that a domino leads
 */
export function getLedSuitFromTable(d: DominoId, absorptionId: AbsorptionId): LedSuit {
  return EFFECTIVE_SUIT[d][absorptionId] as LedSuit;
}

/**
 * Get legal plays from a hand given what was led
 *
 * @param hand - Bitmask of dominoes in hand (bit i = domino i present)
 * @param absorptionId - Current absorption configuration
 * @param leadDominoId - The domino that was led (null if leading)
 * @returns Bitmask of legal plays
 */
export function getLegalPlaysMask(
  hand: number,
  absorptionId: AbsorptionId,
  leadDominoId: DominoId | null
): number {
  if (leadDominoId === null) return hand; // leading: any domino

  const ledSuit = EFFECTIVE_SUIT[leadDominoId][absorptionId];
  const canFollow = hand & SUIT_MASK[absorptionId][ledSuit];
  return canFollow !== 0 ? canFollow : hand; // must follow if able
}

/**
 * Determine the winner of a trick
 *
 * @param trick - Array of domino IDs in play order
 * @param absorptionId - Current absorption configuration
 * @param powerId - Current power configuration
 * @param leadPlayer - Player index who led
 * @returns Player index who won the trick
 */
export function getTrickWinnerFromTable(
  trick: readonly DominoId[],
  absorptionId: AbsorptionId,
  powerId: PowerId,
  leadPlayer: number
): number {
  const ledSuit = EFFECTIVE_SUIT[trick[0]][absorptionId];

  let winner = 0;
  let maxRank = RANK[trick[0]][powerId];

  for (let i = 1; i < trick.length; i++) {
    const domino = trick[i];
    const dominoSuit = EFFECTIVE_SUIT[domino][absorptionId];

    // Only dominoes in led suit OR with power can win
    const inLedSuit = (dominoSuit === ledSuit);
    const hasPower = HAS_POWER[domino][powerId];

    if (!inLedSuit && !hasPower) {
      continue; // played off, can't win
    }

    const rank = RANK[domino][powerId];
    if (rank > maxRank) {
      maxRank = rank;
      winner = i;
    }
  }

  return (leadPlayer + winner) % 4;
}

/**
 * Check if a domino can follow the led suit
 */
export function canFollowFromTable(
  dominoId: DominoId,
  absorptionId: AbsorptionId,
  ledSuit: SuitId
): boolean {
  return (SUIT_MASK[absorptionId][ledSuit] & (1 << dominoId)) !== 0;
}

/**
 * Check if a domino has trump power
 */
export function isTrumpFromTable(dominoId: DominoId, powerId: PowerId): boolean {
  return HAS_POWER[dominoId][powerId];
}

/**
 * Get the rank of a domino for comparison
 */
export function getRankFromTable(dominoId: DominoId, powerId: PowerId): number {
  return RANK[dominoId][powerId];
}

/**
 * Convert a hand (array of Domino) to a bitmask
 */
export function handToBitmask(hand: readonly Domino[]): number {
  let mask = 0;
  for (const domino of hand) {
    mask |= (1 << dominoToId(domino));
  }
  return mask;
}

/**
 * Get all suits a domino can follow (for UI hints)
 * Returns array of suit indices this domino can legally follow
 */
export function getSuitsForDomino(
  dominoId: DominoId,
  absorptionId: AbsorptionId
): SuitId[] {
  const suits: SuitId[] = [];
  const dominoMask = 1 << dominoId;

  for (let suit = 0; suit < 8; suit++) {
    if (SUIT_MASK[absorptionId][suit] & dominoMask) {
      suits.push(suit);
    }
  }

  return suits;
}
