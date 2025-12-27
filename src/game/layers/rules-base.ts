/**
 * Base Rule Implementations (Crystal Palace)
 *
 * This is the SINGLE SOURCE OF TRUTH for base trump/suit/follow-suit semantics.
 * All other modules (base.ts, compose.ts, AI, etc.) must delegate here.
 *
 * Key insight: Trump conflates two independent operations:
 * 1. Absorption - restructures which dominoes belong to which suit
 * 2. Power - determines which dominoes beat others
 *
 * All absorbed dominoes lead suit 7 (the "called" suit). This makes the S₇ symmetry
 * explicit: all pip trumps are isomorphic, differing only in which pip triggers
 * absorption.
 *
 * Implementation: Delegates to precomputed lookup tables in domino-tables.ts
 * for O(1) performance and GPU-readiness. See bead t42-9xy3 for the theory.
 *
 * Special contracts (nello, sevens, plunge, splash) override these in their layers.
 */

import type { GameState, Domino, LedSuit } from '../types';
import {
  dominoToId,
  getAbsorptionId,
  getPowerId,
  getLedSuitFromTable,
  canFollowFromTable,
  isTrumpFromTable,
  getSuitsForDomino,
} from '../core/domino-tables';

/**
 * Base implementation: What suit does this domino lead?
 *
 * - Absorbed dominoes (trump) lead suit 7 (CALLED)
 * - Non-absorbed dominoes lead their higher pip
 */
export function getLedSuitBase(state: GameState, domino: Domino): LedSuit {
  const dominoId = dominoToId(domino);
  const absorptionId = getAbsorptionId(state.trump);
  return getLedSuitFromTable(dominoId, absorptionId);
}

/**
 * Base implementation: What suits does this domino belong to?
 *
 * - Absorbed dominoes belong only to suit 7 (CALLED)
 * - Non-absorbed doubles belong to their pip suit
 * - Non-absorbed non-doubles belong to both pip suits
 */
export function suitsWithTrumpBase(state: GameState, domino: Domino): LedSuit[] {
  const dominoId = dominoToId(domino);
  const absorptionId = getAbsorptionId(state.trump);
  return getSuitsForDomino(dominoId, absorptionId) as LedSuit[];
}

/**
 * Base implementation: Can this domino follow the led suit?
 *
 * - Absorbed suit (7) led: only absorbed dominoes can follow
 * - Regular suit led: non-absorbed dominoes with that pip can follow
 * - Absorbed dominoes cannot follow regular suits
 */
export function canFollowBase(state: GameState, led: LedSuit, domino: Domino): boolean {
  const dominoId = dominoToId(domino);
  const absorptionId = getAbsorptionId(state.trump);
  return canFollowFromTable(dominoId, absorptionId, led);
}

/**
 * Rank computation with pre-computed configuration IDs.
 *
 * Implements τ(d, ℓ, δ) = (tier << 4) + rank from SUIT_ALGEBRA.md §8.
 *
 * This is the optimized version used by calculateTrickWinner to avoid
 * recomputing absorptionId/powerId for each domino in a trick.
 *
 * Three-tier ranking with 6-bit encoding:
 * - Tier 2 (trump):   32-46  (binary 10_xxxx)
 * - Tier 1 (follows): 16-30  (binary 01_xxxx)
 * - Tier 0 (slough):  0      (binary 00_0000)
 */
export function rankInTrickWithConfig(
  absorptionId: number,
  powerId: number,
  led: LedSuit,
  domino: Domino
): number {
  const dominoId = dominoToId(domino);
  const isDouble = domino.high === domino.low;
  const pipSum = domino.high + domino.low;

  // Determine tier
  const hasPower = isTrumpFromTable(dominoId, powerId);
  const followsSuit = canFollowFromTable(dominoId, absorptionId, led);

  let tier: number;
  if (hasPower) {
    tier = 2;
  } else if (followsSuit) {
    tier = 1;
  } else {
    tier = 0;
  }

  // Tier 0 (slough): all return 0
  if (tier === 0) {
    return 0;
  }

  // Determine rank within tier
  // Per SUIT_ALGEBRA.md §8: when κ(δ) = D° (absorptionId = 7), doubles rank by pip value
  let rank: number;
  if (absorptionId === 7 && isDouble) {
    // Doubles form suit 7 (doubles-trump or doubles-suit/nello): rank by pip value (0-6)
    rank = domino.high;
  } else if (isDouble) {
    // Lone double in pip suit: rank = 14 (highest possible)
    rank = 14;
  } else {
    // Non-doubles: rank = pip sum (0-12)
    rank = pipSum;
  }

  // τ = (tier << 4) + rank
  return (tier << 4) + rank;
}

/**
 * Base implementation: What is this domino's rank for trick-taking?
 *
 * Delegates to rankInTrickWithConfig after computing configuration IDs.
 * For batch operations (like calculateTrickWinner), use rankInTrickWithConfig
 * directly to avoid recomputing IDs for each domino.
 */
export function rankInTrickBase(state: GameState, led: LedSuit, domino: Domino): number {
  const absorptionId = getAbsorptionId(state.trump);
  const powerId = getPowerId(state.trump);
  return rankInTrickWithConfig(absorptionId, powerId, led, domino);
}

/**
 * Base implementation: Is this domino trump (absorbed with power)?
 *
 * - Doubles-trump: all doubles have power
 * - Regular suit trump: dominoes containing the trump pip have power
 * - No-trump/not-selected: nothing has power
 */
export function isTrumpBase(state: GameState, domino: Domino): boolean {
  const dominoId = dominoToId(domino);
  const powerId = getPowerId(state.trump);
  return isTrumpFromTable(dominoId, powerId);
}

/**
 * Base implementation: Is this play valid?
 *
 * Uses canFollowBase for follow-suit validation.
 */
export function isValidPlayBase(
  state: GameState,
  domino: Domino,
  playerId: number
): boolean {
  if (state.phase !== 'playing' || state.trump.type === 'not-selected') return false;

  const player = state.players[playerId];
  if (!player || !player.hand.some(d => d.id === domino.id)) return false;

  // First play of trick is always legal
  if (state.currentTrick.length === 0) return true;

  const leadSuit = state.currentSuit;
  if (leadSuit === -1) return true;

  // Check if player must follow suit (has any domino that can follow)
  const mustFollow = player.hand.some(d => canFollowBase(state, leadSuit as LedSuit, d));
  if (!mustFollow) return true;

  // Must play a domino that follows suit
  return canFollowBase(state, leadSuit as LedSuit, domino);
}

/**
 * Base implementation: Get all valid plays for this player.
 *
 * Uses canFollowBase for follow-suit validation.
 */
export function getValidPlaysBase(
  state: GameState,
  playerId: number
): Domino[] {
  if (state.phase !== 'playing' || state.trump.type === 'not-selected') return [];

  const player = state.players[playerId];
  if (!player) return [];

  // First play of trick - all dominoes are valid
  if (state.currentTrick.length === 0) return [...player.hand];

  const leadSuit = state.currentSuit;
  if (leadSuit === -1) return [...player.hand];

  // Filter to dominoes that can follow suit
  const followers = player.hand.filter(d => canFollowBase(state, leadSuit as LedSuit, d));

  // If player has dominoes that can follow, must play one of them
  return followers.length > 0 ? followers : [...player.hand];
}
