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
 * Base implementation: What is this domino's rank for trick-taking?
 *
 * Higher rank wins. Three-tier ranking:
 * - 200+: Trump/absorbed with power (doubles get +50)
 * - 50+: Follows suit (doubles get +50)
 * - 0-12: Slough (just pip sum)
 *
 * Note: The RANK table encodes power-only ranking. This function adds
 * the "follows suit" tier which depends on what was led.
 */
export function rankInTrickBase(state: GameState, led: LedSuit, domino: Domino): number {
  const dominoId = dominoToId(domino);
  const absorptionId = getAbsorptionId(state.trump);
  const powerId = getPowerId(state.trump);

  const isDouble = domino.high === domino.low;
  const pipSum = domino.high + domino.low;

  // Check if domino has power (is trump)
  const hasPower = isTrumpFromTable(dominoId, powerId);

  if (hasPower) {
    // Trump tier: 200+ (doubles get +50)
    return isDouble ? 200 + domino.high + 50 : 200 + pipSum;
  }

  // Check if domino follows the led suit
  const followsSuit = canFollowFromTable(dominoId, absorptionId, led);

  if (followsSuit) {
    // Following suit tier: 50+ (doubles get +50)
    return isDouble ? 50 + domino.high + 50 : 50 + pipSum;
  }

  // Slough tier: just pip sum
  return pipSum;
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
