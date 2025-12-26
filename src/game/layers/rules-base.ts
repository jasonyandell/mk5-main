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
 * Special contracts (nello, sevens, plunge, splash) override these in their layers.
 */

import type { GameState, Domino, LedSuit } from '../types';
import { CALLED } from '../types';
import { getTrumpSuit, isRegularSuitTrump, isDoublesTrump, dominoHasSuit } from '../core/dominoes';

/** Suit 7 is the called suit - where absorbed dominoes go */
const CALLED_SUIT = CALLED;

/**
 * Base implementation: What suit does this domino lead?
 *
 * - Absorbed dominoes (trump) lead suit 7
 * - Non-absorbed dominoes lead their higher pip
 */
export function getLedSuitBase(state: GameState, domino: Domino): LedSuit {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;

  // When doubles are trump, doubles are absorbed
  if (isDoublesTrump(trumpSuit)) {
    return isDouble ? CALLED_SUIT : domino.high as LedSuit;
  }

  // When a regular suit is trump (0-6), dominoes containing that pip are absorbed
  if (isRegularSuitTrump(trumpSuit)) {
    if (dominoHasSuit(domino, trumpSuit)) {
      return CALLED_SUIT;
    }
  }

  // Non-absorbed: higher pip
  return domino.high as LedSuit;
}

/**
 * Base implementation: What suits does this domino belong to?
 *
 * - Absorbed dominoes belong only to suit 7
 * - Non-absorbed doubles belong to their pip suit
 * - Non-absorbed non-doubles belong to both pip suits
 */
export function suitsWithTrumpBase(state: GameState, domino: Domino): LedSuit[] {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;

  // Doubles trump: doubles are absorbed
  if (isDoublesTrump(trumpSuit)) {
    if (isDouble) {
      return [CALLED_SUIT];
    }
    return [domino.high as LedSuit, domino.low as LedSuit];
  }

  // Regular suit trump (0-6): dominoes containing trump pip are absorbed
  if (isRegularSuitTrump(trumpSuit)) {
    if (dominoHasSuit(domino, trumpSuit)) {
      return [CALLED_SUIT];
    }
    if (isDouble) {
      return [domino.high as LedSuit];
    }
    return [domino.high as LedSuit, domino.low as LedSuit];
  }

  // No trump: natural suits
  if (isDouble) {
    return [domino.high as LedSuit];
  }
  return [domino.high as LedSuit, domino.low as LedSuit];
}

/**
 * Base implementation: Can this domino follow the led suit?
 *
 * - Absorbed suit (7) led: only absorbed dominoes can follow
 * - Regular suit led: non-absorbed dominoes with that pip can follow
 * - Absorbed dominoes cannot follow regular suits
 */
export function canFollowBase(state: GameState, led: LedSuit, domino: Domino): boolean {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;

  // Check if this domino is absorbed
  const isAbsorbed = isDoublesTrump(trumpSuit)
    ? isDouble
    : isRegularSuitTrump(trumpSuit) && dominoHasSuit(domino, trumpSuit);

  // Absorbed suit led (7): only absorbed dominoes can follow
  if (led === CALLED_SUIT) {
    return isAbsorbed;
  }

  // Regular suit led: absorbed dominoes cannot follow
  if (isAbsorbed) {
    return false;
  }

  // Non-absorbed domino, regular suit led: check pip membership
  return dominoHasSuit(domino, led);
}

/**
 * Base implementation: What is this domino's rank for trick-taking?
 *
 * Higher rank wins. Ranking tiers:
 * - 200+: Trump/absorbed with power (doubles get +50)
 * - 50+: Follows suit (doubles get +50)
 * - 0-12: Slough (just pip sum)
 */
export function rankInTrickBase(state: GameState, led: LedSuit, domino: Domino): number {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;
  const pipSum = domino.high + domino.low;

  // Check if domino is absorbed (has power)
  const isAbsorbed = isDoublesTrump(trumpSuit)
    ? isDouble
    : isRegularSuitTrump(trumpSuit) && dominoHasSuit(domino, trumpSuit);

  if (isAbsorbed) {
    return isDouble ? 200 + domino.high + 50 : 200 + pipSum;
  }

  // Check if domino follows suit
  const followsSuit = (led === CALLED_SUIT)
    ? false  // Non-absorbed can't follow absorbed suit
    : dominoHasSuit(domino, led);

  if (followsSuit) {
    return isDouble ? 50 + domino.high + 50 : 50 + pipSum;
  }

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
  const trumpSuit = getTrumpSuit(state.trump);

  if (isDoublesTrump(trumpSuit)) {
    return domino.high === domino.low;
  }

  if (isRegularSuitTrump(trumpSuit)) {
    return dominoHasSuit(domino, trumpSuit);
  }

  return false;
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
