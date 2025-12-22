/**
 * Base Rule Implementations (Crystal Palace)
 *
 * This is the SINGLE SOURCE OF TRUTH for base trump/suit/follow-suit semantics.
 * All other modules (base.ts, compose.ts, AI, etc.) must delegate here.
 *
 * These pure functions implement standard Texas 42 rules:
 * - Higher pip leads (or 7 for doubles-trump)
 * - Trump dominoes belong only to trump suit
 * - Follow-suit validation
 * - Rank calculation for trick-taking
 *
 * Special contracts (nello, sevens, plunge, splash) override these in their layers.
 */

import type { GameState, Domino, LedSuit } from '../types';
import { DOUBLES_AS_TRUMP } from '../types';
import { getTrumpSuit, isRegularSuitTrump, isDoublesTrump, dominoHasSuit } from '../core/dominoes';

/**
 * Base implementation: What suit does this domino lead?
 *
 * - Doubles-trump: doubles lead suit 7, non-doubles lead higher pip
 * - Regular trump: trump dominoes lead trump suit, others lead higher pip
 * - No-trump: higher pip (or the pip value for doubles)
 */
export function getLedSuitBase(state: GameState, domino: Domino): LedSuit {
  const trumpSuit = getTrumpSuit(state.trump);

  // When doubles are trump, doubles lead suit 7
  if (isDoublesTrump(trumpSuit)) {
    return domino.high === domino.low ? DOUBLES_AS_TRUMP : domino.high as LedSuit;
  }

  // When a regular suit is trump (0-6)
  if (isRegularSuitTrump(trumpSuit)) {
    // Trump dominoes lead trump suit
    if (dominoHasSuit(domino, trumpSuit)) {
      return trumpSuit as LedSuit;
    }
  }

  // Non-trump or no-trump: higher pip
  return domino.high as LedSuit;
}

/**
 * Base implementation: What suits does this domino belong to?
 *
 * A domino's suit membership depends on trump selection:
 * - Trump dominoes belong only to trump suit (absorbed)
 * - Doubles with doubles-trump belong only to suit 7
 * - Otherwise, dominoes belong to their natural suits
 */
export function suitsWithTrumpBase(state: GameState, domino: Domino): LedSuit[] {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;

  // Doubles trump: doubles belong only to suit 7
  if (isDoublesTrump(trumpSuit)) {
    if (isDouble) {
      return [7 as LedSuit];
    }
    // Non-doubles belong to both their pips
    return [domino.high as LedSuit, domino.low as LedSuit];
  }

  // Regular suit trump (0-6)
  if (isRegularSuitTrump(trumpSuit)) {
    // Trump dominoes belong only to trump suit
    if (dominoHasSuit(domino, trumpSuit)) {
      return [trumpSuit as LedSuit];
    }
    // Non-trump doubles belong to their pip
    if (isDouble) {
      return [domino.high as LedSuit];
    }
    // Non-trump non-doubles belong to both pips
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
 * A domino can follow if it belongs to the led suit.
 * Trump absorption rules apply.
 */
export function canFollowBase(state: GameState, led: LedSuit, domino: Domino): boolean {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;

  // Doubles led (suit 7): only doubles can follow
  if (led === 7) {
    return isDouble;
  }

  // Doubles trump: doubles belong only to suit 7, can't follow regular suits
  if (isDoublesTrump(trumpSuit) && isDouble) {
    return false;
  }

  // Regular suit trump: trump dominoes can't follow non-trump suits
  if (isRegularSuitTrump(trumpSuit) && dominoHasSuit(domino, trumpSuit)) {
    // Can only follow if led suit IS the trump suit
    return led === trumpSuit;
  }

  // Check if domino has the led suit
  return dominoHasSuit(domino, led);
}

/**
 * Base implementation: What is this domino's rank for trick-taking?
 *
 * Higher rank wins. Ranking tiers:
 * - 200+: Trump (with pip bonus, doubles get +50)
 * - 50+: Follows suit (with pip bonus, doubles get +50)
 * - 0-12: Slough (just pip sum)
 */
export function rankInTrickBase(state: GameState, led: LedSuit, domino: Domino): number {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;
  const pipSum = domino.high + domino.low;

  // Check if domino is trump
  const dominoIsTrump = isDoublesTrump(trumpSuit)
    ? isDouble
    : isRegularSuitTrump(trumpSuit) && dominoHasSuit(domino, trumpSuit);

  if (dominoIsTrump) {
    return isDouble ? 200 + domino.high + 50 : 200 + pipSum;
  }

  // Check if domino follows suit
  let followsSuit = false;
  if (led === 7) {
    followsSuit = isDouble;
  } else if (!(isDoublesTrump(trumpSuit) && isDouble)) {
    followsSuit = dominoHasSuit(domino, led);
  }

  if (followsSuit) {
    return isDouble ? 50 + domino.high + 50 : 50 + pipSum;
  }

  return pipSum;
}

/**
 * Base implementation: Is this domino trump?
 *
 * - Doubles-trump: all doubles are trump
 * - Regular suit trump: dominoes containing the trump suit are trump
 * - No-trump/not-selected: nothing is trump
 */
export function isTrumpBase(state: GameState, domino: Domino): boolean {
  const trumpSuit = getTrumpSuit(state.trump);

  if (isDoublesTrump(trumpSuit)) {
    return domino.high === domino.low;  // All doubles are trump
  }

  if (isRegularSuitTrump(trumpSuit)) {
    return dominoHasSuit(domino, trumpSuit);
  }

  return false;  // No trump or no-trump game
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
