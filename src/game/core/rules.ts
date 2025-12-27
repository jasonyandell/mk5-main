import type { GameState, Bid, Domino, TrumpSelection, PlayedDomino, LedSuit, LedSuitOrNone, Play } from '../types';
import { BID_TYPES } from '../constants';
import { calculateTrickPoints } from './scoring';
import { getSuitName } from '../game-terms';
import { composeRules } from '../layers/compose';
import { baseLayer } from '../layers';
import { createMinimalState } from './minimal-state';

/**
 * Validates mark bids with tournament progression rules
 */
export function isValidMarkBid(bid: Bid, lastBid: Bid, _previousBids: Bid[]): boolean {
  if (bid.value === undefined) return false;

  // After point bids, can bid 1 or 2 marks
  if (lastBid.type === BID_TYPES.POINTS) {
    return bid.value >= 1 && bid.value <= 2;
  }

  // Mark bid progression: can only bid one more mark than the last mark bid
  if (lastBid.type === BID_TYPES.MARKS) {
    // Can always bid 2 marks (standard max opening bid rule)
    if (bid.value === 2) return true;

    // For 3+ marks, can only bid one more than the last mark bid if last bid was 2+
    if (bid.value >= 3 && lastBid.value! >= 2 && bid.value === lastBid.value! + 1) return true;

    return false;
  }

  return bid.value >= 1;
}

/**
 * Gets the winner of a trick using the threaded rules system.
 *
 * This is a convenience function that creates a minimal state for the rules.
 * For proper layer support, use rules.calculateTrickWinner(state, trick) directly.
 */
export function getTrickWinner(trick: { player: number; domino: Domino }[], trump: TrumpSelection, leadSuit: LedSuitOrNone): number {
  const rules = composeRules([baseLayer]);
  const state = createMinimalState({
    trump,
    currentSuit: leadSuit,
    currentTrick: trick as Play[]
  });
  return rules.calculateTrickWinner(state, trick as Play[]);
}

/**
 * Gets the points in a trick (alias for calculateTrickPoints)
 */
export function getTrickPoints(trick: { player: number; domino: Domino }[]): number {
  return calculateTrickPoints(trick as Play[]);
}

/**
 * Determines the winner of a trick using the threaded rules system.
 *
 * This is a convenience function that creates a minimal state for the rules.
 * For proper layer support, use rules.calculateTrickWinner(state, trick) directly.
 */
export function determineTrickWinner(trick: { player: number; domino: Domino }[] | PlayedDomino[], trump: TrumpSelection, leadSuit: LedSuitOrNone): number {
  return getTrickWinner(trick as { player: number; domino: Domino }[], trump, leadSuit);
}

/**
 * Gets the numeric value of a trump suit
 */
export function getTrumpValue(trump: TrumpSelection): number {
  switch (trump.type) {
    case 'not-selected': return -1;
    case 'suit': return trump.suit!;
    case 'doubles': return 7;
    case 'no-trump': return 8;
    default: return -1;
  }
}

/**
 * Gets the current suit being led in a trick for display purposes
 */
export function getCurrentSuit(state: GameState): string {
  if (state.currentSuit === -1) {
    return 'None (no domino led)';
  }

  if (state.trump.type === 'not-selected') {
    return 'None (no trump set)';
  }

  const leadSuit = state.currentSuit;

  // Handle special cases
  if (leadSuit === 7) {
    return 'Doubles (Trump)';
  }

  // Use centralized display logic with trump context
  if (leadSuit >= 0 && leadSuit <= 6) {
    // Check if this suit is trump
    const isTrump = state.trump.type === 'suit' && state.trump.suit === leadSuit;
    const suitName = getSuitName(leadSuit as LedSuit, { numeric: true });
    return isTrump ? `${suitName} (Trump)` : suitName;
  }

  return `Unknown (${leadSuit})`;
}

