import type { GameState, Bid } from '../types';
import type { GameRules } from '../layers/types';
import { BID_TYPES } from '../constants';

/**
 * Result of bidding completion analysis
 */
export interface BiddingCompletionResult {
  phase: 'trump_selection';
  winningBidder: number;
  currentPlayer: number;
  currentBid: Bid;
}

/**
 * Analyzes whether bidding is complete and determines the next game phase.
 * Returns state updates for phase transition or undefined if bidding continues.
 *
 * @param bids - Array of all bids placed so far
 * @param state - Current game state (for trump selector determination)
 * @param rules - Game rules instance
 * @returns State updates if bidding complete with a winner, undefined if bidding incomplete or all passed
 */
export function analyzeBiddingCompletion(
  bids: Bid[],
  state: GameState,
  rules: GameRules
): BiddingCompletionResult | undefined {
  // Check if bidding is complete
  if (bids.length !== 4) {
    return undefined;
  }

  const nonPassBids = bids.filter(b => b.type !== BID_TYPES.PASS);

  if (nonPassBids.length === 0) {
    // All pass case - handled by redeal action
    return undefined;
  }

  // Find winning bidder
  const winningBid = nonPassBids.reduce((highest, current) => {
    const highestValue = rules.getBidComparisonValue(highest);
    const currentValue = rules.getBidComparisonValue(current);
    return currentValue > highestValue ? current : highest;
  });

  return {
    phase: 'trump_selection',
    winningBidder: winningBid.player,
    currentPlayer: rules.getTrumpSelector(state, winningBid),
    currentBid: winningBid
  };
}
