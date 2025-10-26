/**
 * Sevens Layer
 *
 * Implements sevens trump type for marks bids.
 *
 * Rules:
 * - Only available when marks bid won (not a bid type, a trump selection)
 * - Domino closest to 7 total pips wins trick
 * - Ties won by first played
 * - No follow-suit requirement (already handled by base rules)
 * - Must win all tricks (early termination)
 * - Bidder leads normally
 */

import type { GameLayer } from './types';
import { checkMustWinAllTricks, getPlayerTeam } from './helpers';

export const sevensLayer: GameLayer = {
  name: 'sevens',

  getValidActions: (state, prev) => {
    // Add sevens as trump option when marks bid won
    if (state.phase === 'trump_selection' && state.currentBid?.type === 'marks') {
      return [
        ...prev,
        {
          type: 'select-trump',
          player: state.winningBidder,
          trump: { type: 'sevens' }
        }
      ];
    }

    return prev;
  },

  rules: {
    // Completely different trick winner algorithm
    calculateTrickWinner: (state, trick, prev) => {
      if (state.trump?.type !== 'sevens') return prev;

      // Distance from 7 total pips
      const distances = trick.map(play =>
        Math.abs(7 - (play.domino.high + play.domino.low))
      );

      const minDistance = Math.min(...distances);

      // First domino with minimum distance wins (tie = first)
      const winnerIndex = trick.findIndex(play =>
        Math.abs(7 - (play.domino.high + play.domino.low)) === minDistance
      );

      return trick[winnerIndex]?.player ?? prev;
    },

    // Early termination on lost trick
    checkHandOutcome: (state, prev) => {
      if (state.trump?.type !== 'sevens') return prev;
      if (prev?.isDetermined) return prev;

      // Use shared helper: bidding team must win all tricks
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      const outcome = checkMustWinAllTricks(state, biddingTeam, true);

      return outcome || prev;
    }
  }
};
