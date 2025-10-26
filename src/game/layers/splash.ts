/**
 * Splash Layer - Special 2-3 mark bid requiring 3+ doubles.
 *
 * From docs/rules.md §8.A:
 * - Requires 3+ doubles in hand
 * - Bid value: Automatic based on current high bid (2-3 marks, jumps over existing bids)
 * - Partner declares trump and leads
 * - Must win all 7 tricks (early termination if opponents win any trick)
 *
 * Nearly identical to plunge layer, just different thresholds:
 * - Doubles requirement: 3+ (vs plunge 4+)
 * - Bid value range: 2-3 marks (vs plunge 4+ marks)
 */

import type { GameLayer } from './types';
import {
  getPartner,
  getPlayerTeam,
  checkMustWinAllTricks,
  countDoubles,
  getHighestMarksBid
} from './helpers';

export const splashLayer: GameLayer = {
  name: 'splash',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    const player = state.players[state.currentPlayer];
    if (!player) return prev; // Guard against undefined
    const doubles = countDoubles(player.hand);

    if (doubles >= 3) {
      // Splash value = highest marks bid + 1, minimum 2, maximum 3
      const highestMarksBid = getHighestMarksBid(state.bids);
      const splashValue = Math.min(3, Math.max(2, highestMarksBid + 1));

      return [...prev, {
        type: 'bid' as const,
        player: state.currentPlayer,
        bid: 'splash' as const,
        value: splashValue  // Automatic, not user choice
      }];
    }

    return prev;
  },

  rules: {
    // Partner selects trump (not bidder)
    getTrumpSelector: (_state, bid, prev) =>
      bid.type === 'splash'
        ? getPartner(bid.player)
        : prev,

    // Partner leads (they selected trump, so already currentPlayer)
    // getFirstLeader passes through (prev already correct)

    // Hand ends if opponents win any trick
    checkHandOutcome: (state, prev) => {
      if (state.currentBid?.type !== 'splash') return prev;
      if (prev?.isDetermined) return prev;

      // Use shared helper: bidding team must win all tricks
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      const outcome = checkMustWinAllTricks(state, biddingTeam, true);

      return outcome || prev;
    }
  }
};
