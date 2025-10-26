/**
 * Plunge layer - Special bid requiring 4+ doubles.
 *
 * From docs/rules.md §8.A:
 * - Requires 4+ doubles in hand
 * - Bid value: Automatic based on current high bid (4+ marks, jumps over existing bids)
 * - Partner declares trump and leads
 * - Must win all 7 tricks (early termination if opponents win any trick)
 */

import type { GameLayer } from './types';
import {
  getPartner,
  getPlayerTeam,
  checkMustWinAllTricks,
  getHighestMarksBid,
  countDoubles
} from './helpers';

export const plungeLayer: GameLayer = {
  name: 'plunge',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    const player = state.players[state.currentPlayer];
    if (!player) return prev; // Guard against undefined
    const doubles = countDoubles(player.hand);

    if (doubles >= 4) {
      // Plunge value = highest marks bid + 1, minimum 4
      const highestMarksBid = getHighestMarksBid(state.bids);
      const plungeValue = Math.max(4, highestMarksBid + 1);

      return [...prev, {
        type: 'bid' as const,
        player: state.currentPlayer,
        bid: 'plunge' as const,
        value: plungeValue  // Automatic, not user choice
      }];
    }

    return prev;
  },

  rules: {
    // Partner selects trump (not bidder)
    getTrumpSelector: (_state, bid, prev) =>
      bid.type === 'plunge'
        ? getPartner(bid.player)
        : prev,

    // Partner leads (they selected trump, so getFirstLeader passes through)
    // getFirstLeader not needed - prev already correct since trumpSelector = partner

    // Hand ends if opponents win any trick
    checkHandOutcome: (state, prev) => {
      if (state.currentBid?.type !== 'plunge') return prev;
      if (prev?.isDetermined) return prev;

      // Use shared helper: bidding team must win all tricks
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      const outcome = checkMustWinAllTricks(state, biddingTeam, true);

      return outcome || prev;
    }
  }
};
