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
import { BID_TYPES } from '../constants';

export const plungeLayer: GameLayer = {
  name: 'plunge',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    const player = state.players[state.currentPlayer];
    if (!player) return prev; // Guard against undefined
    const doubles = countDoubles(player.hand);

    // Filter out all base plunge bids - we'll replace with our calculated one
    const filtered = prev.filter(action =>
      !(action.type === 'bid' && action.bid === 'plunge')
    );

    if (doubles >= 4) {
      // Plunge value = highest marks bid + 1, minimum 4
      const highestMarksBid = getHighestMarksBid(state.bids);
      const plungeValue = Math.max(4, highestMarksBid + 1);

      return [...filtered, {
        type: 'bid' as const,
        player: state.currentPlayer,
        bid: 'plunge' as const,
        value: plungeValue  // Automatic, not user choice
      }];
    }

    // If player doesn't have enough doubles, remove all plunge options
    return filtered;
  },

  rules: {
    isValidBid: (state, bid, playerHand, prev) => {
      if (bid.type !== BID_TYPES.PLUNGE) return prev;

      // Plunge validation
      if (bid.value === undefined || bid.value < 4) return false;
      if (!playerHand || countDoubles(playerHand) < 4) return false;

      // Opening bid constraints
      const previousBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);
      if (previousBids.length === 0) {
        return true; // Plunge allowed as opening bid
      }

      return true; // Plunge allowed as subsequent bid (can jump)
    },

    getBidComparisonValue: (bid, prev) => {
      if (bid.type === BID_TYPES.PLUNGE) {
        return bid.value! * 42;
      }
      return prev;
    },

    calculateScore: (state, prev) => {
      if (state.currentBid?.type !== BID_TYPES.PLUNGE) return prev;

      // Plunge: bidding team must take all tricks
      const bidder = state.players[state.winningBidder];
      if (!bidder) return prev;
      const biddingTeam = bidder.teamId;
      const opponentTeam = biddingTeam === 0 ? 1 : 0;

      const nonBiddingTeamTricks = state.tricks.filter(trick => {
        if (trick.winner === undefined) return false;
        const winner = state.players[trick.winner];
        if (!winner) return false;
        return winner.teamId === opponentTeam;
      }).length;

      const newMarks: [number, number] = [state.teamMarks[0], state.teamMarks[1]];
      if (nonBiddingTeamTricks === 0) {
        newMarks[biddingTeam] += state.currentBid.value!;
      } else {
        newMarks[opponentTeam] += state.currentBid.value!;
      }
      return newMarks;
    },

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
