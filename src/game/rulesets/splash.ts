/**
 * Splash RuleSet - Special 2-3 mark bid requiring 3+ doubles.
 *
 * From docs/rules.md §8.A:
 * - Requires 3+ doubles in hand
 * - Bid value: Automatic based on current high bid (2-3 marks, jumps over existing bids)
 * - Partner declares trump and leads
 * - Must win all 7 tricks (early termination if opponents win any trick)
 *
 * Nearly identical to plunge rule set, just different thresholds:
 * - Doubles requirement: 3+ (vs plunge 4+)
 * - Bid value range: 2-3 marks (vs plunge 4+ marks)
 */

import type { GameRuleSet } from './types';
import {
  getPartner,
  getPlayerTeam,
  checkMustWinAllTricks,
  countDoubles,
  getHighestMarksBid
} from './helpers';
import { BID_TYPES } from '../constants';

export const splashRuleSet: GameRuleSet = {
  name: 'splash',

  getValidActions: (state, prev) => {
    if (state.phase !== 'bidding') return prev;

    const player = state.players[state.currentPlayer];
    if (!player) return prev; // Guard against undefined
    const doubles = countDoubles(player.hand);

    // Filter out all base splash bids - we'll replace with our calculated one
    const filtered = prev.filter(action =>
      !(action.type === 'bid' && action.bid === 'splash')
    );

    if (doubles >= 3) {
      // Splash value = highest marks bid + 1, minimum 2, maximum 3
      const highestMarksBid = getHighestMarksBid(state.bids);
      const splashValue = Math.min(3, Math.max(2, highestMarksBid + 1));

      return [...filtered, {
        type: 'bid' as const,
        player: state.currentPlayer,
        bid: 'splash' as const,
        value: splashValue  // Automatic, not user choice
      }];
    }

    // If player doesn't have enough doubles, remove all splash options
    return filtered;
  },

  rules: {
    isValidBid: (state, bid, playerHand, prev) => {
      if (bid.type !== BID_TYPES.SPLASH) return prev;

      // Splash validation
      if (bid.value === undefined || bid.value < 2 || bid.value > 3) return false;
      if (!playerHand || countDoubles(playerHand) < 3) return false;

      // Opening bid constraints
      const previousBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);
      if (previousBids.length === 0) {
        return true; // Splash allowed as opening bid
      }

      return true; // Splash allowed as subsequent bid (can jump)
    },

    getBidComparisonValue: (bid, prev) => {
      if (bid.type !== BID_TYPES.SPLASH) return prev;
      return bid.value! * 42;
    },

    calculateScore: (state, prev) => {
      if (state.currentBid?.type !== BID_TYPES.SPLASH) return prev;

      // Splash: bidding team must take all tricks
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
