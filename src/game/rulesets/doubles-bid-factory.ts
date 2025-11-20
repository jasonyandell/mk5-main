/**
 * Factory for creating doubles-based special bid rulesets (plunge/splash).
 *
 * These bids share identical logic except for thresholds:
 * - Plunge: 4+ doubles, 4+ marks, partner trumps/leads, must win all tricks
 * - Splash: 3+ doubles, 2-3 marks, partner trumps/leads, must win all tricks
 */

import type { GameRuleSet } from './types';
import {
  getPartner,
  getPlayerTeam,
  checkTrickBasedHandOutcome,
  getHighestMarksBid,
  countDoubles
} from './helpers';

interface DoublesBidConfig {
  name: 'plunge' | 'splash';
  minDoubles: number;
  minValue: number;
  maxValue?: number; // undefined means no upper limit
}

export function createDoublesBidRuleSet(config: DoublesBidConfig): GameRuleSet {
  const { name, minDoubles, minValue, maxValue } = config;

  return {
    name,

    getValidActions: (state, prev) => {
      // Add special bid during bidding phase when player has required doubles
      if (state.phase !== 'bidding') return prev;

      const player = state.players[state.currentPlayer];
      if (!player) return prev;

      const doubles = countDoubles(player.hand);
      if (doubles >= minDoubles) {
        const highestMarksBid = getHighestMarksBid(state.bids);
        const calculatedValue = Math.max(minValue, highestMarksBid + 1);
        const bidValue = maxValue !== undefined
          ? Math.min(maxValue, calculatedValue)
          : calculatedValue;

        return [...prev, {
          type: 'bid' as const,
          player: state.currentPlayer,
          bid: name,
          value: bidValue
        }];
      }

      return prev;
    },

    rules: {

      isValidBid: (state, bid, playerHand, prev) => {
        if (bid.type !== name) return prev;

        // Validate bid value
        if (bid.value === undefined || bid.value < minValue) return false;
        if (maxValue !== undefined && bid.value > maxValue) return false;

        // Validate doubles requirement
        if (!playerHand || countDoubles(playerHand) < minDoubles) return false;

        // Opening bid constraints
        const previousBids = state.bids.filter(b => b.type !== 'pass');
        if (previousBids.length === 0) {
          return true; // Allowed as opening bid
        }

        return true; // Allowed as subsequent bid (can jump)
      },

      getBidComparisonValue: (bid, prev) => {
        if (bid.type === name) {
          return bid.value! * 42;
        }
        return prev;
      },

      calculateScore: (state, prev) => {
        if (state.currentBid?.type !== name) return prev;

        // Bidding team must take all tricks
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
        bid.type === name
          ? getPartner(bid.player)
          : prev,

      // Partner leads (they selected trump, so getFirstLeader passes through)
      // getFirstLeader not needed - prev already correct since trumpSelector = partner

      // Hand ends if opponents win any trick
      checkHandOutcome: (state, prev) => {
        if (state.currentBid?.type !== name) return prev;
        const biddingTeam = getPlayerTeam(state, state.winningBidder);
        return checkTrickBasedHandOutcome(state, biddingTeam, true);
      }
    }
  };
}
