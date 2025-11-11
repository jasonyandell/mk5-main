/**
 * Nello RuleSet - Special contract where bidder must lose all tricks.
 *
 * From docs/rules.md §8.A:
 * - Must bid at least 1 mark (standard marks bid, NOT a separate bid type)
 * - Nello is selected as trump during trump_selection phase
 * - Partner sits out with dominoes face-down (3-player tricks)
 * - No trump suit declared
 * - Doubles form own suit (standard variant)
 * - Objective: Bidder must lose every trick
 * - Bidder leads normally
 */

import type { GameRuleSet } from './types';
import type { LedSuit } from '../types';
import { getPartner, getPlayerTeam, checkMustWinAllTricks } from './helpers';
import { getNextPlayer as getNextPlayerCore } from '../core/players';
import { BID_TYPES } from '../constants';

export const nelloRuleSet: GameRuleSet = {
  name: 'nello',

  getValidActions: (state, prev) => {
    // Add nello as trump option when marks bid won
    if (state.phase === 'trump_selection' && state.currentBid?.type === 'marks') {
      return [
        ...prev,
        {
          type: 'select-trump',
          player: state.winningBidder,
          trump: { type: 'nello' }
        }
      ];
    }

    return prev;
  },

  rules: {

    isValidTrump: (trump, prev) => {
      if (trump.type === 'nello') return true;
      return prev;
    },

    calculateScore: (state, prev) => {
      // Nello can be bid as MARKS with nello trump
      if (state.currentBid.type !== BID_TYPES.MARKS || state.trump.type !== 'nello') {
        return prev;
      }

      // Nello scoring: bidding team must take no tricks
      const bidder = state.players[state.winningBidder];
      if (!bidder) return prev;

      const biddingTeam = bidder.teamId;
      const biddingTeamTricks = state.tricks.filter(trick => {
        if (trick.winner === undefined) return false;
        const winner = state.players[trick.winner];
        if (!winner) return false;
        return winner.teamId === biddingTeam;
      }).length;

      const newMarks: [number, number] = [state.teamMarks[0], state.teamMarks[1]];
      if (biddingTeamTricks === 0) {
        newMarks[biddingTeam] += state.currentBid.value!;
      } else {
        newMarks[biddingTeam === 0 ? 1 : 0] += state.currentBid.value!;
      }
      return newMarks;
    },

    // Bidder leads normally in nello (no override needed)
    // getFirstLeader passes through prev

    // Skip partner in turn order
    getNextPlayer: (state, current, prev) => {
      if (state.trump?.type !== 'nello') return prev;

      const partner = getPartner(state.winningBidder);
      let next = getNextPlayerCore(current);

      if (next === partner) {
        next = getNextPlayerCore(next);
      }

      return next;
    },

    // 3-player tricks (partner sits out)
    isTrickComplete: (state, prev) =>
      state.trump?.type === 'nello'
        ? state.currentTrick.length === 3
        : prev,

    // Hand ends if bidder wins any trick
    checkHandOutcome: (state, prev) => {
      if (state.trump?.type !== 'nello') return prev;
      if (prev?.isDetermined) return prev; // Already ended

      // Use shared helper: bidding team must not win any tricks
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      const outcome = checkMustWinAllTricks(state, biddingTeam, false);

      return outcome || prev;
    },

    // Doubles form own suit (suit 7)
    getLedSuit: (state, domino, prev) => {
      if (state.trump?.type !== 'nello') return prev;

      return (domino.high === domino.low ? 7 : domino.high) as LedSuit;
    },

    // Standard trick-taking but with no trump (already handled by getLedSuit)
    // calculateTrickWinner uses prev (base implementation works)
  }
};
