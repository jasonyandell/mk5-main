/**
 * Nello Layer - Special contract where bidder must lose all tricks.
 *
 * From docs/rules.md §8.A:
 * - Must bid at least 1 mark (standard marks bid, NOT a separate bid type)
 * - Nello is selected as trump during trump_selection phase
 * - Partner sits out with dominoes face-down (3-player tricks)
 * - No trump suit declared
 * - Doubles form own suit (standard)
 * - Objective: Bidder must lose every trick
 * - Bidder leads normally
 */

import type { Layer } from './types';
import type { LedSuit, Domino } from '../types';
import { getPartner, getPlayerTeam, checkTrickBasedHandOutcome } from './helpers';
import { getNextPlayer as getNextPlayerCore } from '../core/players';
import { BID_TYPES } from '../constants';

export const nelloLayer: Layer = {
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
      const biddingTeam = getPlayerTeam(state, state.winningBidder);
      return checkTrickBasedHandOutcome(state, biddingTeam, false);
    },

    // Doubles form own suit (suit 7)
    getLedSuit: (state, domino, prev) => {
      if (state.trump?.type !== 'nello') return prev;

      return (domino.high === domino.low ? 7 : domino.high) as LedSuit;
    },

    // In nello, doubles belong ONLY to suit 7
    suitsWithTrump: (state, domino, prev) => {
      if (state.trump?.type !== 'nello') return prev;

      const isDouble = domino.high === domino.low;
      if (isDouble) {
        return [7 as LedSuit];
      }
      // Non-doubles belong to both their pips
      return [domino.high as LedSuit, domino.low as LedSuit];
    },

    // In nello, doubles can only follow suit 7
    canFollow: (state, led, domino, prev) => {
      if (state.trump?.type !== 'nello') return prev;

      const isDouble = domino.high === domino.low;

      // Doubles led: only doubles can follow
      if (led === 7) {
        return isDouble;
      }

      // Regular suit led: doubles CANNOT follow
      if (isDouble) {
        return false;
      }

      // Non-doubles: check if they have the led suit
      return domino.high === led || domino.low === led;
    },

    // In nello, getValidPlays must use nello's canFollow logic
    // (Base getValidPlaysBase uses canFollowBase which doesn't recognize nello absorption)
    getValidPlays: (state, playerId, prev) => {
      if (state.trump?.type !== 'nello') return prev;
      if (state.phase !== 'playing') return [];

      const player = state.players[playerId];
      if (!player) return [];

      // First play of trick - all dominoes are valid
      if (state.currentTrick.length === 0) return [...player.hand];

      // No suit led yet
      if (state.currentSuit === -1) return [...player.hand];

      const leadSuit = state.currentSuit as LedSuit;

      // Use nello's canFollow logic: doubles can only follow suit 7
      const nelloCanFollow = (led: LedSuit, domino: Domino): boolean => {
        const isDouble = domino.high === domino.low;
        if (led === 7) return isDouble;
        if (isDouble) return false;
        return domino.high === led || domino.low === led;
      };

      const followers = player.hand.filter(d => nelloCanFollow(leadSuit, d));
      return followers.length > 0 ? followers : [...player.hand];
    },

    // Nello delegates to base for ranking.
    // Per SUIT_ALGEBRA.md §8: κ(δ) = D° triggers pip-value ranking for doubles.
    // Base implementation checks absorptionId === 7, which covers nello.
    // No override needed - just pass through to prev (which is rankInTrickBase result).
    rankInTrick: (_state, _led, _domino, prev) => prev
  }
};
