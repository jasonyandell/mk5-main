/**
 * Nello Layer - Special contract where bidder must lose all tricks.
 *
 * From docs/rules.md §8.A:
 * - Must bid at least 1 mark
 * - Partner sits out with dominoes face-down (3-player tricks)
 * - No trump suit declared
 * - Doubles form own suit (standard variant)
 * - Objective: Bidder must lose every trick
 * - Bidder leads normally
 */

import type { GameLayer } from './types';
import type { LedSuit } from '../types';
import { getPartner, getPlayerTeam, checkMustWinAllTricks } from './helpers';
import { getNextPlayer as getNextPlayerCore } from '../core/players';

export const nelloLayer: GameLayer = {
  name: 'nello',

  getValidActions: (state, prev) => {
    // Add nello as trump option when marks bid won
    if (state.phase === 'trump_selection' &&
        state.currentBid?.type === 'marks') {
      return [...prev, {
        type: 'select-trump',
        player: state.winningBidder,
        trump: { type: 'nello' }
      }];
    }

    return prev;
  },

  rules: {
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
