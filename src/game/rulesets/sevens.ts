/**
 * Sevens RuleSet
 *
 * Implements sevens trump type for marks bids.
 *
 * Rules:
 * - Only available when marks bid won (not a bid type, a trump selection)
 * - Domino closest to 7 total pips wins trick
 * - Ties won by first played
 * - Must play domino closest to 7 total pips (replaces follow-suit)
 * - Must win all tricks (early termination)
 * - Bidder leads first trick
 * - Winner leads next trick
 */

import type { GameRuleSet } from './types';
import { checkMustWinAllTricks, getPlayerTeam } from './helpers';
import { BID_TYPES } from '../constants';

export const sevensRuleSet: GameRuleSet = {
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
    isValidTrump: (trump, prev) => {
      if (trump.type === 'sevens') return true;
      return prev;
    },

    getFirstLeader: (_state, trumpSelector, trump, prev) => {
      if (trump.type !== 'sevens') return prev;
      // Sevens: bidder leads first trick (standard behavior)
      return trumpSelector;
    },

    // Winner of trick leads next (not just team leader)
    getNextPlayer: (state, currentPlayer, prev) => {
      if (state.trump?.type !== 'sevens') return prev;

      // If trick just completed, winner leads next trick
      if (state.currentTrick.length === 0 && state.tricks.length > 0) {
        const lastTrick = state.tricks[state.tricks.length - 1];
        if (lastTrick && lastTrick.winner !== undefined) {
          return lastTrick.winner; // Winner leads
        }
      }

      // During trick, normal rotation
      return (currentPlayer + 1) % 4;
    },

    calculateScore: (state, prev) => {
      // Sevens can be bid as MARKS with sevens trump
      if (state.currentBid.type !== BID_TYPES.MARKS || state.trump.type !== 'sevens') {
        return prev;
      }

      // Sevens: bidding team must take all tricks
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
    },

    // ============================================
    // VALIDATION RULES: Must play closest to 7
    // ============================================

    isValidPlay: (state, domino, playerId, prev) => {
      // If not sevens, use previous RuleSet's logic
      if (state.trump?.type !== 'sevens') return prev;

      // Basic validation
      if (state.phase !== 'playing') return false;

      const player = state.players[playerId];
      if (!player || !player.hand.some((d) => d.id === domino.id)) return false;

      // Sevens: must play domino closest to 7 total pips
      const distances = player.hand.map(d => Math.abs(7 - (d.high + d.low)));
      const minDistance = Math.min(...distances);
      const dominoDistance = Math.abs(7 - (domino.high + domino.low));

      return dominoDistance === minDistance;
    },

    getValidPlays: (state, playerId, prev) => {
      // If not sevens, use previous RuleSet's logic
      if (state.trump?.type !== 'sevens') return prev;

      const player = state.players[playerId];
      if (!player || player.hand.length === 0) return [];

      // Sevens: must play domino(es) closest to 7 total pips
      const distances = player.hand.map(d => Math.abs(7 - (d.high + d.low)));
      const minDistance = Math.min(...distances);

      // Return all dominoes at minimum distance (player chooses if multiple)
      return player.hand.filter(d =>
        Math.abs(7 - (d.high + d.low)) === minDistance
      );
    }
  }
};
