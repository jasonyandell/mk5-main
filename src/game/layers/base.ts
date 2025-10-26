/**
 * Base layer for standard Texas 42 rules.
 *
 * This layer implements the foundational game mechanics:
 * - Bidder selects trump
 * - Trump selector leads first trick
 * - Clockwise turn order
 * - 4 plays per trick
 * - All 7 tricks must be played
 * - Standard trump/suit hierarchy for trick winners
 * - Higher pip leads (or 7 for doubles-trump)
 *
 * Base layer knows nothing about special contracts (nello, plunge, splash, sevens).
 */

import type { GameState, Bid, TrumpSelection, Domino, Play, LedSuit } from '../types';
import type { GameLayer } from './types';
import { getDominoValue, getTrumpSuit } from '../core/dominoes';
import { getNextPlayer as getNextPlayerCore } from '../core/players';
import { GAME_CONSTANTS } from '../constants';
import { DOUBLES_AS_TRUMP } from '../types';

/**
 * Checks if a domino follows the led suit (contains the led suit number)
 */
function dominoFollowsSuit(domino: { high: number; low: number }, ledSuit: LedSuit): boolean {
  return domino.high === ledSuit || domino.low === ledSuit;
}

/**
 * Checks if a domino is trump based on numeric trump value
 */
function isDominoTrump(domino: { high: number; low: number }, numericTrump: number | null): boolean {
  if (numericTrump === null) return false;

  // Special case: doubles trump (numericTrump === 7)
  if (numericTrump === 7) {
    return domino.high === domino.low;
  }

  // Regular trump (contains trump suit number)
  return domino.high === numericTrump || domino.low === numericTrump;
}

/**
 * Converts TrumpSelection to numeric value for trick-taking logic
 */
function trumpToNumeric(trump: TrumpSelection): number | null {
  switch (trump.type) {
    case 'not-selected': return null;
    case 'suit': return trump.suit!;
    case 'doubles': return 7;
    case 'no-trump': return 8;
    case 'nello': return null;  // Nello has no trump
    case 'sevens': return null; // Sevens has no trump hierarchy
  }
}

export const baseLayer: GameLayer = {
  name: 'base',

  rules: {
    /**
     * WHO selects trump after bidding completes?
     *
     * Base: Winning bidder selects trump
     */
    getTrumpSelector(_state: GameState, winningBid: Bid): number {
      return winningBid.player;
    },

    /**
     * WHO leads the first trick after trump is selected?
     *
     * Base: Trump selector (the bidder) leads first trick
     */
    getFirstLeader(_state: GameState, trumpSelector: number, _trump: TrumpSelection): number {
      return trumpSelector;
    },

    /**
     * WHO plays next after current player?
     *
     * Base: Clockwise rotation (0 -> 1 -> 2 -> 3 -> 0)
     */
    getNextPlayer(_state: GameState, currentPlayer: number): number {
      return getNextPlayerCore(currentPlayer);
    },

    /**
     * WHEN is the current trick complete?
     *
     * Base: After 4 plays (one from each player)
     */
    isTrickComplete(state: GameState): boolean {
      return state.currentTrick.length === GAME_CONSTANTS.PLAYERS;
    },

    /**
     * WHEN should the hand end early (before all 7 tricks)?
     *
     * Base: Never - standard 42 plays all 7 tricks
     * Returns null unless all tricks are complete
     */
    checkHandOutcome(state: GameState) {
      // Only end after all 7 tricks in standard play
      if (state.tricks.length < GAME_CONSTANTS.TRICKS_PER_HAND) {
        return null;
      }

      return {
        isDetermined: true,
        reason: 'All tricks played'
      };
    },

    /**
     * HOW does a domino determine what suit it leads?
     *
     * Base: Higher pip leads, or 7 if doubles-trump
     * - Doubles-trump: doubles lead suit 7, non-doubles lead higher pip
     * - Regular trump: trump dominoes lead trump suit, others lead higher pip
     * - No-trump: higher pip (or the pip value for doubles)
     */
    getLedSuit(state: GameState, domino: Domino): LedSuit {
      const trumpSuit = getTrumpSuit(state.trump);

      // When doubles are trump, doubles lead suit 7
      if (trumpSuit === DOUBLES_AS_TRUMP) {
        return domino.high === domino.low ? DOUBLES_AS_TRUMP : domino.high as LedSuit;
      }

      // When a regular suit is trump (0-6)
      if (trumpSuit >= 0 && trumpSuit <= 6) {
        // Trump dominoes lead trump suit
        if (domino.high === trumpSuit || domino.low === trumpSuit) {
          return trumpSuit as LedSuit;
        }
      }

      // Non-trump or no-trump: higher pip
      return domino.high as LedSuit;
    },

    /**
     * HOW is the winner of a trick determined?
     *
     * Base: Standard trick-taking hierarchy
     * 1. Trump beats non-trump
     * 2. Higher trump wins
     * 3. Following suit beats non-following
     * 4. Higher value wins among followers
     */
    calculateTrickWinner(state: GameState, trick: Play[]): number {
      if (trick.length === 0) {
        throw new Error('Trick cannot be empty');
      }

      const leadPlay = trick[0];
      if (!leadPlay) {
        throw new Error('Cannot determine winner of empty trick');
      }

      const numericTrump = trumpToNumeric(state.trump);
      const ledSuit = state.currentSuit;

      let winningPlay = leadPlay;
      let winningValue = getDominoValue(leadPlay.domino, state.trump);
      let winningIsTrump = isDominoTrump(leadPlay.domino, numericTrump);

      for (let i = 1; i < trick.length; i++) {
        const play = trick[i];
        if (!play) {
          throw new Error(`Invalid trick play at index ${i}`);
        }

        const playValue = getDominoValue(play.domino, state.trump);
        const playIsTrump = isDominoTrump(play.domino, numericTrump);

        // Trump always beats non-trump
        if (playIsTrump && !winningIsTrump) {
          winningPlay = play;
          winningValue = playValue;
          winningIsTrump = true;
        }
        // Both trump - higher value wins
        else if (playIsTrump && winningIsTrump && playValue > winningValue) {
          winningPlay = play;
          winningValue = playValue;
        }
        // Both non-trump - must follow suit and higher value wins
        else if (!playIsTrump && !winningIsTrump &&
                 ledSuit >= 0 && dominoFollowsSuit(play.domino, ledSuit as LedSuit) &&
                 playValue > winningValue) {
          winningPlay = play;
          winningValue = playValue;
        }
      }

      return winningPlay.player;
    }
  }
};
