/**
 * Base rule set for standard Texas 42 rules.
 *
 * This rule set implements the foundational game mechanics:
 * - Bidder selects trump
 * - Trump selector leads first trick
 * - Clockwise turn order
 * - 4 plays per trick
 * - All 7 tricks must be played
 * - Standard trump/suit hierarchy for trick winners
 * - Higher pip leads (or 7 for doubles-trump)
 *
 * Base rule set knows nothing about special contracts (nello, plunge, splash, sevens).
 */

import type { GameState, Bid, TrumpSelection, Domino, Play, LedSuit, GameAction } from '../types';
import type { GameRuleSet, GameRules } from './types';
import { getDominoValue, getTrumpSuit } from '../core/dominoes';
import { getNextPlayer as getNextPlayerCore } from '../core/players';
import { GAME_CONSTANTS, BID_TYPES, TRUMP_SELECTIONS } from '../constants';
import { DOUBLES_AS_TRUMP } from '../types';
import { isValidMarkBid } from '../core/rules';
import { calculateRoundScore } from '../core/scoring';
import { composeRules } from './compose';
import { checkHandOutcome as checkStandardHandOutcome } from '../core/handOutcome';

/**
 * Generates structural actions only (pass, redeal, consensus, trump selection, plays).
 * Does NOT generate bids - those are added by RuleSets via getValidActions.
 *
 * @param state Current game state
 * @param rules Composed rules for validation (optional, used for getValidPlays)
 * @returns Structural actions only
 */
export function generateStructuralActions(
  state: GameState,
  rules?: GameRules
): GameAction[] {
  switch (state.phase) {
    case 'bidding':
      return getBiddingActions(state);
    case 'trump_selection':
      return getTrumpSelectionActions(state);
    case 'playing':
      return getPlayingActions(state, rules);
    case 'scoring':
      return getScoringActions(state);
    default:
      return [];
  }
}

/**
 * Gets valid bidding actions (structural actions only - bids added by RuleSets)
 */
function getBiddingActions(state: GameState): GameAction[] {
  const actions: GameAction[] = [];

  // Check if bidding is complete
  if (state.bids.length === 4) {
    const nonPassBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);
    if (nonPassBids.length === 0) {
      // All passed - need redeal
      actions.push({
        type: 'redeal',
        autoExecute: true,
        meta: { authority: 'system' }
      });
      return actions;
    }
    // Otherwise, bidding is complete - no more actions
    return actions;
  }

  // Check if current player has already bid
  if (state.bids.some(b => b.player === state.currentPlayer)) {
    return actions;
  }

  // Pass action (only structural action - bids added by RuleSets via getValidActions)
  actions.push({ type: 'pass', player: state.currentPlayer });

  return actions;
}

/**
 * Gets valid trump selection actions
 */
function getTrumpSelectionActions(state: GameState): GameAction[] {
  const actions: GameAction[] = [];

  if (state.winningBidder === -1) return actions;

  // Generate trump selection actions
  // Use currentPlayer (which may be partner for plunge/splash) not winningBidder
  Object.values(TRUMP_SELECTIONS).forEach(trumpSelection => {
    actions.push({
      type: 'select-trump',
      player: state.currentPlayer,
      trump: trumpSelection as TrumpSelection
    });
  });

  return actions;
}

/**
 * Gets valid playing actions
 */
function getPlayingActions(state: GameState, rules?: GameRules): GameAction[] {
  const actions: GameAction[] = [];

  if (state.trump.type === 'not-selected') return actions;

  // Check if trick is complete (use rules if provided, otherwise default to 4)
  const isTrickComplete = rules ? rules.isTrickComplete(state) : state.currentTrick.length === 4;

  // If trick is complete, add consensus actions
  if (isTrickComplete) {
    // All players who haven't agreed yet can agree (not just current player)
    // This is important for nello where the partner sits out but still needs to agree
    for (let playerId = 0; playerId < state.players.length; playerId++) {
      if (!state.consensus.completeTrick.has(playerId)) {
        actions.push({ type: 'agree-complete-trick', player: playerId });
      }
    }

    // If all have agreed, the trick can be completed
    if (state.consensus.completeTrick.size === state.players.length) {
      actions.push({
        type: 'complete-trick',
        autoExecute: true,
        meta: { authority: 'system' }
      });
    }
    return actions;
  }

  // Get valid plays for current player using threaded rules
  const threadedRules = rules || composeRules([baseRuleSet]);
  const validPlays = threadedRules.getValidPlays(state, state.currentPlayer);
  validPlays.forEach((domino: Domino) => {
    actions.push({
      type: 'play',
      player: state.currentPlayer,
      dominoId: domino.id.toString()
    });
  });

  return actions;
}

/**
 * Gets valid scoring actions
 */
function getScoringActions(state: GameState): GameAction[] {
  const actions: GameAction[] = [];

  // Only the current player can agree to score the hand
  if (!state.consensus.scoreHand.has(state.currentPlayer)) {
    actions.push({ type: 'agree-score-hand', player: state.currentPlayer });
  }

  // If all have agreed, the hand can be scored
  if (state.consensus.scoreHand.size === state.players.length) {
    actions.push({
      type: 'score-hand',
      autoExecute: true,
      meta: { authority: 'system' }
    });
  }

  return actions;
}

/**
 * Helper function to get bid comparison value for internal use
 */
function getBidValue(bid: Bid): number {
  if (bid.value === undefined) return 0;
  switch (bid.type) {
    case BID_TYPES.POINTS:
      return bid.value;
    case BID_TYPES.MARKS:
      return bid.value * 42;
    default:
      return 0;
  }
}

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

export const baseRuleSet: GameRuleSet = {
  name: 'base',

  /**
   * Add standard bids (points and marks) during bidding phase.
   * Filter out special trump selections during trump selection phase.
   */
  getValidActions: (state, prev) => {
    let actions = [...prev];

    // Add standard bids during bidding phase
    if (state.phase === 'bidding') {
      const currentPlayer = state.currentPlayer;
      const player = state.players[currentPlayer];
      if (!player) return actions;

      // Helper to validate bids
      const validateBid = (bid: Bid): boolean => {
        if (!state.bids) return false;
        const playerBids = state.bids.filter(b => b.player === bid.player);
        if (playerBids.length > 0) return false;
        if (state.currentPlayer !== bid.player) return false;

        const previousBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);

        // Opening bid constraints
        if (previousBids.length === 0) {
          if (bid.type === BID_TYPES.POINTS) {
            return bid.value !== undefined &&
                   bid.value >= GAME_CONSTANTS.MIN_BID &&
                   bid.value <= GAME_CONSTANTS.MAX_BID;
          }
          if (bid.type === BID_TYPES.MARKS) {
            return bid.value !== undefined && bid.value >= 1 && bid.value <= 2;
          }
          return false;
        }

        // Subsequent bids must be higher
        const lastBid = previousBids[previousBids.length - 1];
        if (!lastBid) return false;

        const lastBidValue = getBidValue(lastBid);
        const currentBidValue = getBidValue(bid);

        if (currentBidValue <= lastBidValue) return false;

        if (bid.type === BID_TYPES.POINTS) {
          return bid.value !== undefined &&
                 bid.value <= GAME_CONSTANTS.MAX_BID &&
                 (lastBid.type !== BID_TYPES.POINTS || bid.value > lastBid.value!);
        }
        if (bid.type === BID_TYPES.MARKS) {
          return isValidMarkBid(bid, lastBid, previousBids);
        }
        return false;
      };

      // Points bids (30-42)
      for (let points = GAME_CONSTANTS.MIN_BID; points <= GAME_CONSTANTS.MAX_BID; points++) {
        const bid: Bid = { type: BID_TYPES.POINTS, value: points, player: currentPlayer };
        if (validateBid(bid)) {
          actions.push({ type: 'bid', player: currentPlayer, bid: BID_TYPES.POINTS, value: points });
        }
      }

      // Marks bids (1-4)
      for (let marks = 1; marks <= 4; marks++) {
        const bid: Bid = { type: BID_TYPES.MARKS, value: marks, player: currentPlayer };
        if (validateBid(bid)) {
          actions.push({ type: 'bid', player: currentPlayer, bid: BID_TYPES.MARKS, value: marks });
        }
      }
    }

    // Filter out special trump selections during trump selection phase
    actions = actions.filter(action => {
      if (action.type === 'select-trump' &&
          (action.trump?.type === 'nello' || action.trump?.type === 'sevens')) {
        return false;
      }
      return true;
    });

    return actions;
  },

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
     * Base: Check if outcome is mathematically determined
     * - Bidding team reached their bid
     * - Bidding team cannot possibly reach their bid
     * - Defending team has set the bid
     * - All 7 tricks complete
     */
    checkHandOutcome(state: GameState) {
      // First check if all tricks complete
      if (state.tricks.length >= GAME_CONSTANTS.TRICKS_PER_HAND) {
        return {
          isDetermined: true,
          reason: 'All tricks played'
        };
      }

      // Then check for early termination based on score
      return checkStandardHandOutcome(state);
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
    calculateTrickWinner(state: GameState, trick: Play[], _prev: number): number {
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
    },

    // ============================================
    // VALIDATION RULES
    // ============================================

    isValidPlay: (_state: GameState, _domino: Domino, _playerId: number, prev: boolean): boolean => {
      // Base implementation is in compose.ts - just pass through
      return prev;
    },

    getValidPlays: (_state: GameState, _playerId: number, prev: Domino[]): Domino[] => {
      // Base implementation is in compose.ts - just pass through
      return prev;
    },

    isValidBid: (state: GameState, bid: Bid, _playerHand: Domino[] | undefined, _prev: boolean): boolean => {
      // Check if bids array exists and player has already bid
      if (!state.bids) return false;
      const playerBids = state.bids.filter(b => b.player === bid.player);
      if (playerBids.length > 0) return false;

      // Check turn order only if we're in an active bidding phase
      if (state.phase === 'bidding' && state.currentPlayer !== bid.player) return false;

      // Pass is always valid if player hasn't bid yet
      if (bid.type === BID_TYPES.PASS) return true;

      const previousBids = state.bids.filter(b => b.type !== BID_TYPES.PASS);

      // Opening bid constraints
      if (previousBids.length === 0) {
        // Opening bid validation for POINTS and MARKS only
        switch (bid.type) {
          case BID_TYPES.POINTS:
            return bid.value !== undefined &&
                   bid.value >= GAME_CONSTANTS.MIN_BID &&
                   bid.value <= GAME_CONSTANTS.MAX_BID;
          case BID_TYPES.MARKS:
            // Maximum opening bid is 2 marks
            return bid.value !== undefined && bid.value >= 1 && bid.value <= 2;
          default:
            return false;
        }
      }

      // All subsequent bids must be higher than current high bid
      const lastBid = previousBids[previousBids.length - 1];
      if (!lastBid) {
        throw new Error('No previous bid found when validating subsequent bid');
      }

      // Get comparison values using helper function
      const lastBidValue = getBidValue(lastBid);
      const currentBidValue = getBidValue(bid);

      if (currentBidValue <= lastBidValue) return false;

      // Subsequent bid validation for POINTS and MARKS only
      switch (bid.type) {
        case BID_TYPES.POINTS:
          return bid.value !== undefined &&
                 bid.value <= GAME_CONSTANTS.MAX_BID &&
                 (lastBid.type !== BID_TYPES.POINTS || bid.value > lastBid.value!);
        case BID_TYPES.MARKS:
          return isValidMarkBid(bid, lastBid, previousBids);
        default:
          return false;
      }
    },

    /**
     * HOW do we compare bids for bidding order?
     *
     * Base: POINTS and MARKS only
     * - Points bids compare by value (30-41)
     * - Marks bids worth 42 points each (value * 42)
     */
    getBidComparisonValue: (bid: Bid, _prev: number): number => {
      if (bid.value === undefined) return 0;
      switch (bid.type) {
        case BID_TYPES.POINTS:
          return bid.value;
        case BID_TYPES.MARKS:
          return bid.value * 42;
        default:
          return 0;
      }
    },

    /**
     * HOW do we validate trump selection?
     *
     * Base: Standard trump types only
     * - Suit trump (0-6): blanks, ones, twos, threes, fours, fives, sixes
     * - Doubles trump
     * - No trump
     */
    isValidTrump: (trump: TrumpSelection, _prev: boolean): boolean => {
      if (trump.type === 'suit') {
        return trump.suit !== undefined && trump.suit >= 0 && trump.suit <= 6;
      }
      return trump.type === 'doubles' || trump.type === 'no-trump';
    },

    /**
     * HOW do we calculate final score at end of hand?
     *
     * Base: Delegate to core scoring function
     * Determines marks awarded based on bid type and team performance
     */
    calculateScore: (state: GameState, _prev: [number, number]): [number, number] => {
      return calculateRoundScore(state);
    }
  }
};
