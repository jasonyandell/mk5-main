/**
 * Rule composition via reduce pattern.
 *
 * Composes multiple Layers into a single GameRules implementation.
 * Later layers can override earlier layers' rules.
 *
 * Pattern: Monadic composition where each rule is a reducer that takes
 * the previous result and returns a new result.
 */

import type { GameRules, Layer } from './types';
import type { GameState, GameAction, Bid, TrumpSelection, Domino, GamePhase, LedSuit } from '../types';
import { getNextPlayer as getNextPlayerCore } from '../core/players';
import { getLedSuit as getLedSuitCore, getTrumpSuit, dominoHasSuit, isRegularSuitTrump, isDoublesTrump } from '../core/dominoes';
import { calculateRoundScore as calculateScoreBase } from '../core/scoring';
import { BID_TYPES } from '../constants';

/**
 * Base implementation for getBidComparisonValue
 */
function getBidComparisonValueBase(bid: Bid): number {
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
 * Base implementation for isValidTrump
 */
function isValidTrumpBase(trump: TrumpSelection): boolean {
  if (trump.type === 'suit') {
    return trump.suit !== undefined && trump.suit >= 0 && trump.suit <= 6;
  }
  return trump.type === 'doubles' || trump.type === 'no-trump';
}

/**
 * Minimal base implementation for isValidBid
 * Real implementation is in base rule set
 */
function isValidBidBase(_state: GameState, _bid: Bid, _playerHand?: import('../types').Domino[]): boolean {
  return false; // Base rule set will override this
}

/**
 * Base implementation for suitsWithTrump - gets what suits a domino can be played as
 *
 * Exported for direct use by AI modules that need the base logic without full rule composition.
 */
export function suitsWithTrumpBase(state: GameState, domino: Domino): LedSuit[] {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;

  // Doubles trump: doubles belong only to suit 7
  if (isDoublesTrump(trumpSuit)) {
    if (isDouble) {
      return [7 as LedSuit];
    }
    return [domino.high as LedSuit, domino.low as LedSuit];
  }

  // Regular suit trump (0-6)
  if (isRegularSuitTrump(trumpSuit)) {
    if (dominoHasSuit(domino, trumpSuit)) {
      return [trumpSuit as LedSuit];
    }
    if (isDouble) {
      return [domino.high as LedSuit];
    }
    return [domino.high as LedSuit, domino.low as LedSuit];
  }

  // No trump: natural suits
  if (isDouble) {
    return [domino.high as LedSuit];
  }
  return [domino.high as LedSuit, domino.low as LedSuit];
}

/**
 * Base implementation for canFollow
 */
function canFollowBase(state: GameState, led: LedSuit, domino: Domino): boolean {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;

  // Doubles led (suit 7): only doubles can follow
  if (led === 7) {
    return isDouble;
  }

  // Doubles trump: doubles belong only to suit 7, can't follow regular suits
  if (isDoublesTrump(trumpSuit) && isDouble) {
    return false;
  }

  // Regular suit trump: trump dominoes can't follow non-trump suits
  if (isRegularSuitTrump(trumpSuit) && dominoHasSuit(domino, trumpSuit)) {
    return led === trumpSuit;
  }

  // Check if domino has the led suit
  return dominoHasSuit(domino, led);
}

/**
 * Base implementation for rankInTrick
 */
function rankInTrickBase(state: GameState, led: LedSuit, domino: Domino): number {
  const trumpSuit = getTrumpSuit(state.trump);
  const isDouble = domino.high === domino.low;
  const pipSum = domino.high + domino.low;

  // Check if domino is trump
  const isTrump = isDoublesTrump(trumpSuit)
    ? isDouble
    : isRegularSuitTrump(trumpSuit) && dominoHasSuit(domino, trumpSuit);

  if (isTrump) {
    return isDouble ? 200 + domino.high + 50 : 200 + pipSum;
  }

  // Check if domino follows suit
  let followsSuit = false;
  if (led === 7) {
    followsSuit = isDouble;
  } else if (!(isDoublesTrump(trumpSuit) && isDouble)) {
    followsSuit = dominoHasSuit(domino, led);
  }

  if (followsSuit) {
    return isDouble ? 50 + domino.high + 50 : 50 + pipSum;
  }

  return pipSum;
}

/**
 * Base implementation for isValidPlay using canFollow
 */
function isValidPlayBase(
  state: GameState,
  domino: Domino,
  playerId: number
): boolean {
  if (state.phase !== 'playing' || state.trump.type === 'not-selected') return false;

  const player = state.players[playerId];
  if (!player || !player.hand.some(d => d.id === domino.id)) return false;

  // First play of trick is always legal
  if (state.currentTrick.length === 0) return true;

  const leadSuit = state.currentSuit;
  if (leadSuit === -1) return true;

  // Check if player must follow suit (has any domino that can follow)
  const mustFollow = player.hand.some(d => canFollowBase(state, leadSuit as LedSuit, d));
  if (!mustFollow) return true;

  // Must play a domino that follows suit
  return canFollowBase(state, leadSuit as LedSuit, domino);
}

/**
 * Base implementation for getValidPlays using canFollow
 */
function getValidPlaysBase(
  state: GameState,
  playerId: number
): Domino[] {
  if (state.phase !== 'playing' || state.trump.type === 'not-selected') return [];

  const player = state.players[playerId];
  if (!player) return [];

  // First play of trick - all dominoes are valid
  if (state.currentTrick.length === 0) return [...player.hand];

  const leadSuit = state.currentSuit;
  if (leadSuit === -1) return [...player.hand];

  // Filter to dominoes that can follow suit
  const followers = player.hand.filter(d => canFollowBase(state, leadSuit as LedSuit, d));

  // If player has dominoes that can follow, must play one of them
  return followers.length > 0 ? followers : [...player.hand];
}

/**
 * Compose multiple layers into a single GameRules implementation.
 *
 * Layers are applied left-to-right via reduce. Each layer's rule receives
 * the previous layer's result and can either pass it through or override it.
 *
 * @param layers Array of layers to compose (base should be first)
 * @returns Composed GameRules with all 13 methods
 */
export function composeRules(layers: Layer[]): GameRules {
  return {
    getTrumpSelector: (state, bid) => {
      let result = bid.player; // Base identity: bidder selects trump

      for (const layer of layers) {
        if (layer.rules?.getTrumpSelector) {
          result = layer.rules.getTrumpSelector(state, bid, result);
        }
      }

      return result;
    },

    getFirstLeader: (state, selector, trump) => {
      let result = selector; // Base identity: selector leads

      for (const layer of layers) {
        if (layer.rules?.getFirstLeader) {
          result = layer.rules.getFirstLeader(state, selector, trump, result);
        }
      }

      return result;
    },

    getNextPlayer: (state, current) => {
      let result = getNextPlayerCore(current); // Base identity: use core helper

      for (const layer of layers) {
        if (layer.rules?.getNextPlayer) {
          result = layer.rules.getNextPlayer(state, current, result);
        }
      }

      return result;
    },

    isTrickComplete: (state) => {
      let result = state.currentTrick.length === 4; // Base identity: 4 plays

      for (const layer of layers) {
        if (layer.rules?.isTrickComplete) {
          result = layer.rules.isTrickComplete(state, result);
        }
      }

      return result;
    },

    checkHandOutcome: (state) => {
      let result: ReturnType<GameRules['checkHandOutcome']> = { isDetermined: false }; // Base identity: not yet determined

      for (const layer of layers) {
        if (layer.rules?.checkHandOutcome) {
          result = layer.rules.checkHandOutcome(state, result);
        }
      }

      return result;
    },

    getLedSuit: (state, domino) => {
      let result = getLedSuitCore(domino, state.trump); // Base identity: use core helper (swapped params)

      for (const layer of layers) {
        if (layer.rules?.getLedSuit) {
          result = layer.rules.getLedSuit(state, domino, result);
        }
      }

      return result;
    },

    suitsWithTrump: (state, domino) => {
      let result = suitsWithTrumpBase(state, domino);

      for (const layer of layers) {
        if (layer.rules?.suitsWithTrump) {
          result = layer.rules.suitsWithTrump(state, domino, result);
        }
      }

      return result;
    },

    canFollow: (state, led, domino) => {
      let result = canFollowBase(state, led, domino);

      for (const layer of layers) {
        if (layer.rules?.canFollow) {
          result = layer.rules.canFollow(state, led, domino, result);
        }
      }

      return result;
    },

    rankInTrick: (state, led, domino) => {
      let result = rankInTrickBase(state, led, domino);

      for (const layer of layers) {
        if (layer.rules?.rankInTrick) {
          result = layer.rules.rankInTrick(state, led, domino, result);
        }
      }

      return result;
    },

    calculateTrickWinner: (state, trick) => {
      if (trick.length === 0) {
        throw new Error('Trick cannot be empty');
      }

      const ledSuit = state.currentSuit as LedSuit;

      // Compose rankInTrick inline to use for ranking
      const getRank = (domino: Domino): number => {
        let rank = rankInTrickBase(state, ledSuit, domino);
        for (const layer of layers) {
          if (layer.rules?.rankInTrick) {
            rank = layer.rules.rankInTrick(state, ledSuit, domino, rank);
          }
        }
        return rank;
      };

      // Find play with highest rank
      let winner = trick[0]!;
      let maxRank = getRank(winner.domino);

      for (let i = 1; i < trick.length; i++) {
        const play = trick[i]!;
        const rank = getRank(play.domino);
        if (rank > maxRank) {
          winner = play;
          maxRank = rank;
        }
      }

      // Allow layers to override the winner
      let result = winner.player;
      for (const layer of layers) {
        if (layer.rules?.calculateTrickWinner) {
          result = layer.rules.calculateTrickWinner(state, trick, result);
        }
      }

      return result;
    },

    // ============================================
    // VALIDATION RULES
    // ============================================

    isValidPlay: (state, domino, playerId) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.isValidPlay?.(state, domino, playerId, prev) ?? prev,
        isValidPlayBase(state, domino, playerId)  // Base implementation
      ),

    getValidPlays: (state, playerId) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.getValidPlays?.(state, playerId, prev) ?? prev,
        getValidPlaysBase(state, playerId)  // Base implementation
      ),

    isValidBid: (state, bid, playerHand) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.isValidBid?.(state, bid, playerHand, prev) ?? prev,
        isValidBidBase(state, bid, playerHand)  // Base implementation
      ),

    getBidComparisonValue: (bid) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.getBidComparisonValue?.(bid, prev) ?? prev,
        getBidComparisonValueBase(bid)
      ),

    isValidTrump: (trump) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.isValidTrump?.(trump, prev) ?? prev,
        isValidTrumpBase(trump)
      ),

    calculateScore: (state) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.calculateScore?.(state, prev) ?? prev,
        calculateScoreBase(state)
      ),

    getPhaseAfterHandComplete: (state) =>
      layers.reduce(
        (prev, layer) =>
          layer.rules?.getPhaseAfterHandComplete?.(state, prev) ?? prev,
        'bidding' as GamePhase
      )
  };
}

/**
 * Apply layers' getValidActions methods to transform an action list.
 *
 * Layers are applied left-to-right. Each layer receives the actions from
 * previous layers and can filter, annotate, or add actions.
 *
 * Internal helper for composeGetValidActions.
 */
function applyLayerActions(
  layers: readonly Layer[],
  state: GameState,
  baseActions: GameAction[]
): GameAction[] {
  let result = baseActions;

  for (const layer of layers) {
    if (layer.getValidActions) {
      result = layer.getValidActions(state, result);
    }
  }

  return result;
}

/**
 * Compose a getValidActions function via function composition.
 *
 * Returns a function that calls the base generator, then threads results
 * through each layer's getValidActions method.
 *
 * @param layers Array of layers to compose
 * @param base Base function that generates structural actions
 * @returns Composed getValidActions function
 */
export function composeGetValidActions(
  layers: readonly Layer[],
  base: (state: GameState) => GameAction[]
): (state: GameState) => GameAction[] {
  return (state: GameState) => {
    const baseActions = base(state);
    return applyLayerActions(layers, state, baseActions);
  };
}
