/**
 * Rule composition via reduce pattern.
 *
 * Composes multiple GameLayers into a single GameRules implementation.
 * Later layers can override earlier layers' rules.
 *
 * Pattern: Monadic composition where each rule is a reducer that takes
 * the previous result and returns a new result.
 */

import type { GameRules, GameLayer } from './types';
import type { GameState, GameAction, Bid, TrumpSelection, Domino, RegularSuit } from '../types';
import { DOUBLES_AS_TRUMP } from '../types';
import { getNextPlayer as getNextPlayerCore } from '../core/players';
import { getLedSuit as getLedSuitCore, getTrumpSuit } from '../core/dominoes';
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
 * Real implementation is in base layer
 */
function isValidBidBase(_state: GameState, _bid: Bid, _playerHand?: import('../types').Domino[]): boolean {
  return false; // Base layer will override this
}

/**
 * Base implementation for isValidPlay using suit analysis from game state
 */
function isValidPlayBase(
  state: GameState,
  domino: Domino,
  playerId: number
): boolean {
  if (state.phase !== 'playing' || state.trump.type === 'not-selected') return false;

  // Validate player bounds
  if (playerId < 0 || playerId >= state.players.length) return false;

  const player = state.players[playerId];
  if (!player || !player.hand.some(d => d.id === domino.id)) return false;

  // First play of trick is always legal
  if (state.currentTrick.length === 0) return true;

  // Use the currentSuit from state instead of computing it
  const leadSuit = state.currentSuit;
  if (leadSuit === -1) return true; // No suit to follow

  // Use suit analysis to check if player can follow suit
  if (!player.suitAnalysis) return true; // If no analysis, allow any play

  // Handle doubles trump special case
  if (leadSuit === 7) {
    // When doubles are led (trump = 7), only doubles can follow
    const doubles = player.suitAnalysis.rank.doubles;
    if (doubles && doubles.length > 0) {
      return doubles.some(d => d.id === domino.id);
    }
    return true; // Can't follow doubles, any play is legal
  }

  // Check if trump is being led
  const trumpSuit = getTrumpSuit(state.trump);
  if (trumpSuit === leadSuit) {
    // Trump is led - must follow with trump if possible
    const trumpDominoes = player.suitAnalysis.rank.trump;
    if (trumpDominoes && trumpDominoes.length > 0) {
      return trumpDominoes.some(d => d.id === domino.id);
    }
    return true; // Can't follow trump, any play is legal
  }

  // For regular suits (0-6), check if player can follow
  const suitDominoes = leadSuit >= 0 && leadSuit <= 6
    ? player.suitAnalysis.rank[leadSuit as RegularSuit]
    : undefined;

  // Filter out trump dominoes - they can't follow non-trump suits
  const nonTrumpSuitDominoes = suitDominoes ? suitDominoes.filter(d => {
    // If trump is a regular suit, dominoes containing that suit are trump
    if (trumpSuit >= 0 && trumpSuit <= 6) {
      return d.high !== trumpSuit && d.low !== trumpSuit;
    }
    // If doubles are trump, doubles can't follow regular suits
    if (trumpSuit === DOUBLES_AS_TRUMP) {
      return d.high !== d.low;
    }
    return true;
  }) : [];

  // If player has non-trump dominoes in the led suit, must play one of them
  if (nonTrumpSuitDominoes.length > 0) {
    return nonTrumpSuitDominoes.some(d => d.id === domino.id);
  }

  // If player can't follow suit (no non-trump dominoes in that suit), any play is legal
  return true;
}

/**
 * Base implementation for getValidPlays using suit analysis
 */
function getValidPlaysBase(
  state: GameState,
  playerId: number
): Domino[] {
  if (state.phase !== 'playing' || state.trump.type === 'not-selected') return [];

  const player = state.players[playerId];
  if (!player) return [];
  if (!player.suitAnalysis) return [...player.hand];

  // First play of trick - all dominoes are valid
  if (state.currentTrick.length === 0) return [...player.hand];

  // Use the currentSuit from state instead of computing it
  const leadSuit = state.currentSuit;
  if (leadSuit === -1) return [...player.hand]; // No suit to follow

  // Handle doubles trump special case
  if (leadSuit === 7) {
    const doubles = player.suitAnalysis.rank.doubles || [];
    // Filter to only include doubles still in hand
    const handIds = new Set(player.hand.map(d => d.id));
    const validDoubles = doubles.filter(d => handIds.has(d.id));
    if (validDoubles.length > 0) {
      return validDoubles;
    }
    return [...player.hand]; // Can't follow doubles, all plays valid
  }

  // Check if trump is being led
  const trumpSuit = getTrumpSuit(state.trump);
  if (trumpSuit === leadSuit) {
    // Trump is led - must follow with trump if possible
    const trumpDominoes = player.suitAnalysis.rank.trump || [];
    // Filter to only include trump dominoes still in hand
    const handIds = new Set(player.hand.map(d => d.id));
    const validTrumpPlays = trumpDominoes.filter(d => handIds.has(d.id));
    if (validTrumpPlays.length > 0) {
      return validTrumpPlays;
    }
    return [...player.hand]; // Can't follow trump, all plays valid
  }

  // Get dominoes that can follow the led suit from suit analysis
  const suitDominoes = leadSuit >= 0 && leadSuit <= 6
    ? (player.suitAnalysis.rank[leadSuit as RegularSuit] || [])
    : [];

  // Filter out trump dominoes - they can't follow non-trump suits
  const nonTrumpSuitDominoes = suitDominoes.filter(d => {
    // If trump is a regular suit, dominoes containing that suit are trump
    if (trumpSuit >= 0 && trumpSuit <= 6) {
      return d.high !== trumpSuit && d.low !== trumpSuit;
    }
    // If doubles are trump, doubles can't follow regular suits
    if (trumpSuit === DOUBLES_AS_TRUMP) {
      return d.high !== d.low;
    }
    return true;
  });

  // IMPORTANT: Suit analysis may be stale after plays, so filter to only include
  // dominoes still in the player's hand
  const handIds = new Set(player.hand.map(d => d.id));
  const validSuitPlays = nonTrumpSuitDominoes.filter(d => handIds.has(d.id));

  // If player has non-trump dominoes in the led suit, must play one of them
  if (validSuitPlays.length > 0) {
    return validSuitPlays;
  }

  // If player can't follow suit (no non-trump dominoes in that suit), all dominoes are valid
  return [...player.hand];
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
export function composeRules(layers: GameLayer[]): GameRules {
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
      let result: ReturnType<GameRules['checkHandOutcome']> = null; // Base identity: no early termination

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

    calculateTrickWinner: (state, trick) => {
      // For base, we need to calculate it inline since it requires more logic
      // This will be overridden by base layer's full implementation
      let result = trick[0]?.player ?? 0; // Fallback: first player

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
      )
  };
}

/**
 * Compose layers' getValidActions methods to transform action generation.
 *
 * Layers are applied left-to-right via reduce. Each layer receives
 * the actions from previous layers and can filter, annotate, or add actions.
 *
 * @param layers Array of layers to compose (base should be first)
 * @param state Current game state
 * @param baseActions Base actions from core game engine
 * @returns Transformed action list after all layers applied
 */
export function composeActions(
  layers: readonly GameLayer[],
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
