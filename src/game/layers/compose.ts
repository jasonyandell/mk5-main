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
import type { GameState, GameAction } from '../types';
import { getNextPlayer as getNextPlayerCore } from '../core/players';
import { getLedSuit as getLedSuitCore } from '../core/dominoes';

/**
 * Compose multiple layers into a single GameRules implementation.
 *
 * Layers are applied left-to-right via reduce. Each layer's rule receives
 * the previous layer's result and can either pass it through or override it.
 *
 * @param layers Array of layers to compose (base should be first)
 * @returns Composed GameRules with all 7 methods
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
    }
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
  layers: GameLayer[],
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
