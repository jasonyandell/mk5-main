/**
 * OneHand rule set - Single hand game mode.
 *
 * Overrides phase transition after hand scoring to prevent new hand from being dealt.
 * Works in conjunction with oneHandActionTransformer which handles bidding/trump automation.
 *
 * This is a RuleSet (execution logic) not an ActionTransformer (action filtering).
 */

import type { GameRuleSet } from './types';
import type { GameState, GamePhase } from '../types';

export const oneHandRuleSet: GameRuleSet = {
  name: 'oneHand',

  rules: {
    /**
     * After hand is scored, transition to 'one-hand-complete' instead of 'bidding'.
     * This prevents dealing a new hand and triggers the completion modal.
     */
    getPhaseAfterHandComplete: (_state: GameState, _prev: GamePhase): GamePhase => {
      return 'one-hand-complete';
    }
  }
};
