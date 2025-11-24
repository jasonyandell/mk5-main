/**
 * OneHand layer - Single hand game mode.
 *
 * Handles both phase transition and action automation for one-hand mode:
 * - Action automation: Scripts bidding/trump selection, auto-executes score-hand
 * - Phase transition: Prevents new hand after scoring (returns 'one-hand-complete')
 */

import type { Layer } from './types';
import type { GameState, GamePhase, GameAction } from '../types';

export const oneHandRuleSet: Layer = {
  name: 'oneHand',

  /**
   * Automate bidding/trump and score-hand consensus.
   *
   * Responsibilities:
   * - Inject scripted bidding/trump actions at start (with autoExecute: true)
   * - Auto-execute score-hand action to skip manual consensus
   * - Script actions execute with system authority, bypassing session checks
   */
  getValidActions: (state: GameState, prev: GameAction[]): GameAction[] => {
    // At start of bidding (no bids yet), inject scripted sequence
    if (state.phase === 'bidding') {
      // Use currentPlayer to ensure we inject actions in correct turn order
      const currentPlayer = state.currentPlayer;

      switch (state.bids.length) {
        case 0:
          return [{
            type: 'pass' as const,
            player: currentPlayer,
            autoExecute: true,
            meta: { scriptId: 'one-hand-setup', step: 1, authority: 'system' as const }
          }];
        case 1:
          return [{
            type: 'pass' as const,
            player: currentPlayer,
            autoExecute: true,
            meta: { scriptId: 'one-hand-setup', step: 2, authority: 'system' as const }
          }];
        case 2:
          return [{
            type: 'pass' as const,
            player: currentPlayer,
            autoExecute: true,
            meta: { scriptId: 'one-hand-setup', step: 3, authority: 'system' as const }
          }];
        case 3:
          return [{
            type: 'bid' as const,
            player: currentPlayer,
            bid: 'points' as const,
            value: 30,
            autoExecute: true,
            meta: { scriptId: 'one-hand-setup', step: 4, authority: 'system' as const }
          }];
        default:
          break;
      }
    }

    // During trump selection, inject scripted trump choice
    if (state.phase === 'trump_selection') {
      // Winner of bid (stored in winningBidder) selects trump
      return [
        {
          type: 'select-trump' as const,
          player: state.winningBidder,
          trump: { type: 'suit' as const, suit: 4 }, // Fours as trump
          autoExecute: true,
          meta: { scriptId: 'one-hand-setup', step: 5, authority: 'system' as const }
        }
      ];
    }

    // After scoring phase, auto-execute score-hand for smooth UX
    if (state.phase === 'scoring' && state.consensus.scoreHand.size === 4) {
      // Replace score-hand with auto-executed version to skip manual button click
      return [
        ...prev.filter(a => a.type !== 'score-hand'),
        {
          type: 'score-hand' as const,
          autoExecute: true,
          meta: { scriptId: 'one-hand-scoring', authority: 'system' as const }
        }
      ];
    }

    return prev;
  },

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
