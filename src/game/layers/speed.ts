/**
 * Speed layer - Auto-execute forced moves.
 *
 * When a player has only one legal action, automatically mark it for execution.
 * This speeds up gameplay by eliminating trivial decisions.
 *
 * Examples of forced moves:
 * - Only one legal play (must follow suit, only have one domino)
 * - Only one legal bid (everyone passed, last bidder)
 * - Consensus actions (complete-trick, score-hand)
 */

import type { Layer } from './types';
import type { GameState, GameAction } from '../types';

export const speedLayer: Layer = {
  name: 'speed',

  /**
   * Auto-execute forced moves.
   *
   * When exactly one action exists for a given player, mark autoExecute: true.
   * GameHost will auto-execute these immediately.
   * No changes to game logic, just action annotation.
   */
  getValidActions: (state: GameState, prev: GameAction[]): GameAction[] => {
    // Don't auto-execute if game is over
    if (state.phase === 'game_end') {
      return prev;
    }

    // Group actions by player (for player-specific actions)
    const actionsByPlayer = new Map<number, GameAction[]>();
    const neutralActions: GameAction[] = [];

    for (const action of prev) {
      if ('player' in action) {
        const playerIndex = action.player;
        if (!actionsByPlayer.has(playerIndex)) {
          actionsByPlayer.set(playerIndex, []);
        }
        actionsByPlayer.get(playerIndex)!.push(action);
      } else {
        neutralActions.push(action);
      }
    }

    // Build result with auto-execute annotations
    const result: GameAction[] = [];

    // Process player-specific actions
    for (const actions of actionsByPlayer.values()) {
      if (actions.length === 1) {
        // Only one legal action for this player - auto-execute it with system authority
        const action = actions[0]!;
        result.push({
          ...action,
          autoExecute: true,
          meta: {
            ...('meta' in action ? action.meta : {}),
            speedMode: true,
            reason: 'only-legal-action',
            authority: 'system' as const
          }
        });
      } else {
        // Multiple actions - keep as-is
        result.push(...actions);
      }
    }

    // Neutral actions (consensus) - auto-execute if they're the only action
    if (neutralActions.length > 0 && actionsByPlayer.size === 0) {
      // Only neutral actions exist - auto-execute them with system authority
      for (const action of neutralActions) {
        result.push({
          ...action,
          autoExecute: true,
          meta: {
            ...('meta' in action ? action.meta : {}),
            speedMode: true,
            reason: 'consensus-action',
            authority: 'system' as const
          }
        });
      }
    } else {
      // Mix of player and neutral actions - keep neutral as-is
      result.push(...neutralActions);
    }

    return result;
  }
};
