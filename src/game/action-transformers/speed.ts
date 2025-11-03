import type { ActionTransformerFactory } from './types';

/**
 * Speed mode: Auto-execute forced moves.
 *
 * When a player has only one legal action, automatically mark it for execution.
 * This speeds up gameplay by eliminating trivial decisions.
 *
 * Examples of forced moves:
 * - Only one legal play (must follow suit, only have one domino)
 * - Only one legal bid (everyone passed, last bidder)
 * - Consensus actions (complete-trick, score-hand)
 *
 * Implementation:
 * - When exactly one action exists for a given player, mark autoExecute: true
 * - GameHost will auto-execute these immediately
 * - No changes to game logic, just action annotation
 */
export const speedVariant: ActionTransformerFactory = () => (base) => (state) => {
  const baseActions = base(state);

  // Don't auto-execute if game is over
  if (state.phase === 'game_end') {
    return baseActions;
  }

  // Group actions by player (for player-specific actions)
  const actionsByPlayer = new Map<number, typeof baseActions>();
  const neutralActions: typeof baseActions = [];

  for (const action of baseActions) {
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
  const result: typeof baseActions = [];

  // Process player-specific actions
  for (const actions of actionsByPlayer.values()) {
    if (actions.length === 1) {
      // Only one legal action for this player - auto-execute it
      const action = actions[0]!;
      result.push({
        ...action,
        autoExecute: true,
        meta: {
          ...('meta' in action ? action.meta : {}),
          speedMode: true,
          reason: 'only-legal-action'
        }
      });
    } else {
      // Multiple actions - keep as-is
      result.push(...actions);
    }
  }

  // Neutral actions (consensus) - auto-execute if they're the only action
  if (neutralActions.length > 0 && actionsByPlayer.size === 0) {
    // Only neutral actions exist - auto-execute them
    for (const action of neutralActions) {
      result.push({
        ...action,
        autoExecute: true,
        meta: {
          ...('meta' in action ? action.meta : {}),
          speedMode: true,
          reason: 'consensus-action'
        }
      });
    }
  } else {
    // Mix of player and neutral actions - keep neutral as-is
    result.push(...neutralActions);
  }

  return result;
};
