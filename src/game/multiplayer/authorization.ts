import type { GameState, GameAction } from '../types';
import type { MultiplayerGameState, ActionRequest, Result, PlayerSession } from './types';
import { ok, err, hasCapability } from './types';
import { executeAction } from '../core/actions';
import { getValidActions } from '../core/gameEngine';

/**
 * Checks if a player session is authorized to execute a specific action.
 *
 * Authorization rules (per vision document lines 152-183):
 * 1. Actions without a 'player' field are neutral (consensus) - any player can execute
 * 2. Actions with a 'player' field require 'act-as-player' capability for that player index
 *
 * @param session - The player session with capabilities
 * @param action - The action to check authorization for
 * @param _state - Game state (currently unused but kept for future authorization rules)
 */
export function canPlayerExecuteAction(
  session: PlayerSession,
  action: GameAction,
  _state: GameState
): boolean {
  // Neutral actions (no player field) are available to everyone
  if (!('player' in action)) {
    return true;
  }

  // Player-specific actions require act-as-player capability for that index
  if ('player' in action && typeof action.player === 'number') {
    return hasCapability(session, {
      type: 'act-as-player',
      playerIndex: action.player
    });
  }

  return false;
}

/**
 * Core composition: authorization + execution.
 * This is the fundamental operation for multiplayer - ensures only authorized
 * actions can mutate state.
 *
 * @returns Result containing new MultiplayerGameState or error message
 */
export function authorizeAndExecute(
  mpState: MultiplayerGameState,
  request: ActionRequest,
  getValidActionsFn: (state: GameState) => GameAction[] = getValidActions
): Result<MultiplayerGameState> {
  const { playerId, action } = request;
  const { coreState, players } = mpState;

  // Find player by playerId
  const player = players.find(p => p.playerId === playerId);
  if (!player) {
    return err(`No player found with ID: ${playerId}`);
  }

  // Check authorization using capability system
  if (!canPlayerExecuteAction(player, action, coreState)) {
    return err(
      `Player ${playerId} lacks capability to execute action: ${action.type}`
    );
  }

  // Validate action is legal in current state
  const validActions = getValidActionsFn(coreState);
  const isValidAction = validActions.some(validAction => {
    // Compare action types and player fields
    if (validAction.type !== action.type) return false;

    // For actions with player field, compare player IDs
    if ('player' in validAction && 'player' in action) {
      if (validAction.player !== action.player) return false;
    }

    // For specific action types, compare additional fields
    switch (action.type) {
      case 'bid':
        return 'bid' in validAction &&
               validAction.bid === action.bid &&
               validAction.value === action.value;
      case 'select-trump':
        return 'trump' in validAction &&
               JSON.stringify(validAction.trump) === JSON.stringify(action.trump);
      case 'play':
        return 'dominoId' in validAction &&
               validAction.dominoId === action.dominoId;
      default:
        return true;
    }
  });

  if (!isValidAction) {
    return err(
      `Action is not valid in current game state: ${action.type}`
    );
  }

  // Execute the action (pure state transition)
  const newCoreState = executeAction(coreState, action);

  // Return new multiplayer state with updated pure state
  // NO filtering here - filtering happens in createView()
  return ok({
    ...mpState,
    coreState: newCoreState,
    lastActionAt: Date.now()
  });
}
