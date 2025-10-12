import type { GameState, GameAction } from '../types';
import type { MultiplayerGameState, ActionRequest, Result } from './types';
import { ok, err } from './types';
import { executeAction } from '../core/actions';
import { getValidActions } from '../core/gameEngine';

/**
 * Checks if a player is authorized to execute a specific action.
 *
 * Authorization rules (extracted from playerView.ts filtering logic):
 * 1. Actions without a 'player' field are neutral (consensus) - any player can execute
 * 2. Actions with a 'player' field can only be executed by that specific player
 */
export function canPlayerExecuteAction(
  playerId: number,
  action: GameAction,
  state: GameState
): boolean {
  // Neutral actions (no player field) are available to everyone
  if (!('player' in action)) {
    return true;
  }

  // Actions with player field are only for that specific player
  return action.player === playerId;
}

/**
 * Gets all valid actions that a specific player can execute.
 * Composes getValidActions with authorization filtering.
 */
export function getValidActionsForPlayer(
  state: GameState,
  playerId: number
): GameAction[] {
  const allActions = getValidActions(state);

  return allActions.filter(action =>
    canPlayerExecuteAction(playerId, action, state)
  );
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
  request: ActionRequest
): Result<MultiplayerGameState> {
  const { playerId, action, sessionId } = request;
  const { state, sessions } = mpState;

  // Validate player ID
  if (playerId < 0 || playerId >= 4) {
    return err(`Invalid player ID: ${playerId}`);
  }

  // Validate session if sessionId provided
  if (sessionId !== undefined) {
    const session = sessions.find(s => s.playerId === playerId);
    if (!session) {
      return err(`No session found for player ${playerId}`);
    }
    if (session.sessionId !== sessionId) {
      return err(`Invalid session ID for player ${playerId}`);
    }
  }

  // Check authorization
  if (!canPlayerExecuteAction(playerId, action, state)) {
    return err(
      `Player ${playerId} is not authorized to execute action: ${action.type}`
    );
  }

  // Validate action is legal in current state
  const validActions = getValidActions(state);
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
  const newState = executeAction(state, action);

  // Return new multiplayer state with updated game state
  return ok({
    state: newState,
    sessions
  });
}
