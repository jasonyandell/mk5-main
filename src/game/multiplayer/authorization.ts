import type { GameState, GameAction } from '../types';
import type { MultiplayerGameState, ActionRequest, Result } from './types';
import { ok, err } from './types';
import { executeAction } from '../core/actions';
import { getValidActions } from '../core/gameEngine';

/**
 * Checks if a player is authorized to execute a specific action.
 *
 * Authorization rules:
 * 1. Actions without a 'player' field are neutral (consensus) - any player can execute
 * 2. Actions with a 'player' field can only be executed by that specific player
 *
 * @param playerIndex - The player's seat index (0-3), not their identity
 * @param action - The action to check authorization for
 * @param _state - Game state (currently unused but kept for future authorization rules)
 */
export function canPlayerExecuteAction(
  playerIndex: number,
  action: GameAction,
  _state: GameState
): boolean {
  // Neutral actions (no player field) are available to everyone
  if (!('player' in action)) {
    return true;
  }

  // Actions with player field are only for that specific player index
  return action.player === playerIndex;
}

/**
 * Gets all valid actions that a specific player can execute.
 * Composes getValidActions with authorization filtering.
 *
 * @param state - Current game state
 * @param playerIndex - The player's seat index (0-3), not their identity
 */
export function getValidActionsForPlayer(
  state: GameState,
  playerIndex: number
): GameAction[] {
  const allActions = getValidActions(state);

  return allActions.filter(action =>
    canPlayerExecuteAction(playerIndex, action, state)
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
  request: ActionRequest,
  getValidActionsFn: (state: GameState) => GameAction[] = getValidActions
): Result<MultiplayerGameState> {
  const { playerId, action } = request;
  const { state, sessions } = mpState;

  // Find player by playerId
  const player = sessions.find(s => s.playerId === playerId);
  if (!player) {
    return err(`No player found with ID: ${playerId}`);
  }

  const playerIndex = player.playerIndex;

  // Check authorization
  if (!canPlayerExecuteAction(playerIndex, action, state)) {
    return err(
      `Player ${playerId} (index ${playerIndex}) is not authorized to execute action: ${action.type}`
    );
  }

  // Validate action is legal in current state
  const validActions = getValidActionsFn(state);
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

  // Convert to FilteredGameState format with handCount
  const filteredPlayers = newState.players.map(player => ({
    id: player.id,
    name: player.name,
    teamId: player.teamId,
    marks: player.marks,
    hand: player.hand,
    handCount: player.hand.length,
    ...(player.suitAnalysis ? { suitAnalysis: player.suitAnalysis } : {})
  }));

  const filteredState: import('../types').FilteredGameState = {
    ...newState,
    players: filteredPlayers
  };

  // Return new multiplayer state with updated game state
  return ok({
    state: filteredState,
    sessions
  });
}
