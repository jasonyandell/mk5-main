import type { GameState, GameAction } from '../types';
import type { MultiplayerGameState, ActionRequest, Result, PlayerSession } from './types';
import { ok, err } from './types';
import { executeAction } from '../core/actions';
import { getValidActions } from '../core/gameEngine';
import { filterActionsForSession } from './capabilityUtils';
import type { StateMachine } from '../action-transformers/types';
import { applyActionTransformers } from '../action-transformers/registry';
import type { GameRules } from '../rulesets/types';
import { composeRules, baseRuleSet } from '../rulesets';

// Default rules (base rule set only, no special contracts)
const defaultRules = composeRules([baseRuleSet]);

function actionsMatch(expected: GameAction, actual: GameAction): boolean {
  if (expected.type !== actual.type) {
    return false;
  }

  if ('player' in expected || 'player' in actual) {
    if (!('player' in expected) || !('player' in actual)) {
      return false;
    }
    if (expected.player !== actual.player) {
      return false;
    }
  }

  switch (actual.type) {
    case 'bid':
      return 'bid' in expected &&
        expected.bid === actual.bid &&
        expected.value === actual.value;
    case 'select-trump':
      return 'trump' in expected &&
        JSON.stringify(expected.trump) === JSON.stringify(actual.trump);
    case 'play':
      return 'dominoId' in expected &&
        expected.dominoId === actual.dominoId;
    default:
      return true;
  }
}

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
  state: GameState,
  getValidActionsFn: (state: GameState) => GameAction[] = getValidActions
): boolean {
  const validActions = getValidActionsFn(state);
  const visibleActions = filterActionsForSession(session, validActions);
  return visibleActions.some(candidate => actionsMatch(candidate, action));
}

/**
 * Get all valid actions for a specific player.
 * Pure function per vision document §3.2.3
 *
 * @param mpState - Multiplayer game state
 * @param playerId - String player ID (e.g., "player-0", "ai-1")
 * @param getValidActionsFn - Optional custom state machine (for testing)
 * @returns Array of actions the player can execute
 */
export function getValidActionsForPlayer(
  mpState: MultiplayerGameState,
  playerId: string,
  getValidActionsFn: StateMachine = getValidActions
): GameAction[] {
  const { coreState, players, enabledVariants } = mpState;

  // Find player session
  const session = players.find(p => p.playerId === playerId);
  if (!session || !session.isConnected) {
    return [];
  }

  // Compose variants with base state machine
  const composedMachine = applyActionTransformers(getValidActionsFn, enabledVariants);

  // Get all valid actions from composed machine
  const allValidActions = composedMachine(coreState);

  // Filter by player's capabilities
  return filterActionsForSession(session, allValidActions);
}

/**
 * Core composition: authorization + execution.
 * This is the fundamental operation for multiplayer - ensures only authorized
 * actions can mutate state.
 *
 * @param mpState - Current multiplayer game state
 * @param request - Action request from a player
 * @param getValidActionsFn - State machine for generating valid actions (with variants applied)
 * @param rules - Game rules for execution (defaults to base layer if not provided)
 * @returns Result containing new MultiplayerGameState or error message
 */
export function authorizeAndExecute(
  mpState: MultiplayerGameState,
  request: ActionRequest,
  getValidActionsFn: (state: GameState) => GameAction[] = getValidActions,
  rules: GameRules = defaultRules
): Result<MultiplayerGameState> {
  const { playerId, action } = request;
  const { coreState, players } = mpState;

  // Find player by playerId
  const session = players.find(p => p.playerId === playerId);
  if (!session) {
    return err(`No player found with ID: ${playerId}`);
  }

  // Check authorization using capability system
  const validActions = getValidActionsFn(coreState);
  const isStructurallyValid = validActions.some(candidate => actionsMatch(candidate, action));
  if (!isStructurallyValid) {
    return err(`Action is not valid in current game state: ${action.type}`);
  }

  const allowedActions = filterActionsForSession(session, validActions);
  const matchedAction = allowedActions.find(candidate => actionsMatch(candidate, action));

  if (!matchedAction) {
    return err(
      `Player ${playerId} lacks capability to execute action: ${action.type}`
    );
  }

  // Execute the action (pure state transition) with threaded rules
  const newCoreState = executeAction(coreState, action, rules);

  // Return new multiplayer state with updated pure state
  // NO filtering here - filtering happens in createView()
  return ok({
    ...mpState,
    coreState: newCoreState,
    lastActionAt: Math.max(mpState.lastActionAt, request.timestamp)
  });
}
