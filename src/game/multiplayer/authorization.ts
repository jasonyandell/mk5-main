import type { GameState, GameAction } from '../types';
import type { MultiplayerGameState, ActionRequest, Result, PlayerSession } from './types';
import type { ExecutionContext } from '../types/execution';
import { ok, err } from './types';
import { executeAction } from '../core/actions';
import { filterActionsForSession } from './capabilityUtils';

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
  ctx: ExecutionContext
): boolean {
  const validActions = ctx.getValidActions(state);
  const visibleActions = filterActionsForSession(session, validActions);
  return visibleActions.some(candidate => actionsMatch(candidate, action));
}

/**
 * Core composition: authorization + execution.
 * This is the fundamental operation for multiplayer - ensures only authorized
 * actions can mutate state.
 *
 * Two execution paths:
 * 1. System authority: Scripted actions bypass capability checks (e.g., one-hand transformer)
 * 2. Player authority: Standard capability-based authorization (default)
 *
 * @param mpState - Current multiplayer game state
 * @param request - Action request from a player
 * @param ctx - Execution context with composed rules and actions
 * @returns Result containing new MultiplayerGameState or error message
 */
export function authorizeAndExecute(
  mpState: MultiplayerGameState,
  request: ActionRequest,
  ctx: ExecutionContext
): Result<MultiplayerGameState> {
  const { playerId, action } = request;
  const { coreState, players } = mpState;

  // Find player by playerId
  const session = players.find(p => p.playerId === playerId);
  if (!session) {
    return err(`No player found with ID: ${playerId}`);
  }

  // System authority: Skip capability checks for scripted actions
  const meta = (action as { meta?: unknown }).meta;
  const hasSystemAuthority = meta && typeof meta === 'object' &&
    'authority' in meta && meta.authority === 'system';

  if (hasSystemAuthority) {
    // Validate action is structurally valid, then execute directly
    const validActions = ctx.getValidActions(coreState);
    const isStructurallyValid = validActions.some(candidate => actionsMatch(candidate, action));
    if (!isStructurallyValid) {
      return err(`Action is not valid in current game state: ${action.type}`);
    }

    // Execute with system authority - no capability checks
    const newCoreState = executeAction(coreState, action, ctx.rules);
    return ok({
      ...mpState,
      coreState: newCoreState
    });
  }

  // Player authority: Standard capability-based authorization
  const validActions = ctx.getValidActions(coreState);
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

  // Execute the action (pure state transition) with composed rules
  const newCoreState = executeAction(coreState, action, ctx.rules);

  // Return new multiplayer state with updated pure state
  // NO filtering here - filtering happens in createView()
  return ok({
    ...mpState,
    coreState: newCoreState
  });
}
