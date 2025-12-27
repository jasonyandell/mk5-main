/**
 * Authorization - Action authorization and execution
 *
 * Core composition: authorization + execution.
 * This is the fundamental operation for multiplayer - ensures only authorized
 * actions can mutate state.
 */

import type { GameState, GameAction } from '../game/types';
import type { MultiplayerGameState, ActionRequest, Result, PlayerSession } from './types';
import type { ExecutionContext } from '../game/types/execution';
import { ok, err } from './types';
import { executeAction, actionsEqual } from '../game/core/actions';
import { filterActionsForSession } from './capabilities';

/**
 * Checks if a player session is authorized to execute a specific action.
 *
 * Authorization rules:
 * 1. Actions without a 'player' field are neutral (consensus) - any player can execute
 * 2. Actions with a 'player' field require 'act-as-player' capability for that player index
 */
export function canPlayerExecuteAction(
  session: PlayerSession,
  action: GameAction,
  state: GameState,
  ctx: ExecutionContext
): boolean {
  const validActions = ctx.getValidActions(state);
  const visibleActions = filterActionsForSession(session, validActions);
  return visibleActions.some(candidate => actionsEqual(candidate, action));
}

/**
 * Core composition: authorization + execution.
 *
 * Two execution paths:
 * 1. System authority: Scripted actions bypass capability checks (e.g., one-hand transformer)
 * 2. Player authority: Standard capability-based authorization (default)
 */
export function authorizeAndExecute(
  mpState: MultiplayerGameState,
  request: ActionRequest,
  ctx: ExecutionContext
): Result<MultiplayerGameState> {
  const { playerId, action } = request;
  const { coreState, players } = mpState;

  const session = players.find(p => p.playerId === playerId);
  if (!session) {
    return err(`No player found with ID: ${playerId}`);
  }

  const meta = (action as { meta?: unknown }).meta;
  const hasSystemAuthority = meta && typeof meta === 'object' &&
    'authority' in meta && meta.authority === 'system';

  if (hasSystemAuthority) {
    const validActions = ctx.getValidActions(coreState);
    const isStructurallyValid = validActions.some(candidate => actionsEqual(candidate, action));
    if (!isStructurallyValid) {
      return err(`Action is not valid in current game state: ${action.type}`);
    }

    const newCoreState = executeAction(coreState, action, ctx.rules);
    return ok({
      ...mpState,
      coreState: newCoreState
    });
  }

  const validActions = ctx.getValidActions(coreState);
  const isStructurallyValid = validActions.some(candidate => actionsEqual(candidate, action));
  if (!isStructurallyValid) {
    return err(`Action is not valid in current game state: ${action.type}`);
  }

  const allowedActions = filterActionsForSession(session, validActions);
  const matchedAction = allowedActions.find(candidate => actionsEqual(candidate, action));

  if (!matchedAction) {
    return err(
      `Player ${playerId} lacks capability to execute action: ${action.type}`
    );
  }

  const newCoreState = executeAction(coreState, action, ctx.rules);

  return ok({
    ...mpState,
    coreState: newCoreState
  });
}
