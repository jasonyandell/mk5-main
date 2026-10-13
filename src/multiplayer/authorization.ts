/**
 * Authorization - Action authorization and execution
 *
 * Core composition: authorization + execution.
 * This is the fundamental operation for multiplayer - ensures only authorized
 * actions can mutate state.
 *
 * Authority never comes from a request payload. A request is matched against
 * the server's own candidates (ctx.getValidActions), and what executes is the
 * matched candidate - never the client's copy, whose metadata is ignored.
 */

import type { GameState, GameAction } from '../game/types';
import type { MultiplayerGameState, ActionRequest, Result, PlayerSession } from './types';
import type { ExecutionContext } from '../game/types/execution';
import { ok, err } from './types';
import { executeAction } from '../game/core/actions';
import { filterActionForSession } from './capabilities';

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

/** Whether a server-generated candidate declares system authority. */
function declaresSystemAuthority(candidate: GameAction): boolean {
  return candidate.meta?.authority === 'system';
}

/** The server's candidate a session may execute for the requested action, if any. */
function authorizedCandidate(
  session: PlayerSession,
  action: GameAction,
  candidates: GameAction[]
): GameAction | undefined {
  return candidates.find(candidate =>
    actionsMatch(candidate, action) && filterActionForSession(session, candidate) !== null
  );
}

function commit(
  mpState: MultiplayerGameState,
  candidate: GameAction,
  ctx: ExecutionContext
): Result<MultiplayerGameState> {
  return ok({
    ...mpState,
    coreState: executeAction(mpState.coreState, candidate, ctx.rules)
  });
}

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
  return authorizedCandidate(session, action, ctx.getValidActions(state)) !== undefined;
}

/**
 * Player authority: execute a client's request (the wire path).
 *
 * The request must match a valid candidate the session holds capabilities for.
 * A client never carries system authority, whatever its payload claims.
 */
export function authorizeAndExecute(
  mpState: MultiplayerGameState,
  request: ActionRequest,
  ctx: ExecutionContext
): Result<MultiplayerGameState> {
  const { playerId, action } = request;

  const session = mpState.players.find(p => p.playerId === playerId);
  if (!session) {
    return err(`No player found with ID: ${playerId}`);
  }

  const candidates = ctx.getValidActions(mpState.coreState);
  if (!candidates.some(candidate => actionsMatch(candidate, action))) {
    return err(`Action is not valid in current game state: ${action.type}`);
  }

  const candidate = authorizedCandidate(session, action, candidates);
  if (!candidate) {
    return err(`Player ${playerId} lacks capability to execute action: ${action.type}`);
  }

  return commit(mpState, candidate, ctx);
}

/**
 * System authority: the engine executes one of its own scripted actions
 * (auto-execute: the one-hand script, redeal, speed-mode forced moves).
 *
 * Reachable only from the server context, and only for a candidate the layers
 * themselves marked `meta.authority: 'system'`. No session is involved.
 */
export function executeSystemAction(
  mpState: MultiplayerGameState,
  action: GameAction,
  ctx: ExecutionContext
): Result<MultiplayerGameState> {
  const candidate = ctx.getValidActions(mpState.coreState).find(c =>
    actionsMatch(c, action) && declaresSystemAuthority(c)
  );
  if (!candidate) {
    return err(`No system-authority action available in current game state: ${action.type}`);
  }

  return commit(mpState, candidate, ctx);
}
