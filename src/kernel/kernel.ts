/**
 * Pure game kernel operations.
 * All functions are pure - no mutations, no side effects, no class methods.
 * These functions take state as input and return new state as output.
 */

import type { GameState, GameAction } from '../game/types';
import type {
  GameView,
  ValidAction,
  PlayerInfo,
  ViewTransition,
  MultiplayerGameState,
  PlayerSession,
  Capability,
  Result
} from '../multiplayer/types';
import { ok, err } from '../multiplayer/types';
import { authorizeAndExecute } from '../multiplayer/authorization';
import { filterActionsForSession, getVisibleStateForSession, resolveSessionForAction } from '../multiplayer/capabilities';
import { updatePlayerSession } from '../multiplayer/stateLifecycle';
import { cloneGameState, getNextStates } from '../game/core/state';
import type { ExecutionContext } from '../game/types/execution';
import { actionToId } from '../game/core/actions';

/**
 * Execute a game action with pure state transition.
 * Handles authorization, execution, and auto-execute processing.
 */
export function executeKernelAction(
  state: MultiplayerGameState,
  playerId: string,
  action: GameAction,
  ctx: ExecutionContext
) {
  const request = { playerId, action };
  const result = authorizeAndExecute(state, request, ctx);

  if (!result.success) {
    return result;
  }

  // Process auto-execute actions
  return processAutoExecuteActions(result.value, ctx);
}

/**
 * Process scripted auto-execute actions until exhausted (pure).
 */
export function processAutoExecuteActions(
  initialState: MultiplayerGameState,
  ctx: ExecutionContext,
  maxIterations: number = 50
) {
  let state = initialState;
  let iterations = 0;

  while (iterations < maxIterations) {
    const actions = ctx.getValidActions(state.coreState);
    const autoAction = actions.find((a: GameAction) => a.autoExecute === true);

    if (!autoAction) {
      break;
    }

    const session = resolveSessionForAction(Array.from(state.players), autoAction);

    if (!session) {
      console.error('Auto-execute failed: no capable session', {
        action: autoAction
      });
      break;
    }

    const result = authorizeAndExecute(
      state,
      { playerId: session.playerId, action: autoAction },
      ctx
    );

    if (!result.success) {
      console.error('Auto-execute failed', {
        action: autoAction,
        error: result.error
      });
      break;
    }

    state = result.value;
    iterations += 1;
  }

  if (iterations === maxIterations) {
    console.error('Auto-execute limit reached');
  }

  return ok(state);
}

/**
 * Build a game view with capability-based filtering (pure).
 * Throws if forPlayerId doesn't correspond to a valid session.
 */
export function buildKernelView(
  state: MultiplayerGameState,
  forPlayerId: string,
  ctx: ExecutionContext,
  metadata: {
    gameId: string;
    layers?: string[];
  }
): GameView {
  const coreState = state.coreState;
  const actionsByPlayer = buildActionsMap(state, ctx);

  const session = state.players.find((p: PlayerSession) => p.playerId === forPlayerId);
  if (!session) {
    throw new Error(`buildKernelView: No session found for playerId "${forPlayerId}"`);
  }

  const visibleState = getVisibleStateForSession(coreState, session);
  const validActions = actionsByPlayer[session.playerId] ?? [];

  // Convert validActions to ViewTransitions for client UI rendering
  const transitions: ViewTransition[] = validActions.map((valid) => ({
    id: actionToId(valid.action),
    label: valid.label,
    action: valid.action,
    ...(valid.group ? { group: valid.group } : {}),
    ...(valid.recommended ? { recommended: valid.recommended } : {})
  }));

  const playerInfoList: PlayerInfo[] = Array.from(state.players).map((playerSession: PlayerSession) => ({
    playerId: playerSession.playerIndex,
    controlType: playerSession.controlType,
    sessionId: playerSession.playerId,
    connected: playerSession.isConnected ?? true,
    name: playerSession.name || `Player ${playerSession.playerIndex + 1}`,
    capabilities: playerSession.capabilities.map((cap: Capability) => ({ ...cap }))
  }));

  return {
    state: visibleState,
    validActions,
    transitions,
    players: playerInfoList,
    metadata: {
      gameId: metadata.gameId,
      ...(metadata.layers?.length ? { layers: metadata.layers } : {})
    }
  };
}

/**
 * Build a map of valid actions per player session (pure).
 * Derives allValidActions and transitions from state - cleaner API.
 */
export function buildActionsMap(
  mpState: MultiplayerGameState,
  ctx: ExecutionContext
): Record<string, ValidAction[]> {
  const coreState = mpState.coreState;
  const allValidActions = ctx.getValidActions(coreState);

  // Compute transitions from state (they're derived, not passed)
  const transitions = getNextStates(coreState, ctx);

  const map: Record<string, ValidAction[]> = {};

  for (const session of mpState.players) {
    map[session.playerId] = buildValidActionsForSession(
      session,
      allValidActions,
      transitions,
      coreState
    );
  }

  // Add unfiltered actions
  map['__unfiltered__'] = buildValidActionsForSession(
    undefined,
    allValidActions,
    transitions,
    coreState
  );

  return map;
}

/**
 * Build valid actions for a specific session (pure).
 */
function buildValidActionsForSession(
  session: PlayerSession | undefined,
  allValidActions: GameAction[],
  transitions: ReturnType<typeof getNextStates>,
  coreState: GameState
): ValidAction[] {
  const availableActions = session
    ? filterActionsForSession(session, allValidActions)
    : allValidActions;

  return availableActions.map((action: GameAction) => {
    const transition = findMatchingTransition(action, transitions);
    const actionClone: GameAction = { ...action };
    const group = getActionGroup(action);
    const recommended = isRecommendedAction(action, coreState);

    if ('meta' in action && action.meta) {
      (actionClone as { meta?: unknown }).meta = JSON.parse(JSON.stringify(action.meta));
    }

    const validAction: ValidAction = {
      action: actionClone,
      label: transition?.label || action.type,
      recommended
    };

    if (group !== undefined) {
      validAction.group = group;
    }

    return validAction;
  });
}

/**
 * Find matching transition for an action (pure).
 */
function findMatchingTransition(
  action: GameAction,
  transitions: ReturnType<typeof getNextStates>
) {
  return transitions.find((t) => {
    if (t.action.type !== action.type) return false;

    if ('player' in t.action && 'player' in action) {
      if (t.action.player !== action.player) return false;
    }

    switch (action.type) {
      case 'bid':
        return 'bid' in t.action &&
          t.action.bid === action.bid &&
          t.action.value === action.value;
      case 'select-trump':
        return 'trump' in t.action &&
          JSON.stringify(t.action.trump) === JSON.stringify(action.trump);
      case 'play':
        return 'dominoId' in t.action && 'dominoId' in action &&
          t.action.dominoId === action.dominoId;
      default:
        return true;
    }
  });
}

/**
 * Get UI group for action (pure).
 */
export function getActionGroup(action: GameAction): string | undefined {
  switch (action.type) {
    case 'bid':
      return 'Bidding';
    case 'play':
      return 'Play Domino';
    case 'select-trump':
      return 'Trump Selection';
    case 'pass':
      return 'Pass';
    default:
      return undefined;
  }
}

/**
 * Check if action is recommended (pure).
 */
export function isRecommendedAction(action: GameAction, _state: GameState): boolean {
  // For consensus actions, recommend immediate agreement
  if (action.type === 'agree-trick' || action.type === 'agree-score') {
    return true;
  }

  // Future: Add AI hints for recommended moves
  return false;
}

/**
 * Update player control type (pure state transition).
 */
export function updatePlayerControlPure(
  state: MultiplayerGameState,
  playerIndex: number,
  type: 'human' | 'ai',
  capabilities: Capability[]
): Result<MultiplayerGameState> {
  const targetSession = state.players.find((session: PlayerSession) => session.playerIndex === playerIndex);
  if (!targetSession) {
    return err(`Player ${playerIndex} not found`);
  }

  const updatedStateResult = updatePlayerSession(state, targetSession.playerId, {
    controlType: type,
    capabilities
  });

  if (!updatedStateResult.success) {
    return updatedStateResult;
  }

  const updatedPlayers = updatedStateResult.value.players;

  // Update core state playerTypes to match
  const updatedCoreState = {
    ...updatedStateResult.value.coreState,
    playerTypes: updatedPlayers.map((s: PlayerSession) => s.controlType) as ('human' | 'ai')[]
  };

  return ok({
    ...updatedStateResult.value,
    coreState: updatedCoreState
  });
}

/**
 * Clone a MultiplayerGameState deeply (pure).
 */
export function cloneMultiplayerState(state: MultiplayerGameState): MultiplayerGameState {
  return {
    gameId: state.gameId,
    coreState: cloneGameState(state.coreState),
    players: state.players.map((session: PlayerSession) => ({
      ...session,
      capabilities: session.capabilities.map((cap: Capability) => ({ ...cap }))
    }))
  };
}

/**
 * Clone an actions map deeply (pure).
 */
export function cloneActionsMap(actions: Record<string, ValidAction[]>): Record<string, ValidAction[]> {
  const clone: Record<string, ValidAction[]> = {};
  for (const [key, list] of Object.entries(actions)) {
    clone[key] = list.map(valid => {
      const actionClone: GameAction = { ...valid.action };
      if ('meta' in actionClone && actionClone.meta) {
        actionClone.meta = JSON.parse(JSON.stringify(actionClone.meta));
      }
      return {
        ...valid,
        action: actionClone
      };
    });
  }
  return clone;
}
