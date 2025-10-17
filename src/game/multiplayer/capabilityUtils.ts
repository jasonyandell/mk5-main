import { cloneGameState } from '../core/state';
import type { GameAction, GameState } from '../types';
import type { PlayerSession, Capability } from './types';
import { hasCapability, hasCapabilityType } from './types';

const HINT_KEY = 'hint';
const AI_INTENT_KEY = 'aiIntent';
const REQUIRED_CAPABILITIES_KEY = 'requiredCapabilities';

type ActionMeta = Record<string, unknown> & {
  requiredCapabilities?: Capability[];
  hint?: string;
  aiIntent?: string;
};

function normalizeMeta(meta: unknown): ActionMeta | undefined {
  if (meta && typeof meta === 'object') {
    return meta as ActionMeta;
  }
  return undefined;
}

function canSeeHand(session: PlayerSession, playerIndex: number): boolean {
  if (hasCapabilityType(session, 'observe-full-state')) {
    return true;
  }

  if (hasCapabilityType(session, 'observe-all-hands')) {
    return true;
  }

  if (hasCapability(session, { type: 'observe-hand', playerIndex })) {
    return true;
  }

  if (
    playerIndex === session.playerIndex &&
    hasCapabilityType(session, 'observe-own-hand')
  ) {
    return true;
  }

  return false;
}

function pruneMetadata(session: PlayerSession, meta: ActionMeta): Record<string, unknown> {
  const output: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(meta)) {
    if (value === undefined) continue;

    if (key === HINT_KEY) {
      if (hasCapabilityType(session, 'see-hints')) {
        output[key] = value;
      }
      continue;
    }

    if (key === AI_INTENT_KEY) {
      if (hasCapabilityType(session, 'see-ai-intent')) {
        output[key] = value;
      }
      continue;
    }

    if (key === REQUIRED_CAPABILITIES_KEY) {
      // Do not forward capability requirements to clients
      continue;
    }

    output[key] = value;
  }

  return output;
}

function canExecuteActionWithCapabilities(
  session: PlayerSession,
  action: GameAction
): boolean {
  const meta = normalizeMeta((action as { meta?: unknown }).meta);
  const required = meta?.requiredCapabilities;

  if (Array.isArray(required) && required.length > 0) {
    const allowed = required.some(cap => hasCapability(session, cap));
    if (!allowed) {
      return false;
    }
  }

  if ('player' in action && typeof action.player === 'number') {
    return hasCapability(session, { type: 'act-as-player', playerIndex: action.player });
  }

  return true;
}

/**
 * Removes metadata the viewing session is not authorized to see.
 */
export function filterActionForSession(
  session: PlayerSession,
  action: GameAction
): GameAction | null {
  if (!canExecuteActionWithCapabilities(session, action)) {
    return null;
  }

  const meta = normalizeMeta((action as { meta?: unknown }).meta);
  if (!meta) {
    return action;
  }

  const filteredMeta = pruneMetadata(session, meta);

  if (Object.keys(filteredMeta).length === 0) {
    const { meta: _ignored, ...rest } = action;
    return { ...rest } as GameAction;
  }

  return { ...action, meta: filteredMeta } as GameAction;
}

/**
 * Filter a list of actions for a given session.
 */
export function filterActionsForSession(
  session: PlayerSession,
  actions: GameAction[]
): GameAction[] {
  const filtered: GameAction[] = [];

  for (const action of actions) {
    const result = filterActionForSession(session, action);
    if (result) {
      filtered.push(result);
    }
  }

  return filtered;
}

/**
 * Returns a redacted view of state for the provided session.
 * The original state is never mutated.
 */
export function getVisibleStateForSession(
  state: GameState,
  session: PlayerSession
): GameState {
  if (hasCapabilityType(session, 'observe-full-state')) {
    return state;
  }

  const clone = cloneGameState(state);

  clone.players = clone.players.map((player, index) => {
    if (canSeeHand(session, index)) {
      return player;
    }

    const { suitAnalysis: _ignored, ...rest } = player;
    return {
      ...rest,
      hand: []
    };
  });

  return clone;
}

/**
 * Attempt to find a session that can execute the provided action.
 */
export function resolveSessionForAction(
  sessions: PlayerSession[],
  action: GameAction
): PlayerSession | undefined {
  if ('player' in action && typeof action.player === 'number') {
    return sessions.find(session =>
      session.playerIndex === action.player &&
      hasCapability(session, { type: 'act-as-player', playerIndex: action.player })
    );
  }

  // Neutral actions: pick the first capable session
  return sessions.find(session =>
    hasCapabilityType(session, 'act-as-player') ||
    hasCapabilityType(session, 'observe-full-state') ||
    session.capabilities.length > 0
  ) ?? sessions[0];
}
