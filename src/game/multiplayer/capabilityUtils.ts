import { cloneGameState } from '../core/state';
import type { GameAction, GameState, FilteredGameState } from '../types';
import type { PlayerSession, Capability } from './types';
import { hasCapability, hasCapabilityType } from './types';

const HINT_KEY = 'hint';
const AI_INTENT_KEY = 'aiIntent';
const REQUIRED_CAPABILITIES_KEY = 'requiredCapabilities';

type ActionMeta = Record<string, unknown> & {
  requiredCapabilities?: Capability[];
  hint?: string;
  aiIntent?: string;
  authority?: 'player' | 'system';
};

function normalizeMeta(meta: unknown): ActionMeta | undefined {
  if (meta && typeof meta === 'object') {
    return meta as ActionMeta;
  }
  return undefined;
}

function canSeeHand(session: PlayerSession, playerIndex: number): boolean {
  // Check if session has observe-hands capability for all hands
  const observeAllCap = session.capabilities.find(
    cap => cap.type === 'observe-hands' && cap.playerIndices === 'all'
  );
  if (observeAllCap) {
    return true;
  }

  // Check if session has observe-hands capability for this specific player
  const observeSpecificCap = session.capabilities.find(
    cap => cap.type === 'observe-hands' &&
    Array.isArray(cap.playerIndices) &&
    cap.playerIndices.includes(playerIndex)
  );
  if (observeSpecificCap) {
    return true;
  }

  return false;
}

function pruneMetadata(meta: ActionMeta): Record<string, unknown> {
  const output: Record<string, unknown> = {};

  for (const [key, value] of Object.entries(meta)) {
    if (value === undefined) continue;

    // Future features (see-hints, see-ai-intent) removed - always filter these keys
    if (key === HINT_KEY || key === AI_INTENT_KEY) {
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

  const filteredMeta = pruneMetadata(meta);

  if (Object.keys(filteredMeta).length === 0) {
    // Remove meta property
    const { meta, ...actionWithoutMeta } = action;
    void meta; // Acknowledge we're intentionally ignoring this
    return { ...actionWithoutMeta } as GameAction;
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
): FilteredGameState {
  const clone = cloneGameState(state);

  // Convert to FilteredGameState format with handCount
  const filteredPlayers = clone.players.map((player, index) => {
    const canSee = canSeeHand(session, index);

    if (canSee) {
      return {
        id: player.id,
        name: player.name,
        teamId: player.teamId,
        marks: player.marks,
        hand: player.hand,
        handCount: player.hand.length,
        ...(player.suitAnalysis ? { suitAnalysis: player.suitAnalysis } : {})
      };
    }

    // Hide hand but keep handCount
    return {
      id: player.id,
      name: player.name,
      teamId: player.teamId,
      marks: player.marks,
      hand: [],
      handCount: player.hand.length
    };
  });

  return {
    ...clone,
    players: filteredPlayers
  };
}

/**
 * Attempt to find a session that can execute the provided action.
 * For system authority actions, capabilities are not checked.
 */
export function resolveSessionForAction(
  sessions: PlayerSession[],
  action: GameAction
): PlayerSession | undefined {
  // Check if action has system authority
  const meta = (action as { meta?: unknown }).meta;
  const hasSystemAuthority = meta && typeof meta === 'object' &&
    'authority' in meta && meta.authority === 'system';

  if (hasSystemAuthority) {
    // System authority: match by player index only, no capability checks
    if ('player' in action && typeof action.player === 'number') {
      return sessions.find(session => session.playerIndex === action.player);
    }
    // Neutral action with system authority: use first session
    return sessions[0];
  }

  // Player authority: require capabilities
  if ('player' in action && typeof action.player === 'number') {
    return sessions.find(session =>
      session.playerIndex === action.player &&
      hasCapability(session, { type: 'act-as-player', playerIndex: action.player })
    );
  }

  // Neutral actions: pick the first capable session
  return sessions.find(session =>
    hasCapabilityType(session, 'act-as-player') ||
    session.capabilities.length > 0
  ) ?? sessions[0];
}
