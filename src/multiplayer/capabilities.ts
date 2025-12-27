/**
 * Capability System - Merged from game/multiplayer/capabilities.ts and capabilityUtils.ts
 *
 * This file contains:
 * - Capability builders (humanCapabilities, aiCapabilities, spectatorCapabilities)
 * - Capability filtering (filterActionsForSession, getVisibleStateForSession)
 * - Session resolution (resolveSessionForAction)
 */

import { cloneGameState } from '../game/core/state';
import type { GameAction, GameState, FilteredGameState } from '../game/types';
import type { PlayerSession, Capability } from './types';
import { hasCapability, hasCapabilityType } from './types';

// ============================================================================
// Capability Builders
// ============================================================================

/**
 * Standard capabilities for a human player.
 * Can act as player and observe their own hand.
 */
export function humanCapabilities(playerIndex: 0 | 1 | 2 | 3): Capability[] {
  return [
    { type: 'act-as-player', playerIndex },
    { type: 'observe-hands', playerIndices: [playerIndex] }
  ];
}

/**
 * Standard capabilities for an AI player.
 * Can act as player and observe their own hand.
 */
export function aiCapabilities(playerIndex: 0 | 1 | 2 | 3): Capability[] {
  return [
    { type: 'act-as-player', playerIndex },
    { type: 'observe-hands', playerIndices: [playerIndex] }
  ];
}

/**
 * Standard capabilities for a spectator.
 * Can observe all hands but cannot execute actions.
 */
export function spectatorCapabilities(): Capability[] {
  return [
    { type: 'observe-hands', playerIndices: 'all' }
  ];
}

/**
 * Build base capability set per control type.
 */
export function buildBaseCapabilities(playerIndex: number, controlType: 'human' | 'ai'): Capability[] {
  const idx = playerIndex as 0 | 1 | 2 | 3;
  return controlType === 'human'
    ? humanCapabilities(idx)
    : aiCapabilities(idx);
}

/**
 * Builder for custom capability sets.
 * Provides fluent API for composing capabilities.
 *
 * @example
 * const caps = buildCapabilities()
 *   .actAsPlayer(0)
 *   .observeHands([0, 1])
 *   .build();
 */
export class CapabilityBuilder {
  private capabilities: Capability[] = [];

  actAsPlayer(playerIndex: 0 | 1 | 2 | 3): this {
    this.capabilities.push({ type: 'act-as-player', playerIndex });
    return this;
  }

  observeHands(playerIndices: number[] | 'all'): this {
    this.capabilities.push({ type: 'observe-hands', playerIndices });
    return this;
  }

  build(): Capability[] {
    return [...this.capabilities];
  }
}

/**
 * Create a new capability builder.
 */
export function buildCapabilities(): CapabilityBuilder {
  return new CapabilityBuilder();
}

// ============================================================================
// Action Filtering
// ============================================================================

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
  const observeAllCap = session.capabilities.find(
    cap => cap.type === 'observe-hands' && cap.playerIndices === 'all'
  );
  if (observeAllCap) {
    return true;
  }

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

/**
 * Prune metadata based on session capabilities.
 *
 * The `requiredCapabilities` field on metadata controls VISIBILITY of metadata fields,
 * not execution authorization. If a session lacks required capabilities, the gated
 * metadata is stripped, but the action itself remains available.
 *
 * Currently:
 * - hint: requires 'see-hints' capability
 * - aiIntent: always stripped (internal use)
 * - requiredCapabilities: always stripped (internal use)
 */
function pruneMetadata(meta: ActionMeta, session: PlayerSession): Record<string, unknown> {
  const output: Record<string, unknown> = {};
  const canSeeHints = hasCapability(session, { type: 'see-hints' });

  for (const [key, value] of Object.entries(meta)) {
    if (value === undefined) continue;

    // Always strip internal metadata
    if (key === AI_INTENT_KEY || key === REQUIRED_CAPABILITIES_KEY) {
      continue;
    }

    // Hint visibility gated by see-hints capability
    if (key === HINT_KEY) {
      if (canSeeHints) {
        output[key] = value;
      }
      continue;
    }

    output[key] = value;
  }

  return output;
}

/**
 * Check if session can EXECUTE this action.
 *
 * Execution authorization is based on act-as-player capability.
 * Note: requiredCapabilities on metadata gates VISIBILITY, not execution.
 */
function canExecuteAction(
  session: PlayerSession,
  action: GameAction
): boolean {
  if ('player' in action && typeof action.player === 'number') {
    return hasCapability(session, { type: 'act-as-player', playerIndex: action.player });
  }

  return true;
}

/**
 * Filter an action for a session's view.
 *
 * Returns null if session cannot execute the action.
 * Otherwise returns action with metadata pruned based on visibility capabilities.
 */
export function filterActionForSession(
  session: PlayerSession,
  action: GameAction
): GameAction | null {
  if (!canExecuteAction(session, action)) {
    return null;
  }

  const meta = normalizeMeta((action as { meta?: unknown }).meta);
  if (!meta) {
    return action;
  }

  const filteredMeta = pruneMetadata(meta, session);

  if (Object.keys(filteredMeta).length === 0) {
    const { meta, ...actionWithoutMeta } = action;
    void meta;
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

// ============================================================================
// State Filtering
// ============================================================================

/**
 * Returns a redacted view of state for the provided session.
 * The original state is never mutated.
 */
export function getVisibleStateForSession(
  state: GameState,
  session: PlayerSession
): FilteredGameState {
  const clone = cloneGameState(state);

  const filteredPlayers = clone.players.map((player, index) => {
    const canSee = canSeeHand(session, index);

    if (canSee) {
      return {
        id: player.id,
        name: player.name,
        teamId: player.teamId,
        marks: player.marks,
        hand: player.hand,
        handCount: player.hand.length
      };
    }

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

// ============================================================================
// Session Resolution
// ============================================================================

/**
 * Attempt to find a session that can execute the provided action.
 * For system authority actions, capabilities are not checked.
 */
export function resolveSessionForAction(
  sessions: PlayerSession[],
  action: GameAction
): PlayerSession | undefined {
  const meta = (action as { meta?: unknown }).meta;
  const hasSystemAuthority = meta && typeof meta === 'object' &&
    'authority' in meta && meta.authority === 'system';

  if (hasSystemAuthority) {
    if ('player' in action && typeof action.player === 'number') {
      return sessions.find(session => session.playerIndex === action.player);
    }
    return sessions[0];
  }

  if ('player' in action && typeof action.player === 'number') {
    return sessions.find(session =>
      session.playerIndex === action.player &&
      hasCapability(session, { type: 'act-as-player', playerIndex: action.player })
    );
  }

  return sessions.find(session =>
    hasCapabilityType(session, 'act-as-player') ||
    session.capabilities.length > 0
  ) ?? sessions[0];
}
