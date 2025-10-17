import type { FilteredGameState, GameAction } from '../types';

/**
 * Capability tokens control what a session can see or do.
 * They are composable and purely descriptive.
 */
export type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-own-hand' }
  | { type: 'observe-hand'; playerIndex: number }
  | { type: 'observe-all-hands' }
  | { type: 'observe-full-state' }
  | { type: 'see-hints' }
  | { type: 'see-ai-intent' }
  | { type: 'replace-ai' }
  | { type: 'configure-variant' }
  | { type: 'undo-actions' };

/**
 * Utility: create a capability matcher for comparisons.
 */
function capabilityMatches(candidate: Capability, target: Capability): boolean {
  if (candidate.type !== target.type) {
    return false;
  }

  switch (candidate.type) {
    case 'act-as-player':
    case 'observe-hand':
      return candidate.playerIndex === (target as typeof candidate).playerIndex;
    default:
      return true;
  }
}

export function hasCapability(session: PlayerSession, target: Capability): boolean {
  return session.capabilities.some(cap => capabilityMatches(cap, target));
}

export function hasCapabilityType<C extends Capability['type']>(
  session: PlayerSession,
  type: C
): boolean {
  return session.capabilities.some(cap => cap.type === type);
}

/**
 * Player session information for multiplayer games
 * Separates identity (playerId) from seat position (playerIndex)
 */
export interface PlayerSession {
  playerId: string;              // Unique identifier: "player-0", "ai-1", "alice", etc
  playerIndex: 0 | 1 | 2 | 3;   // Seat position in the game
  controlType: 'human' | 'ai';   // Who controls this player
  isConnected?: boolean;          // Connection status (optional, defaults to true)
  name?: string;                  // Display name (optional)
  capabilities: Capability[];     // Capability tokens determining visibility/authority
}

/**
 * Multiplayer game state wraps FilteredGameState with session management
 * The state is always filtered based on the current perspective's capabilities
 */
export interface MultiplayerGameState {
  state: FilteredGameState;
  sessions: PlayerSession[];
}

/**
 * Request to execute an action on behalf of a player
 */
export interface ActionRequest {
  playerId: string;  // Player identity (e.g., "player-0", "ai-1")
  action: GameAction;
}

/**
 * Result type for operations that can fail
 */
export type Result<T> =
  | { ok: true; value: T }
  | { ok: false; error: string };

/**
 * Helper to create success result
 */
export function ok<T>(value: T): Result<T> {
  return { ok: true, value };
}

/**
 * Helper to create error result
 */
export function err<T>(error: string): Result<T> {
  return { ok: false, error };
}
