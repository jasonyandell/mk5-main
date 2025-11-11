import type { GameAction, GameState } from '../types';

/**
 * Capability tokens control what a session can see or do.
 * They are composable and purely descriptive.
 *
 * REMOVED CAPABILITIES (future features, not yet implemented):
 * - observe-own-hand, observe-hand, observe-all-hands, observe-full-state
 *   Replaced by: observe-hands with playerIndices array or 'all'
 * - see-hints, see-ai-intent, replace-ai
 *   Future features for tutorials, coaching, debugging
 * - configure-action-transformer, undo-actions
 *   Future features for advanced game configuration
 */
export type Capability =
  | { type: 'act-as-player'; playerIndex: number }
  | { type: 'observe-hands'; playerIndices: number[] | 'all' };

/**
 * Utility: create a capability matcher for comparisons.
 */
function capabilityMatches(candidate: Capability, target: Capability): boolean {
  if (candidate.type !== target.type) {
    return false;
  }

  switch (candidate.type) {
    case 'act-as-player':
      return candidate.playerIndex === (target as typeof candidate).playerIndex;
    case 'observe-hands':
      // For observe-hands, exact match on playerIndices
      const targetCap = target as typeof candidate;
      if (candidate.playerIndices === 'all' || targetCap.playerIndices === 'all') {
        return candidate.playerIndices === targetCap.playerIndices;
      }
      // Arrays must have same indices
      return JSON.stringify([...candidate.playerIndices].sort()) ===
             JSON.stringify([...targetCap.playerIndices].sort());
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
 * Multiplayer game state stores pure GameState and filters on-demand.
 * This is the canonical type from the vision document (remixed-855ccfd5.md lines 108-115).
 * Authority stores pure state; filtering happens per-client in createView().
 *
 * Pure data only: gameId + coreState + players
 * Execution configuration (rulesets, transformers) lives in ExecutionContext.
 */
export interface MultiplayerGameState {
  gameId: string;                           // Unique game identifier
  coreState: GameState;                     // Pure GameState (NOT filtered)
  players: readonly PlayerSession[];        // Immutable player sessions
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
  | { success: true; value: T }
  | { success: false; error: string };

/**
 * Helper to create success result
 */
export function ok<T>(value: T): Result<T> {
  return { success: true, value };
}

/**
 * Helper to create error result
 */
export function err<T>(error: string): Result<T> {
  return { success: false, error };
}
