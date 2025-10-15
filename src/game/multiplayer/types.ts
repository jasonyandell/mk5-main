import type { GameState, GameAction } from '../types';

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
  capabilities?: string[];        // Future: what this player can do/see (optional)
}

/**
 * Multiplayer game state wraps GameState with session management
 */
export interface MultiplayerGameState {
  state: GameState;
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
