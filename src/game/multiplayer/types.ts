import type { GameState, GameAction } from '../types';

/**
 * Player session information for multiplayer games
 */
export interface PlayerSession {
  playerId: number;
  sessionId: string;
  type: 'human' | 'ai';
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
  playerId: number;
  action: GameAction;
  sessionId?: string; // Optional for local games, required for network games
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
