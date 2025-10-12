import type { GameAction } from '../types';
import type { MultiplayerGameState, Result } from './types';

/**
 * GameClient - The single source of truth for game state.
 *
 * This is the ONLY interface for reading game state and executing actions.
 * All implementations (local, worker, network) must conform to this interface.
 *
 * Design principles:
 * - Composition over inheritance
 * - Pure authorization + execution at the core
 * - Transport-agnostic (works in-process, cross-worker, or over network)
 * - Observable state changes via subscription
 */
export interface GameClient {
  /**
   * Get the current game state (read-only).
   * This is a synchronous snapshot - state may change after read.
   */
  getState(): MultiplayerGameState;

  /**
   * Request to execute an action on behalf of a player.
   * Authorization and validation happen inside - may fail with error.
   *
   * @returns Promise<Result<void>> - ok if action executed, error otherwise
   */
  requestAction(playerId: number, action: GameAction): Promise<Result<void>>;

  /**
   * Subscribe to state changes.
   * Listener is called immediately with current state, then on every state change.
   *
   * @returns Unsubscribe function
   */
  subscribe(listener: (state: MultiplayerGameState) => void): () => void;

  /**
   * Change player control type (human <-> AI).
   * This manages AI lifecycle - spawning/killing AI workers as needed.
   *
   * @param playerId - Player to change control for
   * @param type - 'human' or 'ai'
   */
  setPlayerControl(playerId: number, type: 'human' | 'ai'): Promise<void>;

  /**
   * Destroy the client and clean up resources.
   * Kills AI workers, closes connections, etc.
   */
  destroy(): void;
}
