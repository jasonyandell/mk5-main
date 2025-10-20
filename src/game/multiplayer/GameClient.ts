import type {
  ActionRequest,
  MultiplayerGameState,
  PlayerSession,
  Result
} from './types';
import type { ValidAction } from '../../shared/multiplayer/protocol';

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
   * Get the current multiplayer game state.
   * Returns a promise to support async transports.
   */
  getState(): Promise<MultiplayerGameState>;

  /**
   * Execute an action on behalf of a player session.
   * Authorization/validation occurs inside the multiplayer layer.
   *
   * @returns Promise<Result<MultiplayerGameState>> - new state or error
   */
  executeAction(request: ActionRequest): Promise<Result<MultiplayerGameState>>;

  /**
   * Join or reconnect a player session.
   *
   * @returns Promise<Result<PlayerSession>> - confirmed session or error
   */
  joinGame(session: PlayerSession): Promise<Result<PlayerSession>>;

  /**
   * Leave (or disconnect) a session.
   */
  leaveGame(playerId: string): Promise<void>;

  /**
   * Get valid actions available to a session (already filtered for capabilities).
   */
  getActions(playerId: string): Promise<ValidAction[]>;

  /**
   * Subscribe to state changes.
   * Listener will be invoked whenever core multiplayer state updates.
   *
   * @returns Unsubscribe function
   */
  subscribe(listener: (state: MultiplayerGameState) => void): () => void;

  /**
   * Change player control type (human <-> AI).
   * This manages AI lifecycle - spawning/killing AI workers as needed.
   *
   * @param playerIndex - Player index (0-3) to change control for
   * @param type - 'human' or 'ai'
   */
  setPlayerControl(playerIndex: number, type: 'human' | 'ai'): Promise<void>;

  /**
   * Destroy the client and clean up resources.
   * Kills AI workers, closes connections, etc.
   */
  destroy(): void;
}
