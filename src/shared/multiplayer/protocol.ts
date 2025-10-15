/**
 * Protocol definitions for client-server communication.
 *
 * This file defines the complete protocol for game communication.
 * NO game engine imports allowed - this is shared between client and server.
 *
 * Design principles:
 * - Messages are serializable (JSON-compatible)
 * - Client never computes game logic
 * - Server pre-calculates everything the client needs
 * - Protocol is transport-agnostic (works over WebSocket, Worker, or direct calls)
 */

import type { GameState, GameAction } from '../../game/types';

// ============================================================================
// Client -> Server Messages
// ============================================================================

/**
 * Request to create a new game with configuration
 */
export interface CreateGameMessage {
  type: 'CREATE_GAME';
  config: GameConfig;
  clientId: string;
}

/**
 * Request to execute a game action
 */
export interface ExecuteActionMessage {
  type: 'EXECUTE_ACTION';
  gameId: string;
  playerId: string;  // Player identity (e.g., "player-0", "ai-1")
  action: GameAction;
}

/**
 * Request to change player control (human <-> AI)
 */
export interface SetPlayerControlMessage {
  type: 'SET_PLAYER_CONTROL';
  gameId: string;
  playerId: number;  // Player index (0-3)
  controlType: 'human' | 'ai';
}

/**
 * Request to join an existing game
 */
export interface JoinGameMessage {
  type: 'JOIN_GAME';
  gameId: string;
  playerId: number;
  clientId: string;
}

/**
 * Subscribe to game updates
 */
export interface SubscribeMessage {
  type: 'SUBSCRIBE';
  gameId: string;
  clientId: string;
}

/**
 * Unsubscribe from game updates
 */
export interface UnsubscribeMessage {
  type: 'UNSUBSCRIBE';
  gameId: string;
  clientId: string;
}

/**
 * All messages a client can send to server
 */
export type ClientMessage =
  | CreateGameMessage
  | ExecuteActionMessage
  | SetPlayerControlMessage
  | JoinGameMessage
  | SubscribeMessage
  | UnsubscribeMessage;

// ============================================================================
// Server -> Client Messages
// ============================================================================

/**
 * Game successfully created
 */
export interface GameCreatedMessage {
  type: 'GAME_CREATED';
  gameId: string;
  view: GameView;
}

/**
 * Game state has been updated
 */
export interface StateUpdateMessage {
  type: 'STATE_UPDATE';
  gameId: string;
  view: GameView;
  lastAction?: GameAction; // What action caused this update
}

/**
 * Error occurred processing request
 */
export interface ErrorMessage {
  type: 'ERROR';
  gameId?: string;
  error: string;
  requestType: ClientMessage['type']; // Which request failed
}

/**
 * Player joined/left game
 */
export interface PlayerStatusMessage {
  type: 'PLAYER_STATUS';
  gameId: string;
  playerId: number;
  status: 'joined' | 'left' | 'control_changed';
  controlType?: 'human' | 'ai';
}

/**
 * Progress update for long-running operations (e.g., seed finding)
 */
export interface ProgressMessage {
  type: 'PROGRESS';
  gameId: string;
  operation: 'seed_finding' | 'ai_thinking';
  progress: number; // 0-100
  message?: string;
}

/**
 * All messages a server can send to client
 */
export type ServerMessage =
  | GameCreatedMessage
  | StateUpdateMessage
  | ErrorMessage
  | PlayerStatusMessage
  | ProgressMessage;

// ============================================================================
// Core Data Types
// ============================================================================

/**
 * Game configuration for creating new games
 */
export interface GameConfig {
  /** Player control types */
  playerTypes: ('human' | 'ai')[];

  /** Game variant (standard, one-hand, etc.) */
  variant?: GameVariant;

  /** Random seed for deterministic games */
  shuffleSeed?: number;

  /** AI difficulty levels (optional) */
  aiDifficulty?: ('beginner' | 'intermediate' | 'expert')[];

  /** Time limits (optional) */
  timeLimits?: {
    perAction?: number; // ms
    perHand?: number; // ms
  };
}

/**
 * Game variant configuration
 */
export interface GameVariant {
  type: 'standard' | 'one-hand' | 'tournament';

  config?: {
    // For one-hand mode
    targetHand?: number;
    maxAttempts?: number;
    originalSeed?: number;

    // For tournament mode
    bestOf?: number;

    // Future variants can add their config here
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    [key: string]: any;
  };
}

/**
 * Complete view of game for a client.
 * This is everything a client needs to render the game.
 * NO client-side game logic required!
 */
export interface GameView {
  /** Current game state */
  state: GameState;

  /** Pre-calculated valid actions for current player */
  validActions: ValidAction[];

  /** Player information */
  players: PlayerInfo[];

  /** Game metadata */
  metadata: {
    gameId: string;
    variant?: GameVariant;
    created: number; // timestamp
    lastUpdate: number; // timestamp
  };
}

/**
 * Valid action with additional UI hints
 */
export interface ValidAction {
  /** The actual game action */
  action: GameAction;

  /** Human-readable label */
  label: string;

  /** Optional grouping for UI organization */
  group?: string;

  /** Keyboard shortcut hint */
  shortcut?: string;

  /** Is this a recommended action? */
  recommended?: boolean;
}

/**
 * Player information for UI
 */
export interface PlayerInfo {
  playerId: number;
  controlType: 'human' | 'ai';
  sessionId?: string;
  connected: boolean;
  name?: string;
  avatar?: string;
}

// ============================================================================
// Transport Interface
// ============================================================================

/**
 * Transport adapter interface.
 * Implementations handle the actual message passing (WebSocket, Worker, etc.)
 */
export interface IGameAdapter {
  /**
   * Send a message to the server
   */
  send(message: ClientMessage): Promise<void>;

  /**
   * Subscribe to server messages
   */
  subscribe(handler: (message: ServerMessage) => void): () => void;

  /**
   * Clean up resources
   */
  destroy(): void;

  /**
   * Check if connected
   */
  isConnected(): boolean;

  /**
   * Optional: Get connection metadata
   */
  getMetadata?(): {
    type: 'in-process' | 'worker' | 'websocket';
    latency?: number;
    gameId?: string;
  };
}

// ============================================================================
// Helper Types
// ============================================================================

/**
 * Result type for async operations
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