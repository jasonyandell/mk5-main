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

import type { GameAction, FilteredGameState } from '../../game/types';
import type { GameConfig, ActionTransformerConfig, GameActionTransformer } from '../../game/types/config';
import type {
  Capability,
  MultiplayerGameState,
  PlayerSession
} from '../../game/multiplayer/types';

export type { GameConfig, ActionTransformerConfig, GameActionTransformer };

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
  sessions?: PlayerSession[];
}

/**
 * Request to execute a game action
 */
export interface ExecuteActionMessage {
  type: 'EXECUTE_ACTION';
  gameId: string;
  playerId: string;  // Player identity (e.g., "player-0", "ai-1")
  action: GameAction;
  timestamp: number;
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
  session: PlayerSession;
  clientId: string;
}

/**
 * Request to disconnect a player session
 */
export interface LeaveGameMessage {
  type: 'LEAVE_GAME';
  gameId: string;
  playerId: string;
  clientId: string;
}

/**
 * Subscribe to game updates
 */
export interface SubscribeMessage {
  type: 'SUBSCRIBE';
  gameId: string;
  clientId: string;
  playerId?: string;
}

/**
 * Unsubscribe from game updates
 */
export interface UnsubscribeMessage {
  type: 'UNSUBSCRIBE';
  gameId: string;
  clientId: string;
  playerId?: string;
}

/**
 * All messages a client can send to server
 */
export type ClientMessage =
  | CreateGameMessage
  | ExecuteActionMessage
  | SetPlayerControlMessage
  | JoinGameMessage
  | LeaveGameMessage
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
  state: MultiplayerGameState;
  actions: Record<string, ValidAction[]>;
}

/**
 * Game state has been updated
 */
export interface StateUpdateMessage {
  type: 'STATE_UPDATE';
  gameId: string;
  view: GameView;
  state: MultiplayerGameState;
  actions: Record<string, ValidAction[]>;
  perspective?: string;
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
  sessionId: string;
  status: 'joined' | 'left' | 'control_changed';
  controlType?: 'human' | 'ai';
  capabilities?: Capability[];
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
 * Complete view of game for a client.
 * This is everything a client needs to render the game.
 * NO client-side game logic required!
 */
export interface GameView {
  /** Current game state */
  state: FilteredGameState;

  /** Pre-calculated valid actions for current player */
  validActions: ValidAction[];

  /** Player information */
  players: PlayerInfo[];

  /** Game metadata */
  metadata: {
    gameId: string;
    variant?: GameActionTransformer;
    variants?: ActionTransformerConfig[];
    created: number; // timestamp
    lastUpdate: number; // timestamp
  };

  // TODO: Add replay-url capability
  // When session has { type: 'replay-url' } capability:
  //   1. GameHost generates compressed URL via encodeGameUrl(seed, actions)
  //   2. Include replayUrl in GameView response
  //   3. Client can use replayUrl for full game replay without storing full actionHistory
  //   4. Benefits: Smaller state payloads, URL-shareable games, works across sessions
  // replayUrl?: string;
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
  capabilities?: Capability[];
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
 * Matches multiplayer layer Result type (src/game/multiplayer/types.ts)
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
