/**
 * Room - Single orchestrator for game instances.
 *
 * Combines responsibilities of GameServer and GameKernel:
 * - Owns all impurities: sessions, AI, transport, subscriptions
 * - Delegates all pure logic to kernel.ts helpers
 * - Composes ExecutionContext (layers)
 * - Manages state transitions via pure functions
 *
 * Design:
 * - Pure helpers: executeKernelAction, buildKernelView, buildActionsMap, etc.
 * - Owned state: mpState, players cache
 * - Orchestration: AIManager, Transport, subscribers
 *
 * This replaces both GameServer and GameKernel with a unified orchestrator.
 */

import type { GameAction } from '../game/types';
import type { GameConfig } from '../game/types/config';
import type {
  MultiplayerGameState,
  PlayerSession,
  Capability,
  Result
} from '../game/multiplayer/types';
import { ok, err } from '../game/multiplayer/types';
import type { ExecutionContext } from '../game/types/execution';
import { createExecutionContext } from '../game/types/execution';
import type { Connection } from './transports/Transport';
import type {
  GameView,
  ValidAction,
  ClientMessage,
  ServerMessage
} from '../shared/multiplayer/protocol';

import { createMultiplayerGame, updatePlayerSession } from '../game/multiplayer/stateLifecycle';
import {
  executeKernelAction,
  buildKernelView,
  buildActionsMap,
  processAutoExecuteActions,
  cloneMultiplayerState,
  updatePlayerControlPure
} from '../kernel/kernel';
import { buildBaseCapabilities } from '../game/multiplayer/capabilities';
import { createInitialState } from '../game/core/state';
import { AIManager } from './ai/AIManager';
import { InProcessTransport } from './transports/InProcessTransport';

interface ClientSubscription {
  clientId: string;
  playerId?: string;
  perspective?: string;
}

/**
 * Room - Main orchestrator for a game instance.
 *
 * Responsibilities:
 * 1. Compose ExecutionContext (layers)
 * 2. Manage multiplayer state via pure helpers
 * 3. Coordinate AI lifecycle (AIManager)
 * 4. Route protocol messages (Transport)
 * 5. Broadcast state updates to subscribers
 * 6. Manage player sessions and connections
 */
export class Room {
  // === OWNED STATE ===
  private mpState: MultiplayerGameState;
  private readonly ctx: ExecutionContext;
  private readonly players: Map<string, PlayerSession>; // Cache for fast lookup
  private readonly metadata: {
    gameId: string;
    config: GameConfig;
  };

  // === ORCHESTRATION (from GameServer) ===
  private connections: Map<string, Connection> = new Map();
  private aiManager: AIManager | null = null;
  private subscriptions: Map<string, ClientSubscription> = new Map();
  private isDestroyed = false;

  /**
   * Create a new Room instance.
   *
   * @param gameId - Unique game identifier
   * @param config - Game configuration (layers, etc.)
   * @param initialPlayers - Initial player sessions (must be 4 players)
   */
  constructor(gameId: string, config: GameConfig, initialPlayers: PlayerSession[]) {
    // Validate players
    if (initialPlayers.length !== 4) {
      throw new Error(`Expected 4 players, got ${initialPlayers.length}`);
    }

    // Ensure player indices are 0-3
    const indices = initialPlayers.map(p => p.playerIndex).sort();
    if (indices.join(',') !== '0,1,2,3') {
      throw new Error('Player indices must be 0, 1, 2, 3');
    }

    // Normalize players with default capabilities
    const normalizedPlayers = initialPlayers.map(player => ({
      ...player,
      capabilities: player.capabilities ?? buildBaseCapabilities(player.playerIndex, player.controlType)
    }));

    // === 1. COMPOSE EXECUTION CONTEXT ===
    this.ctx = createExecutionContext(config);

    // Store metadata
    this.metadata = {
      gameId,
      config
    };

    // === 2. CREATE MULTIPLAYER STATE ===

    // Create initial game state
    const initialState = createInitialState({
      playerTypes: config.playerTypes,
      ...(config.shuffleSeed !== undefined ? { shuffleSeed: config.shuffleSeed } : {}),
      ...(config.theme !== undefined ? { theme: config.theme } : {}),
      ...(config.colorOverrides !== undefined ? { colorOverrides: config.colorOverrides } : {}),
      ...(config.dealOverrides !== undefined ? { dealOverrides: config.dealOverrides } : {})
    });

    // Call createMultiplayerGame()
    this.mpState = createMultiplayerGame({
      gameId,
      coreState: initialState,
      players: normalizedPlayers
    });

    // === 3. BUILD PLAYERS MAP CACHE ===
    this.players = new Map(this.mpState.players.map(p => [p.playerId, p]));

    // === 5. PROCESS AUTO-EXECUTE ===
    const autoResult = processAutoExecuteActions(this.mpState, this.ctx);
    if (autoResult.success) {
      this.mpState = autoResult.value;
    }

    // === 6. INITIALIZE AI MANAGER AND SPAWN AI CLIENTS ===
    this.aiManager = new AIManager('beginner');

    // Create internal transport for AI clients
    const aiTransport = new InProcessTransport();
    aiTransport.setRoom(this);

    // Spawn AI clients immediately with internal transport
    const aiPlayers = normalizedPlayers.filter(s => s.controlType === 'ai');
    for (const player of aiPlayers) {
      const aiConnection = aiTransport.connect(`ai-${player.playerId}`);
      this.connections.set(`ai-${player.playerId}`, aiConnection);
      this.aiManager.spawnAI(
        player.playerIndex,
        gameId,
        player.playerId,
        aiConnection
      );
    }
  }

  // === KERNEL API (delegated to pure helpers) ===

  /**
   * Execute an action with authorization and auto-execute processing.
   *
   * @param playerId - Player ID executing the action
   * @param action - Game action to execute
   * @returns Result with success/error
   */
  executeAction(playerId: string, action: GameAction): Result<MultiplayerGameState> {
    if (this.isDestroyed) {
      return err('Room has been destroyed');
    }

    // Delegate to executeKernelAction(mpState, playerId, action, ctx)
    const result = executeKernelAction(this.mpState, playerId, action, this.ctx);

    if (!result.success) {
      return result;
    }

    // On success: update mpState, sync players cache
    this.mpState = result.value;
    this.syncPlayersCache();

    // Notify listeners
    this.notifyListeners();

    return result;
  }

  /**
   * Get current game view for a specific player (capability-filtered).
   *
   * @param forPlayerId - Optional player ID to filter view for
   * @returns Game view with filtered state and actions
   */
  getView(forPlayerId?: string): GameView {
    // Delegate to buildKernelView(mpState, forPlayerId, ctx, metadata)
    const layers = this.metadata.config.layers;
    return buildKernelView(this.mpState, forPlayerId, this.ctx, {
      gameId: this.metadata.gameId,
      ...(layers?.length ? { layers } : {})
    });
  }

  /**
   * Get full multiplayer state snapshot (deep clone).
   *
   * @returns Cloned multiplayer state
   */
  getState(): MultiplayerGameState {
    // Delegate to cloneMultiplayerState(mpState)
    return cloneMultiplayerState(this.mpState);
  }

  /**
   * Get valid actions for a specific player.
   *
   * @param playerId - Player ID to get actions for
   * @returns Array of valid actions for the player
   */
  getActionsForPlayer(playerId: string): ValidAction[] {
    // Delegate to buildActionsMap(mpState, ctx)
    const actionsMap = buildActionsMap(this.mpState, this.ctx);
    return actionsMap[playerId] ?? [];
  }

  /**
   * Get valid actions for all players.
   *
   * @returns Map of playerId -> valid actions
   */
  getActionsMap(): Record<string, ValidAction[]> {
    // Delegate to buildActionsMap(mpState, ctx)
    return buildActionsMap(this.mpState, this.ctx);
  }

  /**
   * Change player control type (human/ai).
   *
   * @param playerId - Player ID to update
   * @param type - New control type ('human' or 'ai')
   */
  setPlayerControl(playerId: string, type: 'human' | 'ai'): void {
    if (this.isDestroyed) {
      throw new Error('Room has been destroyed');
    }

    // Find player session
    const session = this.players.get(playerId);
    if (!session) {
      throw new Error(`Player ${playerId} not found`);
    }

    // Build new capabilities
    const capabilities = buildBaseCapabilities(session.playerIndex, type);

    // Call updatePlayerControlPure(mpState, playerIndex, type, capabilities)
    const result = updatePlayerControlPure(
      this.mpState,
      session.playerIndex,
      type,
      capabilities
    );

    if (!result.success) {
      throw new Error(result.error);
    }

    // Update mpState, sync cache
    this.mpState = result.value;
    this.syncPlayersCache();

    // Manage AI lifecycle
    if (this.aiManager) {
      if (type === 'ai') {
        // Create in-process transport and connection for this AI
        const aiTransport = new InProcessTransport();
        aiTransport.setRoom(this);
        const aiConnection = aiTransport.connect(`ai-${playerId}`);
        this.connections.set(`ai-${playerId}`, aiConnection);
        this.aiManager.spawnAI(session.playerIndex, this.metadata.gameId, playerId, aiConnection);
      } else {
        this.aiManager.destroyAI(session.playerIndex);
        // Remove AI connection when switching to human
        this.connections.delete(`ai-${playerId}`);
      }
    }

    // Notify listeners
    this.notifyListeners();
  }

  // === SESSION MANAGEMENT (orchestration logic) ===

  /**
   * Connect or replace a player session.
   *
   * @param session - Player session to join
   */
  joinPlayer(session: PlayerSession): Result<PlayerSession> {
    if (this.isDestroyed) {
      return err('Room has been destroyed');
    }

    // Find existing session
    const existing = this.players.get(session.playerId);
    if (!existing) {
      return err(`Player ${session.playerId} not found`);
    }

    // Merge session logic (from GameKernel.joinPlayer)
    const merged: PlayerSession = {
      ...existing,
      ...session,
      isConnected: true,
      capabilities: (session.capabilities ?? existing.capabilities).map((cap: Capability) => ({ ...cap }))
    };

    // Call updatePlayerSession(mpState, playerId, merged)
    const result = updatePlayerSession(this.mpState, existing.playerId, merged);
    if (!result.success) {
      return err(result.error);
    }

    // Update mpState, sync cache
    this.mpState = result.value;
    this.syncPlayersCache();

    // Notify listeners with new view
    this.notifyListeners();

    return ok(merged);
  }

  /**
   * Disconnect a player session.
   *
   * @param playerId - Player ID to disconnect
   */
  leavePlayer(playerId: string): Result<PlayerSession> {
    if (this.isDestroyed) {
      return err('Room has been destroyed');
    }

    // Find existing session
    const existing = this.players.get(playerId);
    if (!existing) {
      return err(`Player ${playerId} not found`);
    }

    // Set isConnected: false
    const updated: PlayerSession = {
      ...existing,
      isConnected: false
    };

    // Call updatePlayerSession(mpState, playerId, { isConnected: false })
    const result = updatePlayerSession(this.mpState, playerId, updated);
    if (!result.success) {
      return err(result.error);
    }

    // Update mpState, sync cache
    this.mpState = result.value;
    this.syncPlayersCache();

    // Notify listeners
    this.notifyListeners();

    return ok(updated);
  }

  /**
   * Get all player sessions.
   *
   * @returns Array of player sessions
   */
  getPlayers(): readonly PlayerSession[] {
    return this.mpState.players;
  }

  /**
   * Get a specific player session by ID.
   *
   * @param playerId - Player ID to find
   * @returns Player session or undefined
   */
  getPlayer(playerId: string): PlayerSession | undefined {
    return this.players.get(playerId);
  }

  // === SERVER API (from GameServer) ===

  /**
   * Get game ID.
   *
   * @returns Game ID string
   */
  getGameId(): string {
    return this.metadata.gameId;
  }

  /**
   * Handle a protocol message from a client.
   * Routes to appropriate handler based on message type.
   *
   * @param clientId - Client ID sending the message
   * @param message - Protocol message to handle
   * @param connection - Connection object for this client
   */
  handleMessage(clientId: string, message: ClientMessage, connection: Connection): void {
    if (this.isDestroyed) {
      console.error('Room: room has been destroyed');
      this.sendError(clientId, 'Room has been destroyed', 'CREATE_GAME');
      return;
    }

    // Store connection on first message (with strict validation)
    const existingConnection = this.connections.get(clientId);
    if (existingConnection && existingConnection !== connection) {
      throw new Error(`Connection conflict for client ${clientId}: different connection already registered`);
    }
    if (!existingConnection) {
      this.connections.set(clientId, connection);
    }

    try {
      switch (message.type) {
        case 'CREATE_GAME':
          this.handleCreateGame(clientId, message);
          break;

        case 'EXECUTE_ACTION':
          this.handleExecuteAction(clientId, message);
          break;

        case 'SET_PLAYER_CONTROL':
          this.handleSetPlayerControl(clientId, message);
          break;

        case 'JOIN_GAME':
          this.handleJoinGame(clientId, message);
          break;

        case 'LEAVE_GAME':
          this.handleLeaveGame(clientId, message);
          break;

        case 'SUBSCRIBE':
          this.handleSubscribe(clientId, message);
          break;

        case 'UNSUBSCRIBE':
          this.handleUnsubscribe(clientId, message);
          break;

        default: {
          const unknownType = (message as { type: string }).type;
          console.error(`Room: Unknown message type: ${unknownType}`);
          this.sendError(clientId, `Unknown message type: ${unknownType}`, 'CREATE_GAME');
          break;
        }
      }
    } catch (error) {
      console.error('Room: Error processing message:', error);
      this.sendError(clientId, `Error: ${error instanceof Error ? error.message : 'Unknown error'}`, message.type);
    }
  }

  /**
   * Handle client disconnect.
   * Called by transport when client disconnects.
   *
   * @param clientId - Client ID that disconnected
   */
  handleClientDisconnect(clientId: string): void {
    // Remove subscriptions for this client
    const sub = this.subscriptions.get(clientId);
    if (sub && sub.playerId) {
      const result = this.leavePlayer(sub.playerId);
      if (!result.success) {
        console.error(`Failed to leave player ${sub.playerId}:`, result.error);
      }
    }

    this.subscriptions.delete(clientId);
  }

  /**
   * Destroy the room and all resources.
   * Cleans up AI manager, subscriptions, connections.
   */
  destroy(): void {
    if (this.aiManager) {
      this.aiManager.destroyAll();
      this.aiManager = null;
    }

    this.subscriptions.clear();
    this.connections.clear();
    this.isDestroyed = true;
  }

  // === PRIVATE HELPERS ===

  /**
   * Rebuild players Map from mpState.players.
   * Called after state updates to keep cache in sync.
   */
  private syncPlayersCache(): void {
    this.players.clear();
    for (const player of this.mpState.players) {
      this.players.set(player.playerId, player);
    }
  }

  /**
   * Notify all subscribers with updated views.
   *
   * CRITICAL: This is the SOLE place where filtered views are built and sent to clients.
   * Each subscriber receives exactly one filtered view per state change.
   *
   * Builds filtered views for each subscriber based on their perspective and sends
   * messages directly via transport. This ensures:
   * - No redundant filtering work
   * - No race conditions between filter computations
   * - Single source of truth for state updates
   */
  private notifyListeners(): void {
    for (const [clientId, sub] of this.subscriptions) {
      // Build filtered view once per subscriber
      const view = this.getView(sub.perspective);

      const msg: ServerMessage = {
        type: 'STATE_UPDATE',
        gameId: this.metadata.gameId,
        view,
        ...(sub.perspective ? { perspective: sub.perspective } : {})
      };

      // Send message directly - no callback indirection
      this.sendMessage(clientId, msg);
    }
  }

  // === PROTOCOL MESSAGE HANDLERS ===

  /**
   * Handle CREATE_GAME message.
   * Game is already created in constructor, just send initial state.
   */
  private handleCreateGame(clientId: string, _message: ClientMessage): void {
    const view = this.getView();
    this.sendMessage(clientId, {
      type: 'GAME_CREATED',
      gameId: this.metadata.gameId,
      view
    });
  }

  /**
   * Handle EXECUTE_ACTION message.
   * Delegates to executeAction() and broadcasts on success.
   */
  private handleExecuteAction(clientId: string, message: ClientMessage): void {
    if (message.type !== 'EXECUTE_ACTION') return;

    const { playerId, action } = message;
    if (!playerId || !action) {
      this.sendError(clientId, 'Missing playerId or action', 'EXECUTE_ACTION');
      return;
    }

    const result = this.executeAction(playerId, action);
    if (!result.success) {
      this.sendError(clientId, result.error || 'Action execution failed', 'EXECUTE_ACTION');
      return;
    }

    // Broadcast handled by executeAction via notifyListeners()
  }

  /**
   * Handle SET_PLAYER_CONTROL message.
   * Changes player control type and manages AI lifecycle.
   */
  private handleSetPlayerControl(clientId: string, message: ClientMessage): void {
    if (message.type !== 'SET_PLAYER_CONTROL') return;

    const { playerId: playerIndex, controlType } = message;
    if (playerIndex === undefined || !controlType) {
      this.sendError(clientId, 'Missing playerIndex or controlType', 'SET_PLAYER_CONTROL');
      return;
    }

    try {
      // Convert playerIndex to playerId
      const playerId = playerIndex < 2 ? `human-${playerIndex}` : `ai-${playerIndex}`;
      this.setPlayerControl(playerId, controlType);

      // Broadcast handled by setPlayerControl via notifyListeners()
    } catch (error) {
      this.sendError(clientId, `Failed to set player control: ${error instanceof Error ? error.message : 'Unknown error'}`, 'SET_PLAYER_CONTROL');
    }
  }

  /**
   * Handle JOIN_GAME message.
   * Adds or updates a player session.
   */
  private handleJoinGame(clientId: string, message: ClientMessage): void {
    if (message.type !== 'JOIN_GAME') return;

    const { session } = message;
    if (!session) {
      this.sendError(clientId, 'Missing session', 'JOIN_GAME');
      return;
    }

    const result = this.joinPlayer(session);
    if (!result.success) {
      this.sendError(clientId, result.error, 'JOIN_GAME');
      return;
    }

    // Update subscription
    const sub: ClientSubscription = {
      clientId,
      playerId: session.playerId,
      perspective: session.playerId
    };
    this.subscriptions.set(clientId, sub);

    // Send status update
    this.sendMessage(clientId, {
      type: 'PLAYER_STATUS',
      gameId: this.metadata.gameId,
      playerId: session.playerIndex,
      sessionId: session.playerId,
      status: 'joined',
      controlType: session.controlType,
      capabilities: session.capabilities
    });

    // Broadcast handled by joinPlayer via notifyListeners()
  }

  /**
   * Handle LEAVE_GAME message.
   * Disconnects a player session.
   */
  private handleLeaveGame(clientId: string, message: ClientMessage): void {
    if (message.type !== 'LEAVE_GAME') return;

    const { playerId } = message;
    if (!playerId) {
      this.sendError(clientId, 'Missing playerId', 'LEAVE_GAME');
      return;
    }

    const player = this.getPlayer(playerId);
    if (!player) {
      this.sendError(clientId, `Player ${playerId} not found`, 'LEAVE_GAME');
      return;
    }

    const result = this.leavePlayer(playerId);
    if (!result.success) {
      this.sendError(clientId, result.error, 'LEAVE_GAME');
      return;
    }

    // Update subscription - delete properties instead of setting to undefined
    const sub = this.subscriptions.get(clientId);
    if (sub) {
      delete sub.playerId;
      delete sub.perspective;
    }

    // Send status update
    this.sendMessage(clientId, {
      type: 'PLAYER_STATUS',
      gameId: this.metadata.gameId,
      playerId: player.playerIndex,
      sessionId: playerId,
      status: 'left'
    });

    // Broadcast handled by leavePlayer via notifyListeners()
  }

  /**
   * Handle SUBSCRIBE message.
   * Registers a client to receive state updates.
   *
   * Only stores subscription metadata. Future updates are sent by notifyListeners(),
   * which is the sole place where filtered views are computed and sent.
   */
  private handleSubscribe(clientId: string, message: ClientMessage): void {
    if (message.type !== 'SUBSCRIBE') return;

    const { playerId } = message;

    // Store subscription metadata
    const sub: ClientSubscription = {
      clientId
    };
    if (playerId) {
      sub.playerId = playerId;
      sub.perspective = playerId;
    }
    this.subscriptions.set(clientId, sub);

    // Send initial state (filter once and send)
    const view = this.getView(playerId);
    const initialMsg: ServerMessage = {
      type: 'STATE_UPDATE',
      gameId: this.metadata.gameId,
      view,
      ...(playerId ? { perspective: playerId } : {})
    };
    this.sendMessage(clientId, initialMsg);

    // Note: Subsequent updates are sent by notifyListeners() when state changes
  }

  /**
   * Handle UNSUBSCRIBE message.
   * Removes a client subscription.
   */
  private handleUnsubscribe(clientId: string, message: ClientMessage): void {
    if (message.type !== 'UNSUBSCRIBE') return;

    const { playerId } = message;

    // Remove subscription
    const sub = this.subscriptions.get(clientId);
    if (sub && sub.playerId === playerId) {
      this.subscriptions.delete(clientId);
    }
  }

  /**
   * Send a message to a client via connection.
   *
   * @param clientId - Client ID to send to
   * @param message - Server message to send
   */
  private sendMessage(clientId: string, message: ServerMessage): void {
    const connection = this.connections.get(clientId);
    if (!connection) {
      console.error(`sendMessage: No connection found for client ${clientId}`);
      return;
    }
    connection.reply(message);
  }

  /**
   * Send an error message to a client.
   *
   * @param clientId - Client ID to send to
   * @param error - Error message
   * @param requestType - Type of request that failed
   */
  private sendError(clientId: string, error: string, requestType: ClientMessage['type']): void {
    this.sendMessage(clientId, {
      type: 'ERROR',
      gameId: this.metadata.gameId,
      error,
      requestType
    });
  }
}
