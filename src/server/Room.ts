/**
 * Room - Transport-agnostic game authority.
 *
 * Responsibilities:
 * 1. Owns game state (via multiplayer state lifecycle)
 * 2. Handles messages via simplified protocol
 * 3. Broadcasts updates to connected clients via send callback
 * 4. Manages player sessions
 *
 * Design:
 * - Room doesn't know HOW to send - just calls the send callback
 * - Same Room works for local play and Cloudflare Durable Objects
 * - AI is external (GameClient + attachAIBehavior in local.ts)
 */

import type { GameAction } from '../game/types';
import type { GameConfig } from '../game/types/config';
import type {
  MultiplayerGameState,
  PlayerSession,
  Result,
  GameView,
  ValidAction
} from '../multiplayer/types';
import { ok, err } from '../multiplayer/types';
import type { ExecutionContext } from '../game/types/execution';
import { createExecutionContext } from '../game/types/execution';
import type { ClientMessage, ServerMessage } from '../multiplayer/protocol';

import { createMultiplayerGame, updatePlayerSession } from '../multiplayer/stateLifecycle';
import {
  executeKernelAction,
  buildKernelView,
  buildActionsMap,
  processAutoExecuteActions,
  cloneMultiplayerState,
  updatePlayerControlPure
} from '../kernel/kernel';
import { buildBaseCapabilities } from '../multiplayer/capabilities';
import { createInitialState } from '../game/core/state';

/**
 * Room - Main orchestrator for a game instance.
 *
 * Transport-agnostic: takes a send callback, doesn't manage connections.
 */
export class Room {
  // === OWNED STATE ===
  private mpState: MultiplayerGameState;
  private readonly ctx: ExecutionContext;
  private readonly players: Map<string, PlayerSession>; // Cache for fast lookup
  private readonly config: GameConfig;
  private readonly gameId: string;

  // === TRANSPORT ===
  private readonly send: (clientId: string, message: ServerMessage) => void;
  private connectedClients: Set<string> = new Set();
  private clientToPlayer: Map<string, string> = new Map(); // clientId → playerId

  private isDestroyed = false;

  /**
   * Create a new Room instance.
   *
   * @param config - Game configuration
   * @param send - Callback to send messages to clients
   */
  constructor(config: GameConfig, send: (clientId: string, message: ServerMessage) => void) {
    this.config = config;
    this.send = send;
    this.gameId = `game-${Date.now()}`;

    // === 1. COMPOSE EXECUTION CONTEXT ===
    this.ctx = createExecutionContext(config);

    // === 2. CREATE INITIAL GAME STATE ===
    const initialState = createInitialState({
      playerTypes: config.playerTypes,
      ...(config.shuffleSeed !== undefined ? { shuffleSeed: config.shuffleSeed } : {}),
      ...(config.theme !== undefined ? { theme: config.theme } : {}),
      ...(config.colorOverrides !== undefined ? { colorOverrides: config.colorOverrides } : {}),
      ...(config.dealOverrides !== undefined ? { dealOverrides: config.dealOverrides } : {})
    });

    // === 3. CREATE DEFAULT PLAYER SESSIONS ===
    const playerTypes = config.playerTypes ?? ['human', 'ai', 'ai', 'ai'];
    const defaultPlayers: PlayerSession[] = playerTypes.map((type, i) => ({
      playerId: `player-${i}`,
      playerIndex: i as 0 | 1 | 2 | 3,
      controlType: type,
      isConnected: false,
      capabilities: buildBaseCapabilities(i as 0 | 1 | 2 | 3, type)
    }));

    // === 4. CREATE MULTIPLAYER STATE ===
    this.mpState = createMultiplayerGame({
      gameId: this.gameId,
      coreState: initialState,
      players: defaultPlayers
    });

    // === 5. BUILD PLAYERS MAP CACHE ===
    this.players = new Map(this.mpState.players.map(p => [p.playerId, p]));

    // === 6. PROCESS AUTO-EXECUTE ===
    const autoResult = processAutoExecuteActions(this.mpState, this.ctx);
    if (autoResult.success) {
      this.mpState = autoResult.value;
    }
  }

  // === PUBLIC API ===

  /**
   * Handle new client connection.
   * Adds client to connected set and sends initial state.
   */
  handleConnect(clientId: string): void {
    if (this.isDestroyed) return;

    this.connectedClients.add(clientId);

    // Send initial state (unfiltered for now - will filter by player when joined)
    const view = this.getView();
    this.send(clientId, { type: 'STATE_UPDATE', view });
  }

  /**
   * Handle client message (simplified protocol).
   */
  handleMessage(clientId: string, message: ClientMessage): void {
    if (this.isDestroyed) {
      this.send(clientId, { type: 'ERROR', error: 'Room has been destroyed' });
      return;
    }

    switch (message.type) {
      case 'EXECUTE_ACTION':
        this.handleExecuteAction(clientId, message);
        break;
      case 'JOIN':
        this.handleJoin(clientId, message);
        break;
      case 'SET_CONTROL':
        this.handleSetControl(clientId, message);
        break;
    }
  }

  /**
   * Handle client disconnect.
   */
  handleDisconnect(clientId: string): void {
    this.connectedClients.delete(clientId);
    this.clientToPlayer.delete(clientId);
  }

  // === KERNEL API (delegated to pure helpers) ===

  /**
   * Execute an action with authorization and auto-execute processing.
   */
  executeAction(playerId: string, action: GameAction): Result<MultiplayerGameState> {
    if (this.isDestroyed) {
      return err('Room has been destroyed');
    }

    const result = executeKernelAction(this.mpState, playerId, action, this.ctx);

    if (!result.success) {
      return result;
    }

    this.mpState = result.value;
    this.syncPlayersCache();
    this.notifyListeners();

    return result;
  }

  /**
   * Get current game view for a specific player (capability-filtered).
   */
  getView(forPlayerId?: string): GameView {
    const layers = this.config.layers;
    return buildKernelView(this.mpState, forPlayerId, this.ctx, {
      gameId: this.gameId,
      ...(layers?.length ? { layers } : {})
    });
  }

  /**
   * Get full multiplayer state snapshot (deep clone).
   */
  getState(): MultiplayerGameState {
    return cloneMultiplayerState(this.mpState);
  }

  /**
   * Get valid actions for a specific player.
   */
  getActionsForPlayer(playerId: string): ValidAction[] {
    const actionsMap = buildActionsMap(this.mpState, this.ctx);
    return actionsMap[playerId] ?? [];
  }

  /**
   * Get valid actions for all players.
   */
  getActionsMap(): Record<string, ValidAction[]> {
    return buildActionsMap(this.mpState, this.ctx);
  }

  /**
   * Change player control type (human/ai).
   */
  setPlayerControl(playerId: string, type: 'human' | 'ai'): void {
    if (this.isDestroyed) {
      throw new Error('Room has been destroyed');
    }

    const session = this.players.get(playerId);
    if (!session) {
      throw new Error(`Player ${playerId} not found`);
    }

    const capabilities = buildBaseCapabilities(session.playerIndex, type);
    const result = updatePlayerControlPure(
      this.mpState,
      session.playerIndex,
      type,
      capabilities
    );

    if (!result.success) {
      throw new Error(result.error);
    }

    this.mpState = result.value;
    this.syncPlayersCache();
    this.notifyListeners();
  }

  // === SESSION MANAGEMENT ===

  /**
   * Update a player session.
   */
  joinPlayer(session: PlayerSession): Result<PlayerSession> {
    if (this.isDestroyed) {
      return err('Room has been destroyed');
    }

    const existing = this.players.get(session.playerId);
    if (!existing) {
      return err(`Player ${session.playerId} not found`);
    }

    const merged: PlayerSession = {
      ...existing,
      ...session,
      isConnected: true,
      capabilities: (session.capabilities ?? existing.capabilities).map(cap => ({ ...cap }))
    };

    const result = updatePlayerSession(this.mpState, existing.playerId, merged);
    if (!result.success) {
      return err(result.error);
    }

    this.mpState = result.value;
    this.syncPlayersCache();
    this.notifyListeners();

    return ok(merged);
  }

  /**
   * Disconnect a player session.
   */
  leavePlayer(playerId: string): Result<PlayerSession> {
    if (this.isDestroyed) {
      return err('Room has been destroyed');
    }

    const existing = this.players.get(playerId);
    if (!existing) {
      return err(`Player ${playerId} not found`);
    }

    const updated: PlayerSession = {
      ...existing,
      isConnected: false
    };

    const result = updatePlayerSession(this.mpState, playerId, updated);
    if (!result.success) {
      return err(result.error);
    }

    this.mpState = result.value;
    this.syncPlayersCache();
    this.notifyListeners();

    return ok(updated);
  }

  /**
   * Get all player sessions.
   */
  getPlayers(): readonly PlayerSession[] {
    return this.mpState.players;
  }

  /**
   * Get a specific player session by ID.
   */
  getPlayer(playerId: string): PlayerSession | undefined {
    return this.players.get(playerId);
  }

  /**
   * Get game ID.
   */
  getGameId(): string {
    return this.gameId;
  }

  /**
   * Destroy the room and all resources.
   */
  destroy(): void {
    this.connectedClients.clear();
    this.clientToPlayer.clear();
    this.isDestroyed = true;
  }

  // === PRIVATE HELPERS ===

  /**
   * Rebuild players Map from mpState.players.
   */
  private syncPlayersCache(): void {
    this.players.clear();
    for (const player of this.mpState.players) {
      this.players.set(player.playerId, player);
    }
  }

  /**
   * Notify all connected clients with updated views.
   */
  private notifyListeners(): void {
    for (const clientId of this.connectedClients) {
      const playerId = this.clientToPlayer.get(clientId);
      const view = this.getView(playerId);
      this.send(clientId, { type: 'STATE_UPDATE', view });
    }
  }

  // === MESSAGE HANDLERS ===

  private handleExecuteAction(clientId: string, message: { type: 'EXECUTE_ACTION'; action: GameAction }): void {
    const playerId = this.clientToPlayer.get(clientId);
    if (!playerId) {
      this.send(clientId, { type: 'ERROR', error: 'Not associated with a player. Send JOIN first.' });
      return;
    }

    const result = this.executeAction(playerId, message.action);
    if (!result.success) {
      this.send(clientId, { type: 'ERROR', error: result.error || 'Action execution failed' });
    }
    // Success broadcast happens via notifyListeners() in executeAction
  }

  private handleJoin(clientId: string, message: { type: 'JOIN'; playerIndex: number; name: string }): void {
    const playerId = `player-${message.playerIndex}`;

    // Validate player index
    if (message.playerIndex < 0 || message.playerIndex > 3) {
      this.send(clientId, { type: 'ERROR', error: 'Invalid player index. Must be 0-3.' });
      return;
    }

    // Associate client with player
    this.clientToPlayer.set(clientId, playerId);

    // Update player session
    const existing = this.players.get(playerId);
    if (existing) {
      const result = this.joinPlayer({
        ...existing,
        isConnected: true
      });
      if (!result.success) {
        this.send(clientId, { type: 'ERROR', error: result.error });
        return;
      }
    }

    // Send filtered view for this player
    const view = this.getView(playerId);
    this.send(clientId, { type: 'STATE_UPDATE', view });
  }

  private handleSetControl(clientId: string, message: { type: 'SET_CONTROL'; playerIndex: number; controlType: 'human' | 'ai' }): void {
    const playerId = `player-${message.playerIndex}`;

    // Validate player index
    if (message.playerIndex < 0 || message.playerIndex > 3) {
      this.send(clientId, { type: 'ERROR', error: 'Invalid player index. Must be 0-3.' });
      return;
    }

    try {
      this.setPlayerControl(playerId, message.controlType);
      // Success broadcast happens via notifyListeners() in setPlayerControl
    } catch (e) {
      this.send(clientId, { type: 'ERROR', error: e instanceof Error ? e.message : 'Unknown error' });
    }
  }
}
