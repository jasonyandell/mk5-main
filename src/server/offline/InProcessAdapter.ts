/**
 * InProcessAdapter - Direct method call adapter for local games.
 *
 * This adapter runs GameHost in the same process and translates
 * protocol messages to direct method calls. Used for single-player
 * and hot-seat multiplayer.
 *
 * Responsibilities:
 * - Creates and manages GameHost instances
 * - Spawns AIClient instances for AI players
 * - Translates protocol to method calls
 * - Manages subscriptions
 */

import type { IGameAdapter } from '../adapters/IGameAdapter';
import type { GameConfig } from '../../game/types/config';
import type {
  ClientMessage,
  ServerMessage
} from '../../shared/multiplayer/protocol';
import { GameRegistry } from '../game/GameHost';
import { AIClient } from '../../game/multiplayer/AIClient';
import type { GameInstance } from '../game/GameHost';

/**
 * In-process game session
 */
interface GameSession {
  instance: GameInstance;
  aiClients: Map<number, AIClient>;
  subscribers: Set<() => void>; // Changed to just store cleanup functions
  aiHandlers: Map<number, (message: ServerMessage) => void>; // Track AI client handlers by playerId
  viewSubscriptions: Map<string, { unsubscribe: () => void }>; // clientId -> unsubscribe
}

/**
 * InProcessAdapter - Runs game in same process
 */
export class InProcessAdapter implements IGameAdapter {
  private sessions = new Map<string, GameSession>();
  private registry = new GameRegistry();
  private currentGameId?: string;
  private messageHandlers = new Set<(message: ServerMessage) => void>();
  private destroyed = false;

  /**
   * Send a message to be processed
   */
  async send(message: ClientMessage): Promise<void> {
    if (this.destroyed) {
      throw new Error('Adapter is destroyed');
    }

    try {
      await this.handleMessage(message);
    } catch (error) {
      // Send error message to subscribers
      this.broadcast({
        type: 'ERROR',
        gameId: this.currentGameId || '',
        error: error instanceof Error ? error.message : String(error),
        requestType: message.type
      });
    }
  }

  /**
   * Subscribe to server messages
   */
  subscribe(handler: (message: ServerMessage) => void): () => void {
    this.messageHandlers.add(handler);

    return () => {
      this.messageHandlers.delete(handler);
    };
  }

  /**
   * Check if connected (always true for in-process)
   */
  isConnected(): boolean {
    return !this.destroyed;
  }

  /**
   * Get adapter metadata
   */
  getMetadata() {
    return {
      type: 'in-process' as const,
      latency: 0,
      ...(this.currentGameId ? { gameId: this.currentGameId } : {})
    };
  }

  /**
   * Destroy adapter and clean up
   */
  destroy(): void {
    if (this.destroyed) return;

    // Destroy all game sessions
    for (const session of this.sessions.values()) {
      // Clean up view subscriptions
      for (const { unsubscribe } of session.viewSubscriptions.values()) {
        try {
          unsubscribe();
        } catch (error) {
          console.error('Error unsubscribing view:', error);
        }
      }
      session.viewSubscriptions.clear();

      for (const cleanup of session.subscribers) {
        try {
          cleanup();
        } catch (error) {
          console.error('Error during subscriber cleanup:', error);
        }
      }
      session.subscribers.clear();

      // Kill AI clients
      for (const aiClient of session.aiClients.values()) {
        aiClient.destroy();
      }
      session.aiClients.clear();

      // Destroy game host
      session.instance.host.destroy();
    }

    this.sessions.clear();
    this.messageHandlers.clear();
    this.destroyed = true;
  }

  /**
   * Private: Handle incoming message
   */
  private async handleMessage(message: ClientMessage): Promise<void> {
    switch (message.type) {
      case 'CREATE_GAME':
        await this.handleCreateGame(message.config, message.clientId);
        break;

      case 'EXECUTE_ACTION':
        await this.handleExecuteAction(
          message.gameId,
          message.playerId,
          message.action
        );
        break;

      case 'SET_PLAYER_CONTROL':
        await this.handleSetPlayerControl(
          message.gameId,
          message.playerId,
          message.controlType
        );
        break;

      case 'SUBSCRIBE':
        this.handleSubscribe(message.gameId, message.clientId, message.playerId);
        break;

      case 'UNSUBSCRIBE':
        this.handleUnsubscribe(message.gameId, message.clientId);
        break;

      default:
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        throw new Error(`Unhandled message type: ${(message as any).type}`);
    }
  }

  /**
   * Private: Handle CREATE_GAME
   */
  private async handleCreateGame(config: GameConfig, clientId: string): Promise<void> {
    // Check if we need to find a competitive seed for one-hand mode
    if (config.variant?.type === 'one-hand' && !config.shuffleSeed) {
      config = await this.findCompetitiveSeed(config);
    }

    // Create game instance (GameRegistry creates players internally now)
    const instance = this.registry.createGame(config);
    this.currentGameId = instance.id;

    // Create session
    const session: GameSession = {
      instance,
      aiClients: new Map(),
      subscribers: new Set(),
      aiHandlers: new Map(),
      viewSubscriptions: new Map()
    };

    this.sessions.set(instance.id, session);

    // Subscribe main client to game updates (player-0's view by default)
    this.subscribeClientToView(session, clientId, 'player-0');

    // Spawn AI clients for AI players
    const players = instance.host.getPlayers();
    for (const player of players) {
      if (player.controlType === 'ai') {
        await this.spawnAIClient(instance.id, player.playerId, player.playerIndex);
      }
    }

    // Send GAME_CREATED message with filtered view for main client
    this.broadcast({
      type: 'GAME_CREATED',
      gameId: instance.id,
      view: instance.host.getView('player-0')
    });
  }

  /**
   * Private: Handle EXECUTE_ACTION
   */
  private async handleExecuteAction(
    gameId: string,
    playerId: string,
    // eslint-disable-next-line @typescript-eslint/no-explicit-any
    action: any
  ): Promise<void> {
    const session = this.sessions.get(gameId);
    if (!session) {
      throw new Error(`Game not found: ${gameId}`);
    }

    const result = session.instance.host.executeAction(playerId, action);

    if (!result.ok) {
      throw new Error(result.error);
    }

    // State update is automatically broadcast via subscription
  }

  /**
   * Private: Handle SET_PLAYER_CONTROL
   */
  private async handleSetPlayerControl(
    gameId: string,
    playerIndex: number,
    controlType: 'human' | 'ai',
    _sessionId?: string
  ): Promise<void> {
    const session = this.sessions.get(gameId);
    if (!session) {
      throw new Error(`Game not found: ${gameId}`);
    }

    // Update control type in host
    session.instance.host.setPlayerControl(playerIndex, controlType);

    // Get player by index
    const players = session.instance.host.getPlayers();
    const player = players.find(p => p.playerIndex === playerIndex);
    if (!player) {
      throw new Error(`Player ${playerIndex} not found`);
    }

    // Manage AI client
    if (controlType === 'ai') {
      // Spawn AI if not exists
      if (!session.aiClients.has(playerIndex)) {
        await this.spawnAIClient(gameId, player.playerId, playerIndex);
      }
    } else {
      // Kill AI if exists
      const aiClient = session.aiClients.get(playerIndex);
      if (aiClient) {
        aiClient.destroy();
        session.aiClients.delete(playerIndex);
      }
    }

    // Send status update
    this.broadcast({
      type: 'PLAYER_STATUS',
      gameId,
      playerId: playerIndex,
      status: 'control_changed',
      controlType,
      capabilities: player.capabilities.map(cap => ({ ...cap }))
    });
  }

  /**
   * Private: Spawn AI client for a player
   */
  private async spawnAIClient(gameId: string, playerId: string, playerIndex: number): Promise<void> {
    const session = this.sessions.get(gameId);
    if (!session) return;

    // Create AI client with adapter pointing back to us
    const aiClient = new AIClient(
      gameId,
      playerIndex,
      this, // AI uses same adapter
      playerId, // Pass playerId for identification
      'beginner' // TODO: Get from config
    );

    // Create a dedicated handler for this AI client
    const aiHandler = (message: ServerMessage) => {
      // Only send STATE_UPDATE messages for this specific AI's game
      if (message.type === 'STATE_UPDATE' && message.gameId === gameId) {
        // The AI client will handle this message
        aiClient['handleServerMessage'](message);
      }
    };

    // Store the handler so we can send messages to this specific AI
    session.aiHandlers.set(playerIndex, aiHandler);

    // Subscribe the AI client to game updates with their playerId
    const unsubscribe = session.instance.host.subscribe(playerId, (view) => {
      // Send STATE_UPDATE directly to this AI client's handler
      const handler = session.aiHandlers.get(playerIndex);
      if (handler) {
        handler({
          type: 'STATE_UPDATE',
          gameId,
          view
        });
      }
    });

    // Store the unsubscribe function
    session.subscribers.add(() => {
      unsubscribe();
      session.aiHandlers.delete(playerIndex);
    });

    // Store AI client
    session.aiClients.set(playerIndex, aiClient);

    // Start AI
    aiClient.start();

    const player = session.instance.host.getPlayers().find(p => p.playerIndex === playerIndex);
    const capabilities = player ? player.capabilities.map(cap => ({ ...cap })) : [];

    // Send status update
    this.broadcast({
      type: 'PLAYER_STATUS',
      gameId,
      playerId: playerIndex,
      status: 'joined',
      controlType: 'ai',
      capabilities
    });
  }

  /**
   * Private: Subscribe a client to a particular player perspective
   */
  private subscribeClientToView(
    session: GameSession,
    clientId: string,
    playerId?: string
  ): void {
    const existing = session.viewSubscriptions.get(clientId);
    if (existing) {
      existing.unsubscribe();
      session.viewSubscriptions.delete(clientId);
    }

    const unsubscribe = session.instance.host.subscribe(playerId, (view) => {
      this.broadcastToSession(session, {
        type: 'STATE_UPDATE',
        gameId: session.instance.id,
        view
      });
    });

    session.viewSubscriptions.set(clientId, { unsubscribe });
    session.subscribers.add(() => {
      if (session.viewSubscriptions.get(clientId)?.unsubscribe === unsubscribe) {
        session.viewSubscriptions.delete(clientId);
      }
      unsubscribe();
    });
  }

  private handleSubscribe(gameId: string, clientId: string, playerId?: string): void {
    const session = this.sessions.get(gameId);
    if (!session) {
      throw new Error(`Game not found: ${gameId}`);
    }

    this.subscribeClientToView(session, clientId, playerId);
  }

  private handleUnsubscribe(gameId: string, clientId: string): void {
    const session = this.sessions.get(gameId);
    if (!session) {
      return;
    }

    const existing = session.viewSubscriptions.get(clientId);
    if (existing) {
      existing.unsubscribe();
      session.viewSubscriptions.delete(clientId);
    }
  }

  /**
   * Private: Find competitive seed for one-hand mode
   */
  private async findCompetitiveSeed(config: GameConfig): Promise<GameConfig> {
    // Send progress updates while searching
    let attempts = 0;
    const maxAttempts = 100;

    const sendProgress = (progress: number) => {
      this.broadcast({
        type: 'PROGRESS',
        gameId: 'pending',
        operation: 'seed_finding',
        progress,
        message: `Searching for competitive hand... (${attempts}/${maxAttempts})`
      });
    };

    // Import seed finder
    const { findCompetitiveSeed } = await import('../../game/core/seedFinder');

    // Search for seed with progress updates
    const seed = await findCompetitiveSeed({
      targetWinRate: 0.5,
      tolerance: 0.1,
      maxAttempts,
      onProgress: (current) => {
        attempts = current;
        sendProgress(Math.round((current / maxAttempts) * 100));
      }
    });

    // Update config with found seed
    return {
      ...config,
      shuffleSeed: seed,
      variant: {
        ...config.variant!,
        config: {
          ...config.variant?.config,
          originalSeed: seed
        }
      }
    };
  }

  /**
   * Private: Broadcast message to all handlers
   */
  private broadcast(message: ServerMessage): void {
    for (const handler of this.messageHandlers) {
      handler(message);
    }
  }

  /**
   * Private: Broadcast to specific game session
   */
  private broadcastToSession(_session: GameSession, message: ServerMessage): void {
    // Broadcast to main client
    this.broadcast(message);

    // Note: AI clients also receive via their subscription
  }
}
