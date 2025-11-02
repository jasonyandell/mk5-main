/**
 * GameServer - Main game orchestrator.
 *
 * Responsibilities:
 * 1. Create and manage GameKernel (pure game logic)
 * 2. Create and manage AIManager (AI lifecycle)
 * 3. Route protocol messages to appropriate handlers
 * 4. Broadcast state updates to all subscribers
 * 5. Manage player connections and sessions
 *
 * NOT responsible for:
 * - Transport implementation (that's Transport's job)
 * - Pure game logic (that's GameKernel's job)
 * - AI strategy/actions (that's AIClient's job)
 *
 * Design:
 * - Owns GameKernel (creates in constructor)
 * - Owns AIManager (creates in constructor)
 * - Gets Transport reference via setTransport() after construction
 * - Processes all protocol messages
 * - Broadcasts updates to all subscribers
 *
 * Clean Hierarchy:
 * GameServer
 *   → GameKernel (pure logic)
 *   → AIManager (AI lifecycle)
 *   → Transport (message routing)
 */

import type { GameConfig } from '../game/types/config';
import type { ClientMessage, ServerMessage, IGameAdapter } from '../shared/multiplayer/protocol';
import type { PlayerSession } from '../game/multiplayer/types';
import { GameKernel, type KernelUpdate } from '../kernel/GameKernel';
import { AIManager } from './ai/AIManager';
import type { Transport } from './transports/Transport';

interface ClientSubscription {
  clientId: string;
  playerId?: string;
  perspective?: string;
}

/**
 * GameServer - Main orchestrator for a game instance
 */
export class GameServer {
  private kernel: GameKernel | null = null;
  private aiManager: AIManager | null = null;
  private transport: Transport | null = null;
  private gameId: string;
  private subscriptions: Map<string, ClientSubscription> = new Map();
  private subscribers: Map<string, (update: KernelUpdate) => void> = new Map();

  /**
   * Create a new GameServer instance.
   *
   * This creates the kernel and spawns AI clients as needed.
   * Does NOT connect to transport yet (use setTransport() after construction).
   */
  constructor(gameId: string, config: GameConfig, adapter: IGameAdapter, playerSessions: PlayerSession[]) {
    this.gameId = gameId;

    // Create GameKernel with pure game logic
    this.kernel = new GameKernel(gameId, config, playerSessions);

    // Create AIManager with adapter
    this.aiManager = new AIManager(adapter, 'beginner');

    // Spawn AI clients based on player sessions
    for (const session of playerSessions) {
      if (session.controlType === 'ai') {
        this.aiManager.spawnAI(session.playerIndex, gameId, session.playerId);
      }
    }
  }

  /**
   * Set the transport for this server.
   * Called after construction to establish message routing.
   */
  setTransport(transport: Transport): void {
    this.transport = transport;
  }

  /**
   * Get game ID
   */
  getGameId(): string {
    return this.gameId;
  }

  /**
   * Get current kernel (for testing)
   */
  getKernel(): GameKernel | null {
    return this.kernel;
  }

  /**
   * Handle a protocol message from a client.
   * Routes to appropriate handler based on message type.
   */
  handleMessage(clientId: string, message: ClientMessage): void {
    if (!this.kernel) {
      console.error('GameServer: kernel not initialized');
      this.sendError(clientId, 'Game not initialized', 'CREATE_GAME');
      return;
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

        default:
          const unknownType = (message as any).type;
          console.error(`GameServer: Unknown message type: ${unknownType}`);
          this.sendError(clientId, `Unknown message type: ${unknownType}`, 'CREATE_GAME');
      }
    } catch (error) {
      console.error('GameServer: Error processing message:', error);
      this.sendError(clientId, `Error: ${error instanceof Error ? error.message : 'Unknown error'}`, message.type);
    }
  }

  /**
   * Handle client disconnect.
   * Called by transport when client disconnects.
   */
  handleClientDisconnect(clientId: string): void {
    // Remove subscriptions for this client
    const sub = this.subscriptions.get(clientId);
    if (sub && sub.playerId) {
      if (this.kernel) {
        const result = this.kernel.leavePlayer(sub.playerId);
        if (!result.success) {
          console.error(`Failed to leave player ${sub.playerId}:`, result.error);
        }
      }
    }

    this.subscriptions.delete(clientId);
    this.subscribers.delete(clientId);
  }

  /**
   * Destroy the server and all resources.
   */
  destroy(): void {
    if (this.aiManager) {
      this.aiManager.destroyAll();
      this.aiManager = null;
    }

    this.kernel = null;
    this.subscriptions.clear();
    this.subscribers.clear();
    this.transport = null;
  }

  /**
   * Private: Handle CREATE_GAME message
   */
  private handleCreateGame(clientId: string, _message: ClientMessage): void {
    // Game is already created in constructor
    // Just send initial state to client
    if (this.kernel) {
      const view = this.kernel.getView();
      const state = this.kernel.getState();
      const actions = this.kernel.getActionsMap();
      this.sendMessage(clientId, {
        type: 'GAME_CREATED',
        gameId: this.gameId,
        view,
        state,
        actions
      });
    }
  }

  /**
   * Private: Handle EXECUTE_ACTION message
   */
  private handleExecuteAction(clientId: string, message: ClientMessage): void {
    if (!this.kernel) return;

    if (message.type !== 'EXECUTE_ACTION') return;

    const { playerId, action, timestamp } = message;
    if (!playerId || !action) {
      this.sendError(clientId, 'Missing playerId or action', 'EXECUTE_ACTION');
      return;
    }

    const result = this.kernel.executeAction(playerId, action, timestamp ?? Date.now());
    if (!result.success) {
      this.sendError(clientId, result.error || 'Action execution failed', 'EXECUTE_ACTION');
      return;
    }

    // Broadcast state update to all subscribers
    this.broadcastToAllClients();
  }

  /**
   * Private: Handle SET_PLAYER_CONTROL message
   */
  private handleSetPlayerControl(clientId: string, message: ClientMessage): void {
    if (!this.kernel || !this.aiManager) return;

    if (message.type !== 'SET_PLAYER_CONTROL') return;

    const { playerId: playerIndex, controlType } = message;
    if (playerIndex === undefined || !controlType) {
      this.sendError(clientId, 'Missing playerIndex or controlType', 'SET_PLAYER_CONTROL');
      return;
    }

    try {
      // Convert playerIndex to playerId for kernel API
      const playerId = playerIndex < 2 ? `human-${playerIndex}` : `ai-${playerIndex}`;
      this.kernel.setPlayerControl(playerId, controlType);

      // Manage AI lifecycle
      if (controlType === 'ai') {
        const aiPlayerId = `ai-${playerIndex}`;
        this.aiManager.spawnAI(playerIndex, this.gameId, aiPlayerId);
      } else {
        this.aiManager.destroyAI(playerIndex);
      }

      // Broadcast state update to all subscribers
      this.broadcastToAllClients();
    } catch (error) {
      this.sendError(clientId, `Failed to set player control: ${error instanceof Error ? error.message : 'Unknown error'}`, 'SET_PLAYER_CONTROL');
    }
  }

  /**
   * Private: Handle JOIN_GAME message
   */
  private handleJoinGame(clientId: string, message: ClientMessage): void {
    if (!this.kernel) return;

    if (message.type !== 'JOIN_GAME') return;

    const { session } = message;
    if (!session) {
      this.sendError(clientId, 'Missing session', 'JOIN_GAME');
      return;
    }

    const result = this.kernel.joinPlayer(session);
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

    // Send state update
    this.sendMessage(clientId, {
      type: 'PLAYER_STATUS',
      gameId: this.gameId,
      playerId: session.playerIndex,
      sessionId: session.playerId,
      status: 'joined',
      controlType: session.controlType,
      capabilities: session.capabilities
    });

    // Broadcast state update to all subscribers
    this.broadcastToAllClients();
  }

  /**
   * Private: Handle LEAVE_GAME message
   */
  private handleLeaveGame(clientId: string, message: ClientMessage): void {
    if (!this.kernel) return;

    if (message.type !== 'LEAVE_GAME') return;

    const { playerId } = message;
    if (!playerId) {
      this.sendError(clientId, 'Missing playerId', 'LEAVE_GAME');
      return;
    }

    const player = this.kernel.getPlayer(playerId);
    if (!player) {
      this.sendError(clientId, `Player ${playerId} not found`, 'LEAVE_GAME');
      return;
    }

    const result = this.kernel.leavePlayer(playerId);
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
      gameId: this.gameId,
      playerId: player.playerIndex,
      sessionId: playerId,
      status: 'left'
    });

    // Broadcast state update to all subscribers
    this.broadcastToAllClients();
  }

  /**
   * Private: Handle SUBSCRIBE message
   */
  private handleSubscribe(clientId: string, message: ClientMessage): void {
    if (!this.kernel) return;

    if (message.type !== 'SUBSCRIBE') return;

    const { playerId } = message;

    // Store subscription
    const sub: ClientSubscription = {
      clientId
    };
    if (playerId) {
      sub.playerId = playerId;
      sub.perspective = playerId;
    }
    this.subscriptions.set(clientId, sub);

    // Create subscriber that receives kernel updates
    const subscriber = (_update: KernelUpdate) => {
      const view = this.kernel!.getView(playerId);
      const state = this.kernel!.getState();
      const actions = this.kernel!.getActionsMap();
      const msg: ServerMessage = {
        type: 'STATE_UPDATE',
        gameId: this.gameId,
        view,
        state,
        actions
      };
      if (playerId) {
        (msg as any).perspective = playerId;
      }
      this.sendMessage(clientId, msg);
    };

    this.subscribers.set(clientId, subscriber);

    // Send initial state
    const view = this.kernel.getView(playerId);
    const state = this.kernel.getState();
    const actions = this.kernel.getActionsMap();
    const initialMsg: ServerMessage = {
      type: 'STATE_UPDATE',
      gameId: this.gameId,
      view,
      state,
      actions
    };
    if (playerId) {
      (initialMsg as any).perspective = playerId;
    }
    this.sendMessage(clientId, initialMsg);
  }

  /**
   * Private: Handle UNSUBSCRIBE message
   */
  private handleUnsubscribe(clientId: string, message: ClientMessage): void {
    if (message.type !== 'UNSUBSCRIBE') return;

    const { playerId } = message;

    // Remove subscription
    const sub = this.subscriptions.get(clientId);
    if (sub && sub.playerId === playerId) {
      this.subscriptions.delete(clientId);
      this.subscribers.delete(clientId);
    }
  }

  /**
   * Private: Broadcast current state to all subscribed clients
   */
  private broadcastToAllClients(): void {
    if (!this.kernel) return;

    for (const [clientId, subscriber] of this.subscribers) {
      const sub = this.subscriptions.get(clientId);
      if (sub) {
        const view = this.kernel.getView(sub.perspective);
        const state = this.kernel.getState();
        const actions = this.kernel.getActionsMap();

        const msg: ServerMessage = {
          type: 'STATE_UPDATE',
          gameId: this.gameId,
          view,
          state,
          actions
        };

        if (sub.perspective) {
          (msg as any).perspective = sub.perspective;
        }

        subscriber(msg);
      }
    }
  }

  /**
   * Private: Send a message to a client
   */
  private sendMessage(clientId: string, message: ServerMessage): void {
    if (!this.transport) {
      console.error('sendMessage called before transport was set');
      return;
    }
    this.transport.send(clientId, message);
  }

  /**
   * Private: Send an error message to a client
   */
  private sendError(clientId: string, error: string, requestType: ClientMessage['type']): void {
    this.sendMessage(clientId, {
      type: 'ERROR',
      gameId: this.gameId,
      error,
      requestType
    });
  }
}
