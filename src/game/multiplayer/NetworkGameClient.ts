/**
 * NetworkGameClient - Protocol-based GameClient implementation.
 *
 * This is a "dumb" client that speaks protocol only.
 * NO game engine imports allowed - only protocol types!
 *
 * Design principles:
 * - Implements GameClient interface for compatibility
 * - Translates GameClient methods to protocol messages
 * - Caches GameView for fast local reads
 * - Works with any Connection (in-process, Worker, WebSocket)
 * - Trusts server completely - no local validation or recomputation
 */

import type { GameClient } from './GameClient';
import type { ActionRequest, PlayerSession, Result } from './types';
import { ok, err } from './types';
import type {
  GameView,
  GameConfig,
  ServerMessage,
  ValidAction
} from '../../shared/multiplayer/protocol';
import type { Connection } from '../../server/transports/Transport';

/**
 * NetworkGameClient - Protocol-speaking GameClient
 */
export class NetworkGameClient implements GameClient {
  private connection: Connection;
  private gameId?: string;
  private cachedView?: GameView;
  private viewsByPerspective = new Map<string, GameView>();
  private listeners = new Set<(view: GameView) => void>();
  private initPromise?: Promise<void>;
  private playerId: string = 'player-0'; // Default to player-0
  private sessionId: string;
  private pendingSubscriptionPlayerId?: string;
  private createGameResolver?: (value: { gameId: string; view: GameView }) => void;
  private createGameRejecter?: (error: Error) => void;
  private viewWaiters: Array<(view: GameView) => void> = [];
  private actionWaiters: Array<(result: Result<GameView>) => void> = [];
  private joinResolvers = new Map<string, (result: Result<PlayerSession>) => void>();

  constructor(connection: Connection, config?: GameConfig) {
    this.connection = connection;
    this.sessionId = `client-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Subscribe to server messages
    this.connection.onMessage((message) => {
      this.handleServerMessage(message);
    });

    // If config provided, create game immediately
    if (config) {
      this.initPromise = this.createGame(config);
    }
  }

  /**
   * Join or reconnect a player session.
   */
  async joinGame(session: PlayerSession): Promise<Result<PlayerSession>> {
    if (this.initPromise) {
      await this.initPromise;
    }

    const gameId = this.gameId;

    if (!gameId) {
      return err('Game not initialized');
    }

    return new Promise<Result<PlayerSession>>((resolve) => {
      this.joinResolvers.set(session.playerId, resolve);
      void (async () => {
        try {
          await this.connection.send({
            type: 'JOIN_GAME',
            gameId,
            session,
            clientId: this.sessionId
          });
        } catch (error) {
          this.joinResolvers.delete(session.playerId);
          resolve(err(error instanceof Error ? error.message : String(error)));
        }
      })();
    });
  }

  /**
   * Disconnect a player session.
   */
  async leaveGame(playerId: string): Promise<void> {
    if (this.initPromise) {
      await this.initPromise;
    }

    const gameId = this.gameId;

    if (!gameId) {
      return;
    }

    await this.connection.send({
      type: 'LEAVE_GAME',
      gameId,
      playerId,
      clientId: this.sessionId
    });
  }

  /**
   * Get current GameView for a specific perspective.
   */
  async getView(perspective: string = this.playerId): Promise<GameView> {
    if (this.initPromise) {
      await this.initPromise;
    }

    const view = this.viewsByPerspective.get(perspective) ?? this.cachedView;
    if (view) {
      return view;
    }

    return new Promise<GameView>((resolve) => {
      this.viewWaiters.push(resolve);
    });
  }

  /**
   * Execute an action for a player session.
   */
  async executeAction(request: ActionRequest): Promise<Result<GameView>> {
    if (this.initPromise) {
      await this.initPromise;
    }

    const gameId = this.gameId;

    if (!gameId) {
      return err('Game not initialized');
    }

    return new Promise<Result<GameView>>((resolve) => {
      this.actionWaiters.push(resolve);
      void (async () => {
        try {
          await this.connection.send({
            type: 'EXECUTE_ACTION',
            gameId,
            playerId: request.playerId,
            action: request.action
          });
        } catch (error) {
          this.removeActionWaiter(resolve);
          resolve(err(error instanceof Error ? error.message : String(error)));
        }
      })();
    });
  }

  /**
   * Subscribe to game view changes
   */
  subscribe(listener: (view: GameView) => void): () => void {
    // Send current view immediately if we have it
    if (this.cachedView) {
      listener(this.cachedView);
    }

    // Add to listeners
    this.listeners.add(listener);

    // Return unsubscribe function
    return () => {
      this.listeners.delete(listener);
    };
  }

  /**
   * Change player control type
   */
  async setPlayerControl(playerId: number, type: 'human' | 'ai'): Promise<void> {
    // Wait for initialization
    if (this.initPromise) {
      await this.initPromise;
    }

    if (!this.gameId) {
      throw new Error('Game not initialized');
    }

    // Send SET_PLAYER_CONTROL message
    await this.connection.send({
      type: 'SET_PLAYER_CONTROL',
      gameId: this.gameId,
      playerId,
      controlType: type
    });
  }

  /**
   * Destroy client and clean up
   */
  destroy(): void {
    // Clear listeners
    this.listeners.clear();

    // Disconnect
    this.connection.disconnect();
  }

  /**
   * Create a new game - returns promise that resolves when GAME_CREATED received
   */
  private async createGame(config: GameConfig): Promise<void> {
    try {
      // Create promise that will be resolved in handleServerMessage
      const gameCreatedPromise = new Promise<{ gameId: string; view: GameView }>((resolve, reject) => {
        this.createGameResolver = resolve;
        this.createGameRejecter = reject;

        // Timeout after 5 seconds
        setTimeout(() => {
          if (this.createGameResolver) {
            this.createGameRejecter?.(new Error('Game creation timeout'));
            delete this.createGameResolver;
            delete this.createGameRejecter;
          }
        }, 5000);
      });

      // Send CREATE_GAME message
      await this.connection.send({
        type: 'CREATE_GAME',
        config,
        clientId: this.sessionId
      });

      // Wait for GAME_CREATED response (promise resolved in handleServerMessage)
      await gameCreatedPromise;
    } catch (error) {
      console.error('Failed to create game:', error);
      throw error;
    }
  }


  /**
   * Handle server messages
   */
  private handleServerMessage(message: ServerMessage): void {
    switch (message.type) {
      case 'GAME_CREATED': {
        this.gameId = message.gameId;
        this.cacheViewForPerspective(this.playerId, message.view);
        this.cachedView = this.cloneView(message.view);

        // Resolve createGame promise if waiting
        if (this.createGameResolver) {
          this.createGameResolver({
            gameId: message.gameId,
            view: message.view
          });
          delete this.createGameResolver;
          delete this.createGameRejecter;
        }

        this.resolveViewWaiters();
        this.resolveActionWaiters(ok(message.view));
        this.resolveJoinResolvers();

        this.notifyListeners();
        if (this.pendingSubscriptionPlayerId !== undefined) {
          void this.sendSubscription(this.pendingSubscriptionPlayerId);
          delete this.pendingSubscriptionPlayerId;
        }
        break;
      }

      case 'STATE_UPDATE': {
        if (message.gameId !== this.gameId) {
          return;
        }

        const perspective = message.perspective ?? this.playerId;
        this.cacheViewForPerspective(perspective, message.view);
        if (perspective === this.playerId) {
          this.cachedView = this.cloneView(message.view);
        }

        this.notifyListeners();
        this.resolveViewWaiters();
        this.resolveActionWaiters(ok(message.view));
        this.resolveJoinResolvers();
        break;
      }

      case 'ERROR': {
        console.error('Server error:', message.error);
        if (message.requestType === 'EXECUTE_ACTION') {
          this.resolveActionWaiters(err(message.error));
        }
        if (message.requestType === 'JOIN_GAME') {
          this.resolveJoinResolversWithError(message.error);
        }
        break;
      }

      case 'PLAYER_STATUS':
        if (message.gameId === this.gameId && this.cachedView) {
          // Update player info in cached view
          const player = this.cachedView.players.find(p => p.playerId === message.playerId);
          if (player) {
            if (message.controlType !== undefined) {
              player.controlType = message.controlType;
            }
            if (message.capabilities) {
              player.capabilities = message.capabilities.map(cap => ({ ...cap }));
            }
          }
          this.notifyListeners();
        }
        break;

      case 'PROGRESS':
        // Could expose progress events if needed
        console.log('Progress:', message);
        break;
    }
  }

  /**
   * Notify all listeners of view change
   */
  private notifyListeners(): void {
    if (!this.cachedView) return;

    for (const listener of this.listeners) {
      listener(this.cachedView);
    }
  }

  private cacheViewForPerspective(perspective: string, view: GameView): void {
    const clone = this.cloneView(view);
    this.viewsByPerspective.set(perspective, clone);
  }

  private cloneView(view: GameView): GameView {
    return {
      state: { ...view.state, players: view.state.players.map(p => ({ ...p, hand: [...p.hand] })) },
      validActions: view.validActions.map(valid => {
        const actionClone = { ...valid.action };
        if ('meta' in actionClone && actionClone.meta) {
          actionClone.meta = JSON.parse(JSON.stringify(actionClone.meta));
        }
        return {
          ...valid,
          action: actionClone
        };
      }),
      transitions: view.transitions.map(transition => {
        const actionClone = { ...transition.action };
        if ('meta' in actionClone && actionClone.meta) {
          actionClone.meta = JSON.parse(JSON.stringify(actionClone.meta));
        }
        return {
          ...transition,
          action: actionClone
        };
      }),
      players: view.players.map(player => {
        const { capabilities, ...rest } = player;
        return {
          ...rest,
          ...(capabilities ? { capabilities: capabilities.map(cap => ({ ...cap })) } : {})
        };
      }),
      metadata: { ...view.metadata }
    };
  }

  private resolveViewWaiters(): void {
    if (!this.cachedView) return;
    const waiters = [...this.viewWaiters];
    this.viewWaiters = [];
    for (const waiter of waiters) {
      waiter(this.cachedView);
    }
  }

  private resolveActionWaiters(result: Result<GameView>): void {
    if (this.actionWaiters.length === 0) return;
    const waiters = [...this.actionWaiters];
    this.actionWaiters = [];
    for (const waiter of waiters) {
      waiter(result);
    }
  }

  private resolveJoinResolvers(): void {
    if (!this.cachedView) return;
    for (const [playerId, resolver] of [...this.joinResolvers.entries()]) {
      const playerInfo = this.cachedView.players.find(p => p.sessionId === playerId);
      if (playerInfo) {
        const session: PlayerSession = {
          playerId: playerInfo.sessionId ?? `player-${playerInfo.playerId}`,
          playerIndex: playerInfo.playerId as 0 | 1 | 2 | 3,
          controlType: playerInfo.controlType,
          isConnected: playerInfo.connected,
          name: playerInfo.name ?? `Player ${playerInfo.playerId + 1}`,
          capabilities: playerInfo.capabilities?.map(cap => ({ ...cap })) ?? []
        };
        resolver(ok(session));
        this.joinResolvers.delete(playerId);
      }
    }
  }

  private resolveJoinResolversWithError(error: string): void {
    if (this.joinResolvers.size === 0) return;
    for (const resolver of this.joinResolvers.values()) {
      resolver(err(error));
    }
    this.joinResolvers.clear();
  }

  private removeActionWaiter(waiter: (result: Result<GameView>) => void): void {
    const index = this.actionWaiters.indexOf(waiter);
    if (index >= 0) {
      this.actionWaiters.splice(index, 1);
    }
  }

  /**
   * Set the current player ID (for UI perspective)
   */
  async setPlayerId(playerIdOrIndex: number | string): Promise<void> {
    if (typeof playerIdOrIndex === 'string') {
      this.playerId = playerIdOrIndex;
    } else {
      // Legacy: accept number and convert to playerId
      this.playerId = `player-${playerIdOrIndex}`;
    }

    const targetPlayerId = this.playerId;
    this.pendingSubscriptionPlayerId = targetPlayerId;

    const existingView = this.viewsByPerspective.get(targetPlayerId);
    if (existingView) {
      this.cachedView = existingView;
    }

    if (this.gameId) {
      await this.sendSubscription(targetPlayerId);
      delete this.pendingSubscriptionPlayerId;
    }
  }

  private async sendSubscription(playerId?: string): Promise<void> {
    if (!this.gameId) return;
    try {
      const message: {
        type: 'SUBSCRIBE';
        gameId: string;
        clientId: string;
        playerId?: string;
      } = {
        type: 'SUBSCRIBE',
        gameId: this.gameId,
        clientId: this.sessionId
      };

      if (playerId !== undefined) {
        message.playerId = playerId;
      }

      await this.connection.send(message);
    } catch (error) {
      console.error('Failed to update subscription:', error);
    }
  }

  /**
   * Expose cached GameView for debugging/testing (non-interface).
   */
  getCachedView(perspective: string = this.playerId): GameView | undefined {
    return this.viewsByPerspective.get(perspective) ?? this.cachedView;
  }

  /**
   * Convenience: get cached actions for a perspective (cloned).
   * NOTE: Only returns actions for the current cached view's perspective.
   */
  getCachedActions(playerId: string = this.playerId): ValidAction[] {
    const view = this.viewsByPerspective.get(playerId) ?? this.cachedView;
    if (!view) return [];

    // Only return actions if this matches the view's perspective
    const viewPlayer = view.players.find(p => p.sessionId === playerId);
    if (!viewPlayer) return [];

    return view.validActions.map(action => ({
      ...action,
      action: { ...action.action }
    }));
  }

  /**
   * Convenience: get cached actions map (cloned).
   * Returns actions per player based on cached views.
   */
  getCachedActionsMap(): Record<string, ValidAction[]> {
    const result: Record<string, ValidAction[]> = {};

    // Build map from all cached views
    for (const [perspective, view] of this.viewsByPerspective.entries()) {
      result[perspective] = view.validActions.map(action => ({
        ...action,
        action: { ...action.action }
      }));
    }

    // Add current view if not already included
    if (this.cachedView && !result[this.playerId]) {
      result[this.playerId] = this.cachedView.validActions.map(action => ({
        ...action,
        action: { ...action.action }
      }));
    }

    return result;
  }
}
