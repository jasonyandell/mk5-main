/**
 * NetworkGameClient - Protocol-based GameClient implementation.
 *
 * This is a "dumb" client that speaks protocol only.
 * NO game engine imports allowed - only protocol types!
 *
 * Design principles:
 * - Implements GameClient interface for compatibility
 * - Translates GameClient methods to protocol messages
 * - Caches GameView and state for fast local reads
 * - Works with any IGameAdapter (in-process, Worker, WebSocket)
 */

import type { GameClient } from './GameClient';
import type { ActionRequest, MultiplayerGameState, PlayerSession, Result } from './types';
import { ok, err } from './types';
import type {
  GameView,
  GameConfig,
  ServerMessage,
  ValidAction
} from '../../shared/multiplayer/protocol';
import type { IGameAdapter } from '../../server/adapters/IGameAdapter';

/**
 * NetworkGameClient - Protocol-speaking GameClient
 */
export class NetworkGameClient implements GameClient {
  private adapter: IGameAdapter;
  private gameId?: string;
  private cachedState?: MultiplayerGameState;
  private cachedView?: GameView;
  private viewsByPerspective = new Map<string, GameView>();
  private cachedActions = new Map<string, ValidAction[]>();
  private listeners = new Set<(state: MultiplayerGameState) => void>();
  private unsubscribe: (() => void) | undefined;
  private initPromise?: Promise<void>;
  private playerId: string = 'player-0'; // Default to player-0
  private sessionId: string;
  private pendingSubscriptionPlayerId?: string;
  private createGameResolver?: (value: { gameId: string; view: GameView; state: MultiplayerGameState }) => void;
  private createGameRejecter?: (error: Error) => void;
  private stateWaiters: Array<(state: MultiplayerGameState) => void> = [];
  private actionWaiters: Array<(result: Result<MultiplayerGameState>) => void> = [];
  private joinResolvers = new Map<string, (result: Result<PlayerSession>) => void>();

  constructor(adapter: IGameAdapter, config?: GameConfig) {
    this.adapter = adapter;
    this.sessionId = `client-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;

    // Subscribe to server messages
    this.unsubscribe = adapter.subscribe((message) => {
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
          await this.adapter.send({
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

    await this.adapter.send({
      type: 'LEAVE_GAME',
      gameId,
      playerId,
      clientId: this.sessionId
    });
  }

  /**
   * Get valid actions for a session.
   */
  async getActions(playerId: string): Promise<ValidAction[]> {
    if (this.initPromise) {
      await this.initPromise;
    }

    if (!this.cachedState) {
      await this.getState();
    }

    return this.getCachedActions(playerId);
  }

  /**
   * Get current multiplayer state.
   */
  async getState(): Promise<MultiplayerGameState> {
    if (this.initPromise) {
      await this.initPromise;
    }

    if (this.cachedState) {
      return this.cachedState;
    }

    return new Promise<MultiplayerGameState>((resolve) => {
      this.stateWaiters.push(resolve);
    });
  }

  /**
   * Execute an action for a player session.
   */
  async executeAction(request: ActionRequest): Promise<Result<MultiplayerGameState>> {
    if (this.initPromise) {
      await this.initPromise;
    }

    const gameId = this.gameId;

    if (!gameId) {
      return err('Game not initialized');
    }

    return new Promise<Result<MultiplayerGameState>>((resolve) => {
      this.actionWaiters.push(resolve);
      void (async () => {
        try {
          await this.adapter.send({
            type: 'EXECUTE_ACTION',
            gameId,
            playerId: request.playerId,
            action: request.action,
            timestamp: request.timestamp ?? Date.now()
          });
        } catch (error) {
          this.removeActionWaiter(resolve);
          resolve(err(error instanceof Error ? error.message : String(error)));
        }
      })();
    });
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener: (state: MultiplayerGameState) => void): () => void {
    // Send current state immediately if we have it
    if (this.cachedState) {
      listener(this.cachedState);
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
    await this.adapter.send({
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
    // Unsubscribe from adapter
    if (this.unsubscribe) {
      this.unsubscribe();
      this.unsubscribe = undefined;
    }

    // Clear listeners
    this.listeners.clear();

    // Destroy adapter
    this.adapter.destroy();
  }

  /**
   * Create a new game - returns promise that resolves when GAME_CREATED received
   */
  private async createGame(config: GameConfig): Promise<void> {
    try {
      // Create promise that will be resolved in handleServerMessage
      const gameCreatedPromise = new Promise<{ gameId: string; view: GameView; state: MultiplayerGameState }>((resolve, reject) => {
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
      await this.adapter.send({
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
        this.cacheState(message.state, message.actions);
        this.cacheViewForPerspective(this.playerId, message.view);
        this.cachedView = this.cloneView(message.view);

        // Resolve createGame promise if waiting
        if (this.createGameResolver) {
          this.createGameResolver({
            gameId: message.gameId,
            view: message.view,
            state: message.state
          });
          delete this.createGameResolver;
          delete this.createGameRejecter;
        }

        this.resolveStateWaiters();
        this.resolveActionWaiters(ok(this.cachedState ?? message.state));
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

        this.cacheState(message.state, message.actions);
        const perspective = message.perspective ?? this.playerId;
        this.cacheViewForPerspective(perspective, message.view);
        if (perspective === this.playerId) {
          this.cachedView = this.cloneView(message.view);
        }

        this.notifyListeners();
        this.resolveStateWaiters();
        this.resolveActionWaiters(ok(this.cachedState ?? message.state));
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
   * Notify all listeners of state change
   */
  private notifyListeners(): void {
    if (!this.cachedState) return;

    for (const listener of this.listeners) {
      listener(this.cachedState);
    }
  }

  private cacheState(
    state: MultiplayerGameState,
    actions: Record<string, ValidAction[]>
  ): void {
    this.cachedState = state;
    this.cachedActions = new Map(
      Object.entries(actions ?? {}).map(([playerId, actionList]) => [
        playerId,
        actionList.map(valid => ({
          ...valid,
          action: { ...valid.action }
        }))
      ])
    );
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

  private resolveStateWaiters(): void {
    if (!this.cachedState) return;
    const waiters = [...this.stateWaiters];
    this.stateWaiters = [];
    for (const waiter of waiters) {
      waiter(this.cachedState);
    }
  }

  private resolveActionWaiters(result: Result<MultiplayerGameState>): void {
    if (this.actionWaiters.length === 0) return;
    const waiters = [...this.actionWaiters];
    this.actionWaiters = [];
    for (const waiter of waiters) {
      waiter(result);
    }
  }

  private resolveJoinResolvers(): void {
    if (!this.cachedState) return;
    for (const [playerId, resolver] of [...this.joinResolvers.entries()]) {
      const session = this.cachedState.players.find(p => p.playerId === playerId);
      if (session) {
        resolver(ok({
          ...session,
          capabilities: session.capabilities.map(cap => ({ ...cap }))
        }));
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

  private removeActionWaiter(waiter: (result: Result<MultiplayerGameState>) => void): void {
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

      await this.adapter.send(message);
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
   */
  getCachedActions(playerId: string = this.playerId): ValidAction[] {
    const actions = this.cachedActions.get(playerId);
    if (!actions) return [];
    return actions.map(action => ({
      ...action,
      action: { ...action.action }
    }));
  }

  /**
   * Convenience: get cached actions map (cloned).
   */
  getCachedActionsMap(): Record<string, ValidAction[]> {
    const result: Record<string, ValidAction[]> = {};
    for (const [playerId, actions] of this.cachedActions.entries()) {
      result[playerId] = actions.map(action => ({
        ...action,
        action: { ...action.action }
      }));
    }
    return result;
  }
}
