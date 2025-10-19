/**
 * NetworkGameClient - Protocol-based GameClient implementation.
 *
 * This is a "dumb" client that speaks protocol only.
 * NO game engine imports allowed - only protocol types!
 *
 * Design principles:
 * - Implements GameClient interface for compatibility
 * - Translates GameClient methods to protocol messages
 * - Caches GameView for synchronous getState()
 * - Works with any IGameAdapter (in-process, Worker, WebSocket)
 */

import type { GameClient } from './GameClient';
import type { GameAction, FilteredGameState } from '../types';
import type { MultiplayerGameState, PlayerSession, Result } from './types';
import { ok, err } from './types';
import type {
  GameView,
  GameConfig,
  ServerMessage
} from '../../shared/multiplayer/protocol';
import type { IGameAdapter } from '../../server/adapters/IGameAdapter';

/**
 * NetworkGameClient - Protocol-speaking GameClient
 */
export class NetworkGameClient implements GameClient {
  private adapter: IGameAdapter;
  private gameId?: string;
  private cachedView?: GameView;
  private listeners = new Set<(state: MultiplayerGameState) => void>();
  private unsubscribe: (() => void) | undefined;
  private initPromise?: Promise<void>;
  private playerId: string = 'player-0'; // Default to player-0
  private sessionId: string;
  private pendingSubscriptionPlayerId?: string;
  private createGameResolver?: (value: { gameId: string; view: GameView }) => void;
  private createGameRejecter?: (error: Error) => void;

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
   * Get current state (synchronous, from cache)
   */
  getState(): MultiplayerGameState {
    if (!this.cachedView) {
      // Return empty state if not initialized
      const emptyState = this.createEmptyState();
      return {
        gameId: 'initializing',
        coreState: emptyState,
        players: [],
        createdAt: Date.now(),
        lastActionAt: Date.now(),
        enabledVariants: []
      };
    }

    return this.viewToMultiplayerState(this.cachedView);
  }

  /**
   * Request action execution
   */
  async requestAction(playerId: string, action: GameAction): Promise<Result<void>> {
    // Wait for initialization if needed
    if (this.initPromise) {
      await this.initPromise;
    }

    if (!this.gameId) {
      return err('Game not initialized');
    }

    // Send EXECUTE_ACTION message
    try {
      await this.adapter.send({
        type: 'EXECUTE_ACTION',
        gameId: this.gameId,
        playerId: playerId, // Use the provided playerId
        action,
        timestamp: Date.now()
      });

      // Success is assumed if send doesn't throw
      // Actual state update comes via STATE_UPDATE message
      return ok(undefined);
    } catch (error) {
      return err(error instanceof Error ? error.message : String(error));
    }
  }

  /**
   * Subscribe to state changes
   */
  subscribe(listener: (state: MultiplayerGameState) => void): () => void {
    // Send current state immediately if we have it
    if (this.cachedView) {
      listener(this.viewToMultiplayerState(this.cachedView));
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
      case 'GAME_CREATED':
        this.gameId = message.gameId;
        this.cachedView = message.view;

        // Resolve createGame promise if waiting
        if (this.createGameResolver) {
          this.createGameResolver({ gameId: message.gameId, view: message.view });
          delete this.createGameResolver;
          delete this.createGameRejecter;
        }

        this.notifyListeners();
        if (this.pendingSubscriptionPlayerId !== undefined) {
          void this.sendSubscription(this.pendingSubscriptionPlayerId);
          delete this.pendingSubscriptionPlayerId;
        }
        break;

      case 'STATE_UPDATE':
        if (message.gameId === this.gameId) {
          this.cachedView = message.view;
          this.notifyListeners();
        }
        break;

      case 'ERROR':
        console.error('Server error:', message.error);
        break;

      case 'PLAYER_STATUS':
        // Update cached view if needed
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
    if (!this.cachedView) return;

    const mpState = this.viewToMultiplayerState(this.cachedView);

    for (const listener of this.listeners) {
      listener(mpState);
    }
  }

  /**
   * Convert GameView to MultiplayerGameState
   */
  private viewToMultiplayerState(view: GameView): MultiplayerGameState {
    // Convert player info to sessions
    const players: PlayerSession[] = view.players.map(player => ({
      playerId: player.sessionId || `player-${player.playerId}`,
      playerIndex: player.playerId as 0 | 1 | 2 | 3,
      controlType: player.controlType,
      capabilities: (player.capabilities ?? []).map(cap => ({ ...cap }))
    }));

    return {
      gameId: view.metadata.gameId,
      coreState: view.state,  // view.state is FilteredGameState, we treat it as coreState for now
      players: players,
      createdAt: view.metadata.created,
      lastActionAt: view.metadata.lastUpdate,
      enabledVariants: view.metadata.variants || []
    };
  }

  /**
   * Create empty game state (before initialization)
   */
  private createEmptyState(): FilteredGameState {
    // This is a minimal empty state that matches the FilteredGameState interface
    return {
      initialConfig: {
        playerTypes: ['human', 'ai', 'ai', 'ai'],
        shuffleSeed: 0,
        theme: 'business',
        colorOverrides: {}
      },
      theme: 'business',
      colorOverrides: {},
      phase: 'setup',
      players: [
        { id: 0, name: 'Player 1', hand: [], handCount: 0, teamId: 0, marks: 0 },
        { id: 1, name: 'Player 2', hand: [], handCount: 0, teamId: 1, marks: 0 },
        { id: 2, name: 'Player 3', hand: [], handCount: 0, teamId: 0, marks: 0 },
        { id: 3, name: 'Player 4', hand: [], handCount: 0, teamId: 1, marks: 0 }
      ],
      currentPlayer: 0,
      dealer: 0,
      bids: [],
      currentBid: { type: 'pass', player: 0 },
      winningBidder: -1,
      trump: { type: 'not-selected' },
      tricks: [],
      currentTrick: [],
      currentSuit: -1,
      teamScores: [0, 0],
      teamMarks: [0, 0],
      gameTarget: 250,
      shuffleSeed: 0,
      playerTypes: ['human', 'ai', 'ai', 'ai'],
      consensus: {
        completeTrick: new Set(),
        scoreHand: new Set()
      },
      actionHistory: []
    };
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

    if (this.gameId) {
      await this.sendSubscription(targetPlayerId);
      delete this.pendingSubscriptionPlayerId;
    }
  }

  /**
   * Get valid actions for current player
   * Server already filtered these based on our subscription playerId
   */
  getValidActions(): GameAction[] {
    if (!this.cachedView) return [];

    // Server already filtered actions for our playerId - trust the server
    return this.cachedView.validActions.map(va => va.action);
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
}
