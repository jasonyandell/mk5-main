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
  private playerIndex: number = 0; // Default to index 0
  private sessionId: string;
  private pendingSubscriptionPlayerId?: string;

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
  getState(): MultiplayerGameState & { state: FilteredGameState } {
    if (!this.cachedView) {
      // Return empty state if not initialized
      return {
        state: this.createEmptyState(),
        sessions: []
      };
    }

    return this.viewToMultiplayerState(this.cachedView);
  }

  /**
   * Request action execution
   */
  async requestAction(_playerId: number, action: GameAction): Promise<Result<void>> {
    // Wait for initialization if needed
    if (this.initPromise) {
      await this.initPromise;
    }

    if (!this.gameId) {
      return err('Game not initialized');
    }

    // Send EXECUTE_ACTION message using our playerId
    try {
      await this.adapter.send({
        type: 'EXECUTE_ACTION',
        gameId: this.gameId,
        playerId: this.playerId, // Use our playerId, not the parameter
        action
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
   * Create a new game
   */
  private async createGame(config: GameConfig): Promise<void> {
    try {
      await this.adapter.send({
        type: 'CREATE_GAME',
        config,
        clientId: this.sessionId
      });

      // Wait for GAME_CREATED response (handled in handleServerMessage)
      // This is a simplified approach - production would use promises
      await new Promise<void>((resolve) => {
        const checkInterval = window.setInterval(() => {
          if (this.gameId) {
            window.clearInterval(checkInterval);
            resolve();
          }
        }, 10);

        // Timeout after 5 seconds
        window.setTimeout(() => {
          window.clearInterval(checkInterval);
          resolve();
        }, 5000);
      });
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
        this.notifyListeners();
        if (this.pendingSubscriptionPlayerId) {
          void this.sendSubscription(this.pendingSubscriptionPlayerId);
          this.pendingSubscriptionPlayerId = undefined;
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
  private viewToMultiplayerState(view: GameView): MultiplayerGameState & { state: FilteredGameState } {
    // Convert player info to sessions
    const sessions: PlayerSession[] = view.players.map(player => ({
      playerId: player.sessionId || `player-${player.playerId}`,
      playerIndex: player.playerId as 0 | 1 | 2 | 3,
      controlType: player.controlType,
      capabilities: (player.capabilities ?? []).map(cap => ({ ...cap }))
    }));

    return {
      state: view.state,  // view.state is FilteredGameState
      sessions
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
      // Try to extract index from playerId like "player-0", "ai-1"
      const match = playerIdOrIndex.match(/-(\d+)$/);
      if (match && match[1]) {
        this.playerIndex = parseInt(match[1], 10);
      } else {
        this.playerIndex = 0;
      }
    } else {
      // Legacy: accept number and convert to playerId
      this.playerIndex = playerIdOrIndex;
      this.playerId = `player-${playerIdOrIndex}`;
    }

    const targetPlayerId = this.playerId;
    this.pendingSubscriptionPlayerId = targetPlayerId;

    if (this.gameId) {
      await this.sendSubscription(targetPlayerId);
      this.pendingSubscriptionPlayerId = undefined;
    }
  }

  /**
   * Get valid actions for current player
   * (Extracted from cached view, not computed)
   */
  getValidActions(playerIndex?: number): GameAction[] {
    if (!this.cachedView) return [];

    const targetPlayer = playerIndex ?? this.playerIndex;

    // Filter valid actions for this player
    return this.cachedView.validActions
      .filter(va => {
        const action = va.action;
        // Neutral actions available to all
        if (!('player' in action)) return true;
        // Player-specific actions
        return action.player === targetPlayer;
      })
      .map(va => va.action);
  }

  private async sendSubscription(playerId?: string): Promise<void> {
    if (!this.gameId) return;
    try {
      await this.adapter.send({
        type: 'SUBSCRIBE',
        gameId: this.gameId,
        clientId: this.sessionId,
        playerId
      });
    } catch (error) {
      console.error('Failed to update subscription:', error);
    }
  }
}
