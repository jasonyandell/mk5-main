/**
 * GameHost - Pure game authority server.
 *
 * This is the ONLY place where game state mutations happen.
 * NO AI code here - AI is external via AIClient.
 *
 * Design principles:
 * - Pure state management (no transport concerns)
 * - Authorization via existing authorizeAndExecute
 * - Variant support via composition
 * - Observable via subscriptions
 */

import type { GameState, GameAction } from '../../game/types';
import type {
  GameConfig,
  GameView,
  ValidAction,
  PlayerInfo,
  GameVariant
} from '../../shared/multiplayer/protocol';
import { authorizeAndExecute, canPlayerExecuteAction } from '../../game/multiplayer/authorization';
import type { MultiplayerGameState, PlayerSession } from '../../game/multiplayer/types';
import { createInitialState } from '../../game/core/state';
import { getValidActions } from '../../game/core/gameEngine';
import { getNextStates } from '../../game/core/gameEngine';
import { applyVariants } from '../../game/variants/registry';
import type { VariantConfig, StateMachine } from '../../game/variants/types';

/**
 * Unique game instance
 */
export interface GameInstance {
  id: string;
  host: GameHost;
  created: number;
  lastUpdate: number;
}

/**
 * GameHost - Pure game server authority
 */
export class GameHost {
  private mpState: MultiplayerGameState;
  private gameId: string;
  private variant: GameVariant | undefined;
  private created: number;
  private lastUpdate: number;
  private listeners: Map<string, (view: GameView) => void> = new Map(); // playerId -> listener
  private players: Map<string, PlayerSession>; // playerId -> PlayerSession
  private getValidActionsComposed: StateMachine;

  constructor(gameId: string, config: GameConfig, players: PlayerSession[]) {
    this.gameId = gameId;
    this.variant = config.variant;
    this.created = Date.now();
    this.lastUpdate = this.created;

    // Validate players
    if (players.length !== 4) {
      throw new Error(`Expected 4 players, got ${players.length}`);
    }

    // Ensure player indices are 0-3
    const indices = players.map(p => p.playerIndex).sort();
    if (indices.join(',') !== '0,1,2,3') {
      throw new Error('Player indices must be 0, 1, 2, 3');
    }

    // Store players by playerId
    this.players = new Map(players.map(p => [p.playerId, p]));

    // Compose variants into single getValidActions function
    const baseGetValidActions = getValidActions;

    // Convert old variant format to new if needed
    const variantConfigs: VariantConfig[] = config.variant
      ? [{ type: config.variant.type, ...(config.variant.config ? { config: config.variant.config } : {}) }]
      : [];

    // Compose variants
    this.getValidActionsComposed = applyVariants(baseGetValidActions, variantConfigs);

    // Create initial game state
    const initialState = createInitialState({
      playerTypes: config.playerTypes,
      ...(config.shuffleSeed !== undefined ? { shuffleSeed: config.shuffleSeed } : {})
    });

    this.mpState = {
      state: initialState,
      sessions: players
    };
  }

  /**
   * Get current game view for clients
   * @param forPlayerId - Optional player ID to filter actions for
   */
  getView(forPlayerId?: string): GameView {
    return this.createView(forPlayerId);
  }

  /**
   * Execute an action with authorization
   */
  executeAction(playerId: string, action: GameAction): { ok: boolean; error?: string } {
    const request = {
      playerId,
      action
    };

    const result = authorizeAndExecute(this.mpState, request);

    if (!result.ok) {
      return { ok: false, error: result.error };
    }

    // Update state
    this.mpState = result.value;
    this.lastUpdate = Date.now();

    // Notify all listeners
    this.notifyListeners();

    return { ok: true };
  }

  /**
   * Change player control type
   */
  setPlayerControl(playerIndex: number, type: 'human' | 'ai'): void {
    // Find player by index
    const sessions = this.mpState.sessions.map(session =>
      session.playerIndex === playerIndex
        ? { ...session, controlType: type }
        : session
    );

    // Update players map
    for (const session of sessions) {
      this.players.set(session.playerId, session);
    }

    // Update state playerTypes to match
    const newState = {
      ...this.mpState.state,
      playerTypes: sessions.map(s => s.controlType) as ('human' | 'ai')[]
    };

    this.mpState = {
      state: newState,
      sessions
    };

    this.lastUpdate = Date.now();
    this.notifyListeners();
  }

  /**
   * Get player sessions
   */
  getPlayers(): PlayerSession[] {
    return this.mpState.sessions;
  }

  /**
   * Get player by playerId
   */
  getPlayer(playerId: string): PlayerSession | undefined {
    return this.players.get(playerId);
  }

  /**
   * Subscribe to game updates
   * @param playerId - Player ID to filter views for (or undefined for unfiltered)
   */
  subscribe(playerId: string | undefined, listener: (view: GameView) => void): () => void {
    // Use a unique key for undefined player IDs
    const listenerKey = playerId || `unfiltered-${Date.now()}-${Math.random()}`;

    // Send current state immediately
    listener(this.createView(playerId));

    // Add to listeners
    this.listeners.set(listenerKey, listener);

    // Return unsubscribe function
    return () => {
      this.listeners.delete(listenerKey);
    };
  }

  /**
   * Destroy the game host
   */
  destroy(): void {
    this.listeners.clear();
  }

  /**
   * Private: Create view for clients
   * @param forPlayerId - Optional player ID to filter actions for
   */
  private createView(forPlayerId?: string): GameView {
    const { state, sessions } = this.mpState;

    // Get all valid actions using composed function
    const allValidActions = this.getValidActionsComposed(state);

    // Get transitions for labels
    const transitions = getNextStates(state);

    // Convert to ValidAction format with labels
    let validActions: ValidAction[] = allValidActions.map(action => {
      // Find matching transition for label
      const transition = transitions.find(t => {
        if (t.action.type !== action.type) return false;

        // Match player if present
        if ('player' in t.action && 'player' in action) {
          if (t.action.player !== action.player) return false;
        }

        // Match other fields based on action type
        switch (action.type) {
          case 'bid':
            return 'bid' in t.action &&
                   t.action.bid === action.bid &&
                   t.action.value === action.value;
          case 'select-trump':
            return 'trump' in t.action &&
                   JSON.stringify(t.action.trump) === JSON.stringify(action.trump);
          case 'play':
            return 'dominoId' in t.action && 'dominoId' in action &&
                   t.action.dominoId === action.dominoId;
          default:
            return true;
        }
      });

      return {
        action,
        label: transition?.label || action.type,
        ...(this.getActionGroup(action) ? { group: this.getActionGroup(action) } : {}),
        ...(this.getActionShortcut(action) ? { shortcut: this.getActionShortcut(action) } : {}),
        recommended: this.isRecommendedAction(action, state)
      } as ValidAction;
    });

    // Filter actions by player if provided
    if (forPlayerId) {
      const player = this.players.get(forPlayerId);
      if (player) {
        // Use existing authorization function to filter by playerIndex
        validActions = validActions.filter(va =>
          canPlayerExecuteAction(player.playerIndex, va.action, state)
        );
      } else {
        // Unknown player = no actions
        validActions = [];
      }
    }

    // Create player info
    const players: PlayerInfo[] = sessions.map(session => ({
      playerId: session.playerIndex, // Still uses numeric ID for protocol compatibility
      controlType: session.controlType,
      sessionId: session.playerId, // Use string playerId in sessionId field for now
      connected: session.isConnected ?? true,
      name: session.name || `Player ${session.playerIndex + 1}`
    }));

    return {
      state,
      validActions,
      players,
      metadata: {
        gameId: this.gameId,
        ...(this.variant ? { variant: this.variant } : {}),
        created: this.created,
        lastUpdate: this.lastUpdate
      }
    };
  }

  /**
   * Private: Get UI group for action
   */
  private getActionGroup(action: GameAction): string | undefined {
    switch (action.type) {
      case 'bid':
        return 'Bidding';
      case 'play':
        return 'Play Domino';
      case 'select-trump':
        return 'Trump Selection';
      case 'pass':
        return 'Pass';
      default:
        return undefined;
    }
  }

  /**
   * Private: Get keyboard shortcut for action
   */
  private getActionShortcut(_action: GameAction): string | undefined {
    // Future: Add keyboard shortcuts
    return undefined;
  }

  /**
   * Private: Check if action is recommended
   */
  private isRecommendedAction(action: GameAction, _state: GameState): boolean {
    // For consensus actions, recommend immediate agreement
    if (action.type === 'agree-score-hand') {
      return true;
    }

    // Future: Add AI hints for recommended moves
    return false;
  }

  /**
   * Private: Notify all listeners
   */
  private notifyListeners(): void {
    // Send each listener a view filtered for their player
    for (const [playerKey, listener] of this.listeners) {
      // Check if this is a player ID or an unfiltered key
      const playerId = playerKey.startsWith('unfiltered-') ? undefined : playerKey;
      const view = this.createView(playerId);
      listener(view);
    }
  }
}

/**
 * Game registry for managing multiple games
 */
export class GameRegistry {
  private games = new Map<string, GameInstance>();

  /**
   * Create a new game
   */
  createGame(config: GameConfig): GameInstance {
    const gameId = this.generateGameId();

    // Create player sessions before GameHost
    const players: PlayerSession[] = config.playerTypes.map((type, i) => ({
      playerId: `${type === 'human' ? 'player' : 'ai'}-${i}`,
      playerIndex: i as 0 | 1 | 2 | 3,
      controlType: type,
      isConnected: true,
      name: `Player ${i + 1}`
    }));

    const host = new GameHost(gameId, config, players);

    const instance: GameInstance = {
      id: gameId,
      host,
      created: Date.now(),
      lastUpdate: Date.now()
    };

    this.games.set(gameId, instance);
    return instance;
  }

  /**
   * Get game by ID
   */
  getGame(gameId: string): GameInstance | undefined {
    return this.games.get(gameId);
  }

  /**
   * Remove game
   */
  removeGame(gameId: string): void {
    const game = this.games.get(gameId);
    if (game) {
      game.host.destroy();
      this.games.delete(gameId);
    }
  }

  /**
   * Generate unique game ID
   */
  private generateGameId(): string {
    return `game-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
  }
}