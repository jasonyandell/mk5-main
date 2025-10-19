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
import type { GameConfig, GameVariant } from '../../game/types/config';
import type {
  GameView,
  ValidAction,
  PlayerInfo
} from '../../shared/multiplayer/protocol';
import { authorizeAndExecute } from '../../game/multiplayer/authorization';
import type { MultiplayerGameState, PlayerSession, Capability } from '../../game/multiplayer/types';
import { filterActionsForSession, getVisibleStateForSession, resolveSessionForAction } from '../../game/multiplayer/capabilityUtils';
import { createInitialState } from '../../game/core/state';
import { getValidActions, getNextStates } from '../../game/core/gameEngine';
import { applyVariants } from '../../game/variants/registry';
import type { VariantConfig, StateMachine } from '../../game/variants/types';

/**
 * GameHost - Pure game server authority
 */
export class GameHost {
  private mpState: MultiplayerGameState;
  private gameId: string;
  private variant: GameVariant | undefined;
  private variantConfigs: VariantConfig[];
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

    const normalizedPlayers = players.map(player => ({
      ...player,
      capabilities: player.capabilities ?? this.buildBaseCapabilities(player.playerIndex, player.controlType)
    }));

    // Store players by playerId
    this.players = new Map(normalizedPlayers.map(p => [p.playerId, p]));

    // Compose variants into single getValidActions function
    const baseGetValidActions = getValidActions;

    this.variantConfigs = [
      ...(config.variant
        ? [{ type: config.variant.type, ...(config.variant.config ? { config: config.variant.config } : {}) }]
        : []),
      ...(config.variants ?? [])
    ];

    // Compose variants
    this.getValidActionsComposed = applyVariants(baseGetValidActions, this.variantConfigs);

    // Create initial game state
    const initialState = createInitialState({
      playerTypes: config.playerTypes,
      ...(config.shuffleSeed !== undefined ? { shuffleSeed: config.shuffleSeed } : {}),
      ...(config.theme !== undefined ? { theme: config.theme } : {}),
      ...(config.colorOverrides !== undefined ? { colorOverrides: config.colorOverrides } : {}),
      variants: this.variantConfigs
    });

    // Store pure GameState (NO filtering - filtering happens in createView())
    const now = Date.now();
    this.mpState = {
      gameId,
      coreState: initialState,
      players: normalizedPlayers,
      createdAt: now,
      lastActionAt: now,
      enabledVariants: this.variantConfigs
    };

    // Process any immediate scripted actions emitted at startup
    this.processAutoExecuteActions();
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

    const result = authorizeAndExecute(this.mpState, request, this.getValidActionsComposed);

    if (!result.ok) {
      return { ok: false, error: result.error };
    }

    // Update state
    this.mpState = result.value;
    this.lastUpdate = Date.now();

    const autoExecuted = this.processAutoExecuteActions();

    // Notify all listeners once after processing scripted actions
    if (autoExecuted) {
      this.lastUpdate = Date.now();
    }
    this.notifyListeners();

    return { ok: true };
  }

  /**
   * Change player control type
   */
  setPlayerControl(playerIndex: number, type: 'human' | 'ai'): void {
    // Find player by index
    const updatedPlayers = this.mpState.players.map(session => {
      if (session.playerIndex !== playerIndex) {
        return session;
      }

      return {
        ...session,
        controlType: type,
        capabilities: this.buildBaseCapabilities(session.playerIndex, type)
      };
    });

    // Update players map
    for (const session of updatedPlayers) {
      this.players.set(session.playerId, session);
    }

    // Update core state playerTypes to match
    const updatedCoreState = {
      ...this.mpState.coreState,
      playerTypes: updatedPlayers.map(s => s.controlType) as ('human' | 'ai')[]
    };

    this.mpState = {
      ...this.mpState,
      coreState: updatedCoreState,
      players: updatedPlayers
    };

    this.lastUpdate = Date.now();
    this.notifyListeners();
  }

  /**
   * Get player sessions
   */
  getPlayers(): readonly PlayerSession[] {
    return this.mpState.players;
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
    // TODO: Add replay-url capability support
    // When forPlayerId session has 'replay-url' capability:
    //   1. Call encodeGameUrl(state.initialConfig.seed, state.actionHistory)
    //   2. Add compressed URL to GameView.replayUrl
    //   3. This enables URL-based replay without sending full actionHistory
    // Reference: src/game/core/url-compression.ts

    const { coreState, players } = this.mpState;

    // Get all valid actions using composed function
    const allValidActions = this.getValidActionsComposed(coreState);
    const session = forPlayerId ? this.players.get(forPlayerId) : undefined;

    // Always convert to FilteredGameState format (with handCount)
    // If session provided, filter based on capabilities; otherwise show all
    const visibleState: import('../../game/types').FilteredGameState = session
      ? getVisibleStateForSession(coreState, session)
      : this.convertToFilteredState(coreState);

    const sessionActions = session
      ? filterActionsForSession(session, allValidActions)
      : allValidActions;

    // Get transitions for labels
    const transitions = getNextStates(coreState);

    // Convert to ValidAction format with labels
    let validActions: ValidAction[] = sessionActions.map(action => {
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
        recommended: this.isRecommendedAction(action, coreState)
      } as ValidAction;
    });

    if (forPlayerId && !session) {
      validActions = [];
    }

    // Create player info
    const playerInfoList: PlayerInfo[] = Array.from(players).map(playerSession => ({
      playerId: playerSession.playerIndex, // Still uses numeric ID for protocol compatibility
      controlType: playerSession.controlType,
      sessionId: playerSession.playerId, // Use string playerId in sessionId field for now
      connected: playerSession.isConnected ?? true,
      name: playerSession.name || `Player ${playerSession.playerIndex + 1}`,
      capabilities: playerSession.capabilities.map(cap => ({ ...cap }))
    }));

    return {
      state: visibleState,
      validActions,
      players: playerInfoList,
      metadata: {
        gameId: this.gameId,
        ...(this.variant ? { variant: this.variant } : {}),
        ...(this.variantConfigs.length ? { variants: this.variantConfigs } : {}),
        created: this.created,
        lastUpdate: this.lastUpdate
      }
    };
  }

  /**
   * Convert pure GameState to FilteredGameState format (adds handCount to players)
   */
  private convertToFilteredState(state: GameState): import('../../game/types').FilteredGameState {
    const filteredPlayers = state.players.map(player => ({
      id: player.id,
      name: player.name,
      teamId: player.teamId,
      marks: player.marks,
      hand: player.hand,
      handCount: player.hand.length,
      ...(player.suitAnalysis ? { suitAnalysis: player.suitAnalysis } : {})
    }));

    return {
      ...state,
      players: filteredPlayers
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
   * Private: Base capability set per control type.
   */
  private buildBaseCapabilities(playerIndex: number, controlType: 'human' | 'ai'): Capability[] {
    const capabilities: Capability[] = [
      { type: 'act-as-player', playerIndex },
      { type: 'observe-own-hand' }
    ];

    if (controlType === 'ai') {
      capabilities.push({ type: 'replace-ai' });
    }

    return capabilities;
  }

  /**
   * Private: Process scripted auto-execute actions until exhausted.
   */
  private processAutoExecuteActions(): boolean {
    const MAX_AUTO_EXEC = 50;
    let iterations = 0;
    let executed = false;

    while (iterations < MAX_AUTO_EXEC) {
      const actions = this.getValidActionsComposed(this.mpState.coreState);
      const autoAction = actions.find(a => a.autoExecute === true);

      if (!autoAction) {
        break;
      }

      const session = resolveSessionForAction(Array.from(this.mpState.players), autoAction);

      if (!session) {
        console.error('Auto-execute failed: no capable session', {
          gameId: this.gameId,
          action: autoAction
        });
        break;
      }

      const result = authorizeAndExecute(
        this.mpState,
        { playerId: session.playerId, action: autoAction },
        this.getValidActionsComposed
      );

      if (!result.ok) {
        console.error('Auto-execute failed', {
          gameId: this.gameId,
          action: autoAction,
          error: result.error
        });
        break;
      }

      this.mpState = result.value;
      this.lastUpdate = Date.now();
      iterations += 1;
      executed = true;
    }

    if (iterations === MAX_AUTO_EXEC) {
      console.error('Auto-execute limit reached', {
        gameId: this.gameId
      });
    }

    return executed;
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

