/**
 * GameKernel - Pure game authority engine.
 *
 * This is the ONLY place where game state mutations happen.
 * NO transport, AI, or multiplayer concerns here - just pure game logic.
 *
 * Design principles:
 * - Pure state management (no transport concerns)
 * - Authorization via existing authorizeAndExecute
 * - Variant support via composition
 * - Observable via subscriptions
 * - Deployable anywhere (local, Worker, Cloudflare)
 */

import type { GameState, GameAction } from '../game/types';
import type { GameConfig, GameVariant } from '../game/types/config';
import type {
  GameView,
  ValidAction,
  PlayerInfo
} from '../shared/multiplayer/protocol';
import { authorizeAndExecute } from '../game/multiplayer/authorization';
import type { MultiplayerGameState, PlayerSession, Capability, Result } from '../game/multiplayer/types';
import { ok, err } from '../game/multiplayer/types';
import { filterActionsForSession, getVisibleStateForSession, resolveSessionForAction } from '../game/multiplayer/capabilityUtils';
import { humanCapabilities, aiCapabilities } from '../game/multiplayer/capabilities';
import { createInitialState, cloneGameState } from '../game/core/state';
import { getValidActions, getNextStates } from '../game/core/gameEngine';
import { applyVariants } from '../game/variants/registry';
import type { VariantConfig, StateMachine } from '../game/variants/types';
import { createMultiplayerGame, updatePlayerSession } from '../game/multiplayer/stateLifecycle';
import { composeRules, baseLayer, getLayersByNames } from '../game/layers';
import type { GameRules } from '../game/layers/types';

export interface KernelUpdate {
  view: GameView;
  state: MultiplayerGameState;
  actions: Record<string, ValidAction[]>;
  perspective?: string;
}

const UNFILTERED_KEY = '__unfiltered__';

/**
 * GameKernel - Pure game logic engine
 *
 * Responsibilities:
 * - Pure state management (no mutations, only replacements)
 * - Action authorization and execution
 * - View generation with capability-based filtering
 * - Subscription management for state updates
 * - Variant and layer composition
 *
 * NOT responsible for:
 * - Transport/networking
 * - AI client lifecycle
 * - Protocol message routing
 * - Client connections
 */
export class GameKernel {
  private mpState: MultiplayerGameState;
  private gameId: string;
  private variant: GameVariant | undefined;
  private variantConfigs: VariantConfig[];
  private created: number;
  private lastUpdate: number;
  private listeners: Map<string, { perspective?: string; listener: (update: KernelUpdate) => void }> = new Map();
  private players: Map<string, PlayerSession>; // playerId -> PlayerSession
  private getValidActionsComposed: StateMachine;
  private layers: import('../game/layers/types').GameLayer[];
  private rules: GameRules;

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

    // Compose layers into rules FIRST (needed for variant composition)
    const enabledLayerNames = config.enabledLayers ?? [];
    const enabledLayers = enabledLayerNames.length > 0
      ? getLayersByNames(enabledLayerNames)
      : [];

    // Store layers for passing to getValidActions
    this.layers = [baseLayer, ...enabledLayers];
    this.rules = composeRules(this.layers);

    // Compose variants with layers/rules threaded through
    // Create a wrapper that calls getValidActions with our layers/rules
    const baseWithLayers: StateMachine = (state: GameState) =>
      baseGetValidActions(state, this.layers, this.rules);

    this.getValidActionsComposed = applyVariants(baseWithLayers, this.variantConfigs);

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
    this.mpState = createMultiplayerGame({
      gameId,
      coreState: initialState,
      players: normalizedPlayers,
      enabledVariants: this.variantConfigs,
      enabledLayers: config.enabledLayers ?? [],
      createdAt: now,
      lastActionAt: now
    });
    this.players = new Map(this.mpState.players.map((p: PlayerSession) => [p.playerId, p]));

    // Process any immediate scripted actions emitted at startup
    this.processAutoExecuteActions();
  }

  private buildUpdate(forPlayerId?: string): KernelUpdate {
    const stateSnapshot = this.cloneMultiplayerState(this.mpState);
    const allValidActions = this.getValidActionsComposed(stateSnapshot.coreState);
    const transitions = getNextStates(stateSnapshot.coreState, this.layers, this.rules);
    const actionsByPlayer = this.buildActionsMap(
      stateSnapshot.players,
      allValidActions,
      transitions,
      stateSnapshot.coreState
    );

    const view = this.createView(forPlayerId, {
      state: stateSnapshot,
      allActions: allValidActions,
      transitions,
      actionsByPlayer
    });

    const update: KernelUpdate = {
      view,
      state: stateSnapshot,
      actions: actionsByPlayer
    };

    if (forPlayerId !== undefined) {
      update.perspective = forPlayerId;
    }

    return update;
  }

  private buildActionsMap(
    sessions: readonly PlayerSession[],
    allValidActions: GameAction[],
    transitions: ReturnType<typeof getNextStates>,
    coreState: GameState
  ): Record<string, ValidAction[]> {
    const map: Record<string, ValidAction[]> = {};
    for (const session of sessions) {
      map[session.playerId] = this.buildValidActionsForSession(
        session,
        allValidActions,
        transitions,
        coreState
      );
    }

    map[UNFILTERED_KEY] = this.buildValidActionsForSession(
      undefined,
      allValidActions,
      transitions,
      coreState
    );

    return map;
  }

  private buildValidActionsForSession(
    session: PlayerSession | undefined,
    allValidActions: GameAction[],
    transitions: ReturnType<typeof getNextStates>,
    coreState: GameState
  ): ValidAction[] {
    const availableActions = session
      ? filterActionsForSession(session, allValidActions)
      : allValidActions;

    return availableActions.map((action: GameAction) => {
      const transition = this.findMatchingTransition(action, transitions);
      const actionClone: GameAction = { ...action };
      const group = this.getActionGroup(action);
      const shortcut = this.getActionShortcut(action);
      const recommended = this.isRecommendedAction(action, coreState);

      if ('meta' in action && action.meta) {
        (actionClone as { meta?: unknown }).meta = JSON.parse(JSON.stringify(action.meta));
      }

      const validAction: ValidAction = {
        action: actionClone,
        label: transition?.label || action.type,
        recommended
      };

      if (group !== undefined) {
        validAction.group = group;
      }

      if (shortcut !== undefined) {
        validAction.shortcut = shortcut;
      }

      return validAction;
    });
  }

  private findMatchingTransition(
    action: GameAction,
    transitions: ReturnType<typeof getNextStates>
  ) {
    return transitions.find((t) => {
      if (t.action.type !== action.type) return false;

      if ('player' in t.action && 'player' in action) {
        if (t.action.player !== action.player) return false;
      }

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
  }

  /**
   * Get current game view for clients
   * @param forPlayerId - Optional player ID to filter actions for
   */
  getView(forPlayerId?: string): GameView {
    return this.buildUpdate(forPlayerId).view;
  }

  /**
   * Get full multiplayer state snapshot.
   */
  getState(): MultiplayerGameState {
    return this.cloneMultiplayerState(this.mpState);
  }

  /**
   * Get valid actions for the given player session.
   */
  getActionsForPlayer(playerId: string): ValidAction[] {
    const allValidActions = this.getValidActionsComposed(this.mpState.coreState);
    const transitions = getNextStates(this.mpState.coreState, this.layers, this.rules);
    const actionsMap = this.buildActionsMap(
      this.mpState.players,
      allValidActions,
      transitions,
      this.mpState.coreState
    );
    return actionsMap[playerId] ?? [];
  }

  /**
   * Get valid actions for all player sessions.
   */
  getActionsMap(): Record<string, ValidAction[]> {
    const allValidActions = this.getValidActionsComposed(this.mpState.coreState);
    const transitions = getNextStates(this.mpState.coreState, this.layers, this.rules);
    return this.buildActionsMap(
      this.mpState.players,
      allValidActions,
      transitions,
      this.mpState.coreState
    );
  }

  /**
   * Execute an action with authorization
   */
  executeAction(playerId: string, action: GameAction, timestamp: number): { success: boolean; error?: string } {
    const request = {
      playerId,
      action,
      timestamp
    };

    const result = authorizeAndExecute(this.mpState, request, this.getValidActionsComposed, this.rules);

    if (!result.success) {
      return { success: false, error: result.error };
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

    return { success: true };
  }

  /**
   * Change player control type
   */
  setPlayerControl(playerIndex: number, type: 'human' | 'ai'): void {
    const targetSession = this.mpState.players.find((session: PlayerSession) => session.playerIndex === playerIndex);
    if (!targetSession) {
      throw new Error(`Player ${playerIndex} not found`);
    }

    const capabilities = this.buildBaseCapabilities(playerIndex, type);
    const updatedStateResult = updatePlayerSession(this.mpState, targetSession.playerId, {
      controlType: type,
      capabilities
    });

    if (!updatedStateResult.success) {
      throw new Error(updatedStateResult.error);
    }

    const updatedPlayers = updatedStateResult.value.players;

    // Update players map
    for (const session of updatedPlayers) {
      this.players.set(session.playerId, session);
    }

    // Update core state playerTypes to match
    const updatedCoreState = {
      ...this.mpState.coreState,
      playerTypes: updatedPlayers.map((s: PlayerSession) => s.controlType) as ('human' | 'ai')[]
    };

    this.mpState = {
      ...updatedStateResult.value,
      coreState: updatedCoreState,
      lastActionAt: Date.now()
    };

    this.lastUpdate = this.mpState.lastActionAt;
    this.notifyListeners();
  }

  /**
   * Connect or replace a player session.
   */
  joinPlayer(session: PlayerSession): Result<PlayerSession> {
    const existing = this.players.get(session.playerId);
    if (!existing) {
      return err(`Player ${session.playerId} not found`);
    }

    const merged: PlayerSession = {
      ...existing,
      ...session,
      isConnected: true,
      capabilities: (session.capabilities ?? existing.capabilities).map((cap: Capability) => ({ ...cap }))
    };

    const result = updatePlayerSession(this.mpState, existing.playerId, merged);
    if (!result.success) {
      return err(result.error);
    }

    this.mpState = {
      ...result.value,
      lastActionAt: Date.now()
    };

    this.players.set(merged.playerId, merged);
    this.lastUpdate = this.mpState.lastActionAt;
    this.notifyListeners();

    return ok(merged);
  }

  /**
   * Disconnect a player session.
   */
  leavePlayer(playerId: string): Result<PlayerSession> {
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

    this.mpState = {
      ...result.value,
      lastActionAt: Date.now()
    };

    this.players.set(playerId, updated);
    this.lastUpdate = this.mpState.lastActionAt;
    this.notifyListeners();

    return ok(updated);
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
  subscribe(playerId: string | undefined, listener: (update: KernelUpdate) => void): () => void {
    const listenerKey = playerId ?? `unfiltered-${Date.now()}-${Math.random()}`;

    const record: { perspective?: string; listener: (update: KernelUpdate) => void } = {
      listener
    };

    if (playerId !== undefined) {
      record.perspective = playerId;
    }

    this.listeners.set(listenerKey, record);

    // Send current state immediately
    listener(this.buildUpdate(playerId));

    // Return unsubscribe function
    return () => {
      this.listeners.delete(listenerKey);
    };
  }

  /**
   * Destroy the game kernel
   */
  destroy(): void {
    this.listeners.clear();
  }

  /**
   * Private: Create view for clients
   * @param forPlayerId - Optional player ID to filter actions for
   */
  private createView(
    forPlayerId?: string,
    context?: {
      state: MultiplayerGameState;
      allActions: GameAction[];
      transitions: ReturnType<typeof getNextStates>;
      actionsByPlayer: Record<string, ValidAction[]>;
    }
  ): GameView {
    const stateContext = context?.state ?? this.cloneMultiplayerState(this.mpState);
    const { coreState, players } = stateContext;
    const allValidActions = context?.allActions ?? this.getValidActionsComposed(coreState);
    const transitions = context?.transitions ?? getNextStates(coreState, this.layers, this.rules);
    const actionsByPlayer =
      context?.actionsByPlayer ??
      this.buildActionsMap(players, allValidActions, transitions, coreState);

    const session = forPlayerId ? players.find((p: PlayerSession) => p.playerId === forPlayerId) : undefined;

    const visibleState: import('../game/types').FilteredGameState = session
      ? getVisibleStateForSession(coreState, session)
      : this.convertToFilteredState(coreState);

    let validActions: ValidAction[];

    if (session) {
      validActions = actionsByPlayer[session.playerId] ?? [];
    } else if (forPlayerId) {
      // Requested perspective but no session found
      validActions = [];
    } else {
      validActions = actionsByPlayer[UNFILTERED_KEY] ??
        this.buildValidActionsForSession(undefined, allValidActions, transitions, coreState);
    }

    const playerInfoList: PlayerInfo[] = Array.from(players).map((playerSession: PlayerSession) => ({
      playerId: playerSession.playerIndex,
      controlType: playerSession.controlType,
      sessionId: playerSession.playerId,
      connected: playerSession.isConnected ?? true,
      name: playerSession.name || `Player ${playerSession.playerIndex + 1}`,
      capabilities: playerSession.capabilities.map((cap: Capability) => ({ ...cap }))
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
  private convertToFilteredState(state: GameState): import('../game/types').FilteredGameState {
    const filteredPlayers = state.players.map((player: GameState['players'][number]) => ({
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

  private cloneActionsMap(actions: Record<string, ValidAction[]>): Record<string, ValidAction[]> {
    const clone: Record<string, ValidAction[]> = {};
    for (const [key, list] of Object.entries(actions)) {
      clone[key] = list.map(valid => {
        const actionClone: GameAction = { ...valid.action };
        if ('meta' in actionClone && actionClone.meta) {
          actionClone.meta = JSON.parse(JSON.stringify(actionClone.meta));
        }
        return {
          ...valid,
          action: actionClone
        };
      });
    }
    return clone;
  }

  private cloneMultiplayerState(state: MultiplayerGameState): MultiplayerGameState {
    return {
      gameId: state.gameId,
      coreState: cloneGameState(state.coreState),
      players: state.players.map((session: PlayerSession) => ({
        ...session,
        capabilities: session.capabilities.map((cap: Capability) => ({ ...cap }))
      })),
      createdAt: state.createdAt,
      lastActionAt: state.lastActionAt,
      enabledVariants: state.enabledVariants.map((variant: VariantConfig) => ({ ...variant })),
      enabledLayers: state.enabledLayers ?? []
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
   * Uses standard capability builders from vision spec §4.3
   */
  private buildBaseCapabilities(playerIndex: number, controlType: 'human' | 'ai'): Capability[] {
    const idx = playerIndex as 0 | 1 | 2 | 3;
    return controlType === 'human'
      ? humanCapabilities(idx)
      : aiCapabilities(idx);
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
      const autoAction = actions.find((a: GameAction) => a.autoExecute === true);

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
        { playerId: session.playerId, action: autoAction, timestamp: Date.now() },
        this.getValidActionsComposed,
        this.rules
      );

      if (!result.success) {
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
    if (this.listeners.size === 0) {
      return;
    }

    const stateSnapshot = this.cloneMultiplayerState(this.mpState);
    const allValidActions = this.getValidActionsComposed(stateSnapshot.coreState);
    const transitions = getNextStates(stateSnapshot.coreState, this.layers, this.rules);
    const actionsMap = this.buildActionsMap(
      stateSnapshot.players,
      allValidActions,
      transitions,
      stateSnapshot.coreState
    );

    for (const [listenerKey, record] of this.listeners) {
      const perspective = record.perspective ?? (listenerKey.startsWith('unfiltered-') ? undefined : listenerKey);
      const view = this.createView(perspective, {
        state: stateSnapshot,
        allActions: allValidActions,
        transitions,
        actionsByPlayer: actionsMap
      });

      const payload: KernelUpdate = {
        view,
        state: this.cloneMultiplayerState(stateSnapshot),
        actions: this.cloneActionsMap(actionsMap)
      };

      if (perspective !== undefined) {
        payload.perspective = perspective;
      }

      record.listener(payload);
    }
  }
}
