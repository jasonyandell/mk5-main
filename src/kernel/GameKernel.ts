/**
 * GameKernel - Pure game authority engine.
 *
 * This is the ONLY place where game state mutations happen.
 * NO transport, AI, or multiplayer concerns here - just pure game logic.
 *
 * Design principles:
 * - Pure state management via kernel.ts pure functions
 * - Thin stateful wrapper for state management only
 * - Authorization via existing authorizeAndExecute
 * - Action transformer support via composition
 * - Deployable anywhere (local, Worker, Cloudflare)
 * - No orchestration or subscription management (GameServer handles that)
 */

import type { GameState, GameAction } from '../game/types';
import type { GameConfig } from '../game/types/config';
import type {
  GameView,
  ValidAction
} from '../shared/multiplayer/protocol';
import type { MultiplayerGameState, PlayerSession, Capability, Result } from '../game/multiplayer/types';
import { ok, err } from '../game/multiplayer/types';
import { humanCapabilities, aiCapabilities } from '../game/multiplayer/capabilities';
import { createInitialState } from '../game/core/state';
import { getValidActions } from '../game/core/gameEngine';
import { applyActionTransformers } from '../game/action-transformers/registry';
import type { ActionTransformerConfig } from '../game/action-transformers/types';
import { createMultiplayerGame, updatePlayerSession } from '../game/multiplayer/stateLifecycle';
import { composeRules, baseRuleSet, getRuleSetsByNames } from '../game/rulesets';
import type { ExecutionContext } from '../game/types/execution';
import {
  executeKernelAction,
  processAutoExecuteActions,
  buildKernelView,
  buildActionsMap,
  updatePlayerControlPure,
  cloneMultiplayerState
} from './kernel';

export interface KernelUpdate {
  view: GameView;
  state: MultiplayerGameState;
  actions: Record<string, ValidAction[]>;
  perspective?: string;
}

/**
 * GameKernel - Pure game logic engine
 *
 * Responsibilities:
 * - Pure state management (no mutations, only replacements)
 * - Action authorization and execution
 * - View generation with capability-based filtering
 * - Action transformer and rule set composition
 *
 * NOT responsible for:
 * - Transport/networking
 * - AI client lifecycle
 * - Protocol message routing
 * - Client connections
 * - Subscription management (GameServer handles that)
 */
export class GameKernel {
  // Mutable state (updated on actions)
  private mpState: MultiplayerGameState;
  private lastUpdate: number;

  // Mutable cache (for efficient player lookups)
  private players: Map<string, PlayerSession>; // playerId -> PlayerSession

  // Immutable config (set once in constructor, never mutated)
  private readonly metadata: {
    gameId: string;
    actionTransformerConfigs: ActionTransformerConfig[];
    created: number;
  };
  private readonly ctx: ExecutionContext;

  constructor(gameId: string, config: GameConfig, players: PlayerSession[]) {
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

    // Compose action transformers into single getValidActions function
    const actionTransformerConfigs = config.variants ?? [];

    // Initialize timestamp
    const createdAt = Date.now();
    this.lastUpdate = createdAt;

    // Initialize immutable metadata (all at once before freezing)
    this.metadata = {
      gameId,
      actionTransformerConfigs,
      created: createdAt
    };

    // Compose rule sets into rules
    const enabledRuleSetNames = config.enabledRuleSets ?? [];
    const enabledRuleSets = enabledRuleSetNames.length > 0
      ? getRuleSetsByNames(enabledRuleSetNames)
      : [];

    const ruleSets = [baseRuleSet, ...enabledRuleSets];
    const rules = composeRules(ruleSets);

    // Compose action transformers with rule sets/rules threaded through
    const baseWithRuleSets = (state: GameState) =>
      getValidActions(state, ruleSets, rules);

    const getValidActionsComposed = applyActionTransformers(baseWithRuleSets, actionTransformerConfigs);

    // Create execution context (immutable after this point)
    this.ctx = Object.freeze({
      ruleSets: Object.freeze(ruleSets),
      rules,
      getValidActions: getValidActionsComposed
    });

    // Create initial game state
    const initialState = createInitialState({
      playerTypes: config.playerTypes,
      ...(config.shuffleSeed !== undefined ? { shuffleSeed: config.shuffleSeed } : {}),
      ...(config.theme !== undefined ? { theme: config.theme } : {}),
      ...(config.colorOverrides !== undefined ? { colorOverrides: config.colorOverrides } : {}),
      variants: actionTransformerConfigs
    });

    // Store pure GameState (NO filtering - filtering happens in buildKernelView())
    const now = Date.now();
    this.mpState = createMultiplayerGame({
      gameId,
      coreState: initialState,
      players: normalizedPlayers,
      enabledActionTransformers: actionTransformerConfigs,
      enabledRuleSets: config.enabledRuleSets ?? [],
      createdAt: now,
      lastActionAt: now
    });
    this.players = new Map(this.mpState.players.map((p: PlayerSession) => [p.playerId, p]));

    // Process any immediate scripted actions emitted at startup using pure function
    const result = processAutoExecuteActions(
      this.mpState,
      this.ctx
    );
    if (result.success) {
      this.mpState = result.value;
    }
  }

  private buildUpdate(forPlayerId?: string): KernelUpdate {
    const stateSnapshot = cloneMultiplayerState(this.mpState);
    const view = buildKernelView(
      stateSnapshot,
      forPlayerId,
      this.ctx,
      {
        gameId: this.metadata.gameId,
        actionTransformerConfigs: this.metadata.actionTransformerConfigs,
        created: this.metadata.created,
        lastUpdate: this.lastUpdate
      }
    );

    // Get actions map (transitions are computed internally)
    const actionsByPlayer = buildActionsMap(stateSnapshot, this.ctx);

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
    return cloneMultiplayerState(this.mpState);
  }

  /**
   * Get valid actions for the given player session.
   */
  getActionsForPlayer(playerId: string): ValidAction[] {
    const actionsMap = buildActionsMap(this.mpState, this.ctx);
    return actionsMap[playerId] ?? [];
  }

  /**
   * Get valid actions for all player sessions.
   */
  getActionsMap(): Record<string, ValidAction[]> {
    return buildActionsMap(this.mpState, this.ctx);
  }

  /**
   * Execute an action with authorization
   */
  executeAction(playerId: string, action: GameAction, timestamp: number): { success: boolean; error?: string } {
    // Use pure function for action execution with auto-execute
    const result = executeKernelAction(
      this.mpState,
      playerId,
      action,
      timestamp,
      this.ctx
    );

    if (!result.success) {
      return { success: false, error: result.error };
    }

    // Update mutable state
    this.mpState = result.value;
    this.lastUpdate = Date.now();

    return { success: true };
  }

  /**
   * Change player control type
   */
  setPlayerControl(playerId: string, type: 'human' | 'ai'): void {
    const session = this.players.get(playerId);
    if (!session) {
      throw new Error(`Player ${playerId} not found`);
    }

    const capabilities = this.buildBaseCapabilities(session.playerIndex, type);

    // Use pure function for state transition
    const result = updatePlayerControlPure(
      this.mpState,
      session.playerIndex,
      type,
      capabilities
    );

    if (!result.success) {
      throw new Error(result.error);
    }

    this.applyStateUpdate(result);
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

    this.applyStateUpdate({
      success: true,
      value: {
        ...result.value,
        lastActionAt: Date.now()
      }
    });

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

    this.applyStateUpdate({
      success: true,
      value: {
        ...result.value,
        lastActionAt: Date.now()
      }
    });

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
   * Private: Apply a state update result to mutable kernel state.
   * Syncs players map and updates timestamps.
   */
  private applyStateUpdate(result: { success: true; value: MultiplayerGameState }): void {
    this.mpState = result.value;

    // Sync players map
    for (const session of this.mpState.players) {
      this.players.set(session.playerId, session);
    }

    this.lastUpdate = this.mpState.lastActionAt ?? Date.now();
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
}
