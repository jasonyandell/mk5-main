/**
 * HeadlessRoom - Minimal Room wrapper for standalone tools.
 *
 * Provides a simple API for tools like gameSimulator and urlReplay to execute
 * game actions without needing the full multiplayer infrastructure (transport,
 * subscriptions, etc.).
 *
 * Design:
 * - Uses Room internally (single composition point)
 * - Bypasses network layer (in-memory only)
 * - Fast enough for simulations (1000s of games)
 * - Standalone: no external dependencies on transport/subscriptions
 *
 * This ensures tools use the same composition path as multiplayer games.
 */

import { Room } from './Room';
import type { GameConfig } from '../game/types/config';
import type { GameAction, GameState } from '../game/types';
import type { PlayerSession } from '../game/multiplayer/types';
import type { ValidAction } from '../shared/multiplayer/protocol';

/**
 * HeadlessRoom - Minimal Room API for tools and scripts.
 *
 * Example usage:
 * ```typescript
 * const room = new HeadlessRoom({ playerTypes: ['ai', 'ai', 'ai', 'ai'] }, 12345);
 * const actions = room.getValidActions(0);
 * room.executeAction(0, actions[0].action);
 * const state = room.getState();
 * ```
 */
export class HeadlessRoom {
  private room: Room;
  private playerIdMap: Map<number, string>; // playerIndex -> playerId

  /**
   * Create a HeadlessRoom instance.
   *
   * @param config - Game configuration
   * @param seed - Optional shuffle seed for deterministic games
   */
  constructor(config: GameConfig, seed?: number) {
    // Normalize config with seed if provided
    const fullConfig: GameConfig = {
      ...config,
      ...(seed !== undefined ? { shuffleSeed: seed } : {})
    };

    // Create 4 player sessions (all AI for tools)
    const playerTypes = config.playerTypes ?? ['ai', 'ai', 'ai', 'ai'];
    const initialPlayers: PlayerSession[] = playerTypes.map((type, index) => ({
      playerId: `player-${index}`,
      playerIndex: index,
      controlType: type,
      isConnected: true
      // capabilities: undefined - Room will populate with buildBaseCapabilities
    } as PlayerSession));

    // Create Room with unique game ID
    const gameId = `headless-${Date.now()}-${Math.random().toString(36).slice(2)}`;
    this.room = new Room(gameId, fullConfig, initialPlayers);

    // Build player index -> ID map for convenience
    this.playerIdMap = new Map(
      initialPlayers.map(p => [p.playerIndex, p.playerId])
    );

    // Note: No transport needed - we bypass the protocol layer entirely
  }

  /**
   * Get current core game state (unfiltered).
   *
   * @returns Current GameState
   */
  getState(): GameState {
    const mpState = this.room.getState();
    return mpState.coreState;
  }

  /**
   * Get valid actions for a player by index.
   *
   * @param playerIndex - Player index (0-3)
   * @returns Array of valid actions
   */
  getValidActions(playerIndex: number): ValidAction[] {
    const playerId = this.playerIdMap.get(playerIndex);
    if (!playerId) {
      throw new Error(`Invalid player index: ${playerIndex}`);
    }

    return this.room.getActionsForPlayer(playerId);
  }

  /**
   * Execute an action for a player by index.
   *
   * @param playerIndex - Player index (0-3)
   * @param action - Game action to execute
   * @throws Error if action fails
   */
  executeAction(playerIndex: number, action: GameAction): void {
    const playerId = this.playerIdMap.get(playerIndex);
    if (!playerId) {
      throw new Error(`Invalid player index: ${playerIndex}`);
    }

    const result = this.room.executeAction(playerId, action);
    if (!result.success) {
      throw new Error(`Action failed: ${result.error}`);
    }
  }

  /**
   * Replay a sequence of actions.
   * Useful for reconstructing game state from action history.
   *
   * @param actions - Array of actions to replay
   * @throws Error if any action fails
   */
  replayActions(actions: GameAction[]): void {
    for (const action of actions) {
      // Determine which player should execute this action
      const playerIndex = this.getPlayerForAction(action);
      this.executeAction(playerIndex, action);
    }
  }

  /**
   * Get full unfiltered multiplayer state.
   * For simulations that need access to all state.
   *
   * @returns Complete MultiplayerGameState
   */
  getUnfilteredState() {
    return this.room.getState();
  }

  /**
   * Get all valid actions across all players.
   *
   * @returns Map of playerId -> valid actions
   */
  getAllActions(): Record<string, ValidAction[]> {
    return this.room.getActionsMap();
  }

  /**
   * Helper to determine which player should execute an action.
   *
   * @param action - Action to execute
   * @returns Player index that should execute this action
   */
  private getPlayerForAction(action: GameAction): number {
    // Actions with explicit player field
    if ('player' in action && typeof action.player === 'number') {
      return action.player;
    }

    // Consensus actions can be executed by any player
    // For tools, we just use player 0
    if (action.type === 'complete-trick' ||
        action.type === 'score-hand' ||
        action.type === 'redeal') {
      return 0;
    }

    // Should not reach here
    throw new Error(`Cannot determine player for action: ${action.type}`);
  }
}
