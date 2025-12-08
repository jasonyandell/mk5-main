/**
 * MCCFR Trainer for Texas 42
 *
 * Implements External Sampling Monte Carlo Counterfactual Regret Minimization.
 * Focuses on trick-taking phase only (bidding/trump selection use separate logic).
 *
 * Algorithm:
 * - Sample opponent actions according to current strategy
 * - Traverse ALL actions for the traversing player
 * - Update regrets: (actionValue - expectedValue) * opponentReachProb
 * - Average strategy converges to Nash equilibrium
 */

import type { GameState } from '../../types';
import type { ValidAction } from '../../../multiplayer/types';
import type { MCCFRConfig, TrainingResult, InfoSetKey, ActionKey, ActionProbabilities } from './types';
import { RegretTable } from './regret-table';
import { actionToKey, getActionKeys, sampleAction } from './action-abstraction';
import { computeCountCentricHash } from '../cfr-metrics';
import { HeadlessRoom } from '../../../server/HeadlessRoom';
import { createSeededRng, type RandomGenerator } from '../hand-sampler';

/**
 * Default MCCFR configuration.
 */
const DEFAULT_CONFIG: MCCFRConfig = {
  iterations: 10000,
  seed: 42,
  regretDiscount: 1.0,  // No discount by default
  regretFloor: -1e9,
  progressInterval: 1000
};

/**
 * MCCFR Trainer class.
 *
 * Usage:
 * ```typescript
 * const trainer = new MCCFRTrainer({ iterations: 100000, seed: 12345 });
 * const result = await trainer.train();
 * const serialized = trainer.serialize();
 * ```
 */
export class MCCFRTrainer {
  private config: MCCFRConfig;
  private regretTable: RegretTable;
  private rng: RandomGenerator;
  private iterationsCompleted: number = 0;
  private trainingStartTime: number = 0;

  constructor(config: Partial<MCCFRConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.regretTable = new RegretTable(this.config.regretFloor);
    this.rng = createSeededRng(this.config.seed);
  }

  /**
   * Run MCCFR training.
   *
   * Each iteration:
   * 1. Create a fresh game with random deal
   * 2. Skip to playing phase (use fixed trump selection for now)
   * 3. Run external sampling MCCFR traversal
   * 4. Update regrets and strategy sums
   */
  async train(): Promise<TrainingResult> {
    this.trainingStartTime = Date.now();
    const { iterations, progressInterval, onProgress } = this.config;

    for (let i = 0; i < iterations; i++) {
      // Generate unique seed for this iteration
      const gameSeed = this.config.seed + i * 1000000;

      // Run one iteration of MCCFR
      this.runIteration(gameSeed, i % 4);

      this.iterationsCompleted++;

      // Progress callback
      if (onProgress && progressInterval && (i + 1) % progressInterval === 0) {
        onProgress(i + 1, iterations, this.regretTable.size);
      }

      // Yield to event loop periodically
      if ((i + 1) % 100 === 0) {
        await new Promise(resolve => setTimeout(resolve, 0));
      }
    }

    const trainingTimeMs = Date.now() - this.trainingStartTime;

    return {
      iterations: this.iterationsCompleted,
      infoSetCount: this.regretTable.size,
      trainingTimeMs,
      iterationsPerSecond: Math.round(this.iterationsCompleted / (trainingTimeMs / 1000))
    };
  }

  /**
   * Run a single MCCFR iteration.
   *
   * @param seed - Random seed for this game
   * @param traversingPlayer - Player index doing full traversal (others sampled)
   */
  private runIteration(seed: number, traversingPlayer: number): void {
    // Create fresh game with only base layer (no consensus layer - we don't need
    // complete-trick/score-hand actions for MCCFR training)
    const room = new HeadlessRoom(
      { playerTypes: ['ai', 'ai', 'ai', 'ai'], shuffleSeed: seed, layers: ['base'] },
      seed
    );

    // Fast-forward through bidding and trump selection with fixed strategy
    this.skipToPlayingPhase(room, seed);

    // Check if game reached playing phase
    const state = room.getState();
    if (state.phase !== 'playing') {
      return; // Some games may end early (e.g., all pass)
    }

    // Run external sampling traversal
    this.externalSamplingTraverse(room, traversingPlayer, 1.0, 1.0);
  }

  /**
   * Skip bidding and trump selection phases with simple fixed strategy.
   * For training, we just need to reach the playing phase consistently.
   */
  private skipToPlayingPhase(room: HeadlessRoom, seed: number): void {
    const iterRng = createSeededRng(seed);
    let actionsExecuted = 0;
    const maxActions = 20; // Bidding + trump shouldn't take more than this

    while (actionsExecuted < maxActions) {
      const state = room.getState();

      if (state.phase === 'playing') {
        return; // Done
      }

      if (state.phase === 'game_end' || state.phase === 'scoring') {
        return; // Game ended
      }

      // Get actions for current player
      const actions = room.getValidActions(state.currentPlayer);
      if (actions.length === 0) {
        // Try consensus actions
        const allActions = room.getAllActions();
        const anyActions = Object.values(allActions).flat();
        const autoAction = anyActions.find(a =>
          a.action.type === 'complete-trick' ||
          a.action.type === 'score-hand' ||
          a.action.autoExecute
        );
        if (autoAction) {
          const player = 'player' in autoAction.action ? autoAction.action.player : 0;
          room.executeAction(player, autoAction.action);
          actionsExecuted++;
          continue;
        }
        return; // No actions available
      }

      // Simple strategy: bid 30 if first to act, otherwise pass
      // For trump: pick first available option
      let selectedAction: ValidAction | undefined;

      if (state.phase === 'bidding') {
        // Simple bidding: always bid 30 if we can, otherwise pass
        const bidAction = actions.find(a =>
          a.action.type === 'bid' && 'value' in a.action && a.action.value === 30
        );
        const passAction = actions.find(a => a.action.type === 'pass');

        // Alternate between bidding and passing based on position
        if (state.currentPlayer === state.dealer) {
          // Dealer: bid 30 to ensure we get to play
          selectedAction = bidAction || passAction || actions[0];
        } else {
          // Non-dealer: pass to let dealer win
          selectedAction = passAction || actions[0];
        }
      } else if (state.phase === 'trump_selection') {
        // Simple trump selection: pick randomly
        const idx = Math.floor(iterRng.random() * actions.length);
        selectedAction = actions[idx];
      } else {
        selectedAction = actions[0];
      }

      if (!selectedAction) {
        return;
      }

      const player = 'player' in selectedAction.action
        ? selectedAction.action.player
        : state.currentPlayer;

      try {
        room.executeAction(player, selectedAction.action);
        actionsExecuted++;
      } catch {
        return;
      }
    }
  }

  /**
   * External sampling MCCFR traversal.
   *
   * For the traversing player: explore ALL actions
   * For other players: sample ONE action according to current strategy
   *
   * @param room - HeadlessRoom instance
   * @param traversingPlayer - Player doing full traversal
   * @param traverserReachProb - Traversing player's reach probability
   * @param opponentReachProb - Product of opponent reach probabilities
   * @returns Counterfactual value for traversing player
   */
  private externalSamplingTraverse(
    room: HeadlessRoom,
    traversingPlayer: number,
    traverserReachProb: number,
    opponentReachProb: number
  ): number {
    const state = room.getState();

    // Terminal state: return utility
    // With base layer only, playing phase ends after all 7 tricks without explicit score-hand
    if (state.phase === 'game_end' || state.phase === 'scoring' || state.phase === 'one-hand-complete') {
      return this.getUtility(state, traversingPlayer);
    }

    // Not in playing phase - return utility
    if (state.phase !== 'playing') {
      return this.getUtility(state, traversingPlayer);
    }

    const currentPlayer = state.currentPlayer;
    const actions = room.getValidActions(currentPlayer);

    // Filter to play actions only
    const playActions = actions.filter(a => a.action.type === 'play');
    if (playActions.length === 0) {
      return this.getUtility(state, traversingPlayer);
    }

    // Get information set key
    const infoSetKey = computeCountCentricHash(state, currentPlayer);
    const actionKeys = getActionKeys(playActions);

    // Get current strategy for this info set
    const strategy = this.regretTable.getStrategy(infoSetKey, actionKeys);

    if (currentPlayer === traversingPlayer) {
      // Traversing player: explore ALL actions
      return this.traverseAllActions(
        room, state, traversingPlayer, playActions, infoSetKey, actionKeys, strategy,
        traverserReachProb, opponentReachProb
      );
    } else {
      // Opponent: sample ONE action according to strategy
      return this.sampleAndTraverse(
        room, traversingPlayer, playActions, strategy,
        traverserReachProb, opponentReachProb, currentPlayer
      );
    }
  }

  /**
   * Traverse all actions for the traversing player.
   */
  private traverseAllActions(
    room: HeadlessRoom,
    state: GameState,
    traversingPlayer: number,
    playActions: ValidAction[],
    infoSetKey: InfoSetKey,
    _actionKeys: ActionKey[], // Available for future use (e.g., logging)
    strategy: ActionProbabilities,
    traverserReachProb: number,
    opponentReachProb: number
  ): number {
    const actionValues = new Map<ActionKey, number>();
    let expectedValue = 0;

    // Explore each action
    for (const action of playActions) {
      const actionKey = actionToKey(action);
      const prob = strategy.get(actionKey) ?? (1 / playActions.length);

      // Clone room state by creating a new room and replaying
      const clonedRoom = this.cloneRoom(room, state);

      // Execute action
      const player = 'player' in action.action ? action.action.player : state.currentPlayer;
      clonedRoom.executeAction(player, action.action);

      // Recurse with updated reach probabilities
      const childValue = this.externalSamplingTraverse(
        clonedRoom,
        traversingPlayer,
        traverserReachProb * prob,
        opponentReachProb
      );

      actionValues.set(actionKey, childValue);
      expectedValue += prob * childValue;
    }

    // Update regrets
    this.regretTable.updateRegrets(infoSetKey, actionValues, expectedValue, opponentReachProb);

    // Update strategy sum
    this.regretTable.updateStrategySum(infoSetKey, strategy, traverserReachProb);

    return expectedValue;
  }

  /**
   * Sample one action and traverse for opponent.
   */
  private sampleAndTraverse(
    room: HeadlessRoom,
    traversingPlayer: number,
    playActions: ValidAction[],
    strategy: ActionProbabilities,
    traverserReachProb: number,
    opponentReachProb: number,
    currentPlayer: number
  ): number {
    // Sample action according to strategy
    const sampledAction = sampleAction(strategy, playActions, this.rng);
    const actionKey = actionToKey(sampledAction);
    const prob = strategy.get(actionKey) ?? (1 / playActions.length);

    // Execute sampled action
    const player = 'player' in sampledAction.action ? sampledAction.action.player : currentPlayer;
    room.executeAction(player, sampledAction.action);

    // Update opponent reach probability
    const newOpponentReachProb = opponentReachProb * prob;

    // Recurse
    return this.externalSamplingTraverse(room, traversingPlayer, traverserReachProb, newOpponentReachProb);
  }

  /**
   * Clone a HeadlessRoom by creating a new one and replaying actions.
   */
  private cloneRoom(_room: HeadlessRoom, state: GameState): HeadlessRoom {
    const newRoom = new HeadlessRoom(
      { playerTypes: ['ai', 'ai', 'ai', 'ai'], shuffleSeed: state.shuffleSeed, layers: ['base'] },
      state.shuffleSeed
    );

    // Replay action history - skip consensus actions (complete-trick, score-hand)
    // since we're using only base layer without consensus
    for (const action of state.actionHistory) {
      if (action.type === 'complete-trick' || action.type === 'score-hand') {
        continue;
      }

      const player = 'player' in action ? action.player : 0;
      newRoom.executeAction(player, action);
    }

    return newRoom;
  }

  /**
   * Calculate utility (team score difference) for a player.
   * Texas 42 is a team game: players 0,2 vs players 1,3.
   */
  private getUtility(state: GameState, playerIndex: number): number {
    const myTeam = playerIndex % 2; // 0 for team 0 (players 0,2), 1 for team 1 (players 1,3)
    const myScore = state.teamScores[myTeam] ?? 0;
    const oppScore = state.teamScores[1 - myTeam] ?? 0;

    // Return point differential (positive is good for player)
    return myScore - oppScore;
  }

  /**
   * Get the regret table for external access.
   */
  getRegretTable(): RegretTable {
    return this.regretTable;
  }

  /**
   * Serialize trained strategy.
   */
  serialize() {
    const trainingTimeMs = Date.now() - this.trainingStartTime;
    return this.regretTable.serialize(this.config, this.iterationsCompleted, trainingTimeMs);
  }

  /**
   * Load a pre-trained strategy.
   */
  static fromSerialized(data: ReturnType<typeof RegretTable.prototype.serialize>): MCCFRTrainer {
    const trainer = new MCCFRTrainer(data.config);
    trainer.regretTable = RegretTable.deserialize(data);
    trainer.iterationsCompleted = data.iterationsCompleted;
    return trainer;
  }
}
