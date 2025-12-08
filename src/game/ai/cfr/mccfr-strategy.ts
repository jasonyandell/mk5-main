/**
 * MCCFR Strategy - AIStrategy implementation using trained MCCFR regrets.
 *
 * Uses the average strategy from training for play decisions.
 * Falls back to uniform random for unseen information sets.
 *
 * Note: MCCFR training focuses on trick-taking (playing phase).
 * Bidding and trump selection use simple heuristics.
 */

import type { AIStrategy } from '../types';
import type { GameState } from '../../types';
import type { ValidAction } from '../../../multiplayer/types';
import type { SerializedStrategy } from './types';
import { RegretTable } from './regret-table';
import { getActionKeys, selectBestAction, sampleAction } from './action-abstraction';
import { computeCountCentricHash } from '../cfr-metrics';
import { defaultRng, type RandomGenerator } from '../hand-sampler';
import { determineBestTrump } from '../hand-strength';

/**
 * Configuration for MCCFRStrategy.
 */
export interface MCCFRStrategyConfig {
  /** Use greedy (best action) or stochastic (sample from distribution) selection. */
  mode: 'greedy' | 'stochastic';

  /** RNG for stochastic mode. */
  rng?: RandomGenerator;
}

const DEFAULT_CONFIG: MCCFRStrategyConfig = {
  mode: 'greedy'
};

/**
 * AI Strategy using trained MCCFR regrets.
 *
 * For playing phase: uses trained average strategy.
 * For bidding/trump: uses simple heuristics (not trained).
 */
export class MCCFRStrategy implements AIStrategy {
  private regretTable: RegretTable;
  private config: MCCFRStrategyConfig;
  private rng: RandomGenerator;

  constructor(
    regretTable: RegretTable,
    config: Partial<MCCFRStrategyConfig> = {}
  ) {
    this.regretTable = regretTable;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.rng = config.rng ?? defaultRng;
  }

  /**
   * Create an MCCFRStrategy from serialized data.
   */
  static fromSerialized(
    data: SerializedStrategy,
    config: Partial<MCCFRStrategyConfig> = {}
  ): MCCFRStrategy {
    const regretTable = RegretTable.deserialize(data);
    return new MCCFRStrategy(regretTable, config);
  }

  /**
   * Choose an action from available valid actions.
   */
  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('MCCFRStrategy: No valid actions available');
    }

    if (validActions.length === 1) {
      return validActions[0]!;
    }

    switch (state.phase) {
      case 'playing':
        return this.choosePlayAction(state, validActions);
      case 'bidding':
        return this.chooseBidAction(validActions);
      case 'trump_selection':
        return this.chooseTrumpAction(state, validActions);
      default:
        return validActions[0]!;
    }
  }

  /**
   * Choose a play action using trained MCCFR strategy.
   */
  private choosePlayAction(state: GameState, validActions: ValidAction[]): ValidAction {
    // Handle complete-trick action
    const completeTrick = validActions.find(a => a.action.type === 'complete-trick');
    if (completeTrick) return completeTrick;

    // Filter to play actions
    const playActions = validActions.filter(a => a.action.type === 'play');
    if (playActions.length === 0) {
      return validActions[0]!;
    }

    if (playActions.length === 1) {
      return playActions[0]!;
    }

    // Get information set key
    const infoSetKey = computeCountCentricHash(state, state.currentPlayer);
    const actionKeys = getActionKeys(playActions);

    // Get average strategy (converged equilibrium strategy)
    const strategy = this.regretTable.getAverageStrategy(infoSetKey, actionKeys);

    // Select action based on mode
    if (this.config.mode === 'greedy') {
      return selectBestAction(strategy, playActions);
    } else {
      return sampleAction(strategy, playActions, this.rng);
    }
  }

  /**
   * Choose a bid action using simple heuristics.
   * MCCFR doesn't train on bidding - always bid 30 or pass.
   */
  private chooseBidAction(validActions: ValidAction[]): ValidAction {
    const passAction = validActions.find(a => a.action.type === 'pass');

    // Simple heuristic: bid 30 if available, otherwise pass
    const bid30 = validActions.find(a =>
      a.action.type === 'bid' && 'value' in a.action && a.action.value === 30
    );

    return bid30 ?? passAction ?? validActions[0]!;
  }

  /**
   * Choose trump using hand strength heuristics.
   */
  private chooseTrumpAction(state: GameState, validActions: ValidAction[]): ValidAction {
    const player = state.players[state.currentPlayer];
    if (!player) return validActions[0]!;

    const bestTrump = determineBestTrump(player.hand, player.suitAnalysis);

    if (bestTrump.type === 'doubles') {
      const action = validActions.find(a =>
        a.action.type === 'select-trump' &&
        'trump' in a.action &&
        a.action.trump.type === 'doubles'
      );
      if (action) return action;
    } else if (bestTrump.type === 'suit' && typeof bestTrump.suit === 'number') {
      const action = validActions.find(a =>
        a.action.type === 'select-trump' &&
        'trump' in a.action &&
        a.action.trump.type === 'suit' &&
        a.action.trump.suit === bestTrump.suit
      );
      if (action) return action;
    }

    return validActions[0]!;
  }

  /**
   * Get stats about the underlying regret table.
   */
  getStats() {
    return this.regretTable.getStats();
  }

  /**
   * Check if the strategy has training data for an info set.
   */
  hasTrainingData(state: GameState): boolean {
    const infoSetKey = computeCountCentricHash(state, state.currentPlayer);
    return this.regretTable.hasNode(infoSetKey);
  }
}

/**
 * Create a fallback strategy that uses MCCFR when available,
 * random otherwise. Simple version without Monte Carlo fallback.
 */
export class HybridMCCFRStrategy implements AIStrategy {
  private mccfrStrategy: MCCFRStrategy | null;
  private rng: RandomGenerator;

  constructor(
    regretTable: RegretTable | null,
    rng: RandomGenerator = defaultRng
  ) {
    this.mccfrStrategy = regretTable ? new MCCFRStrategy(regretTable) : null;
    this.rng = rng;
  }

  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length <= 1) {
      return validActions[0]!;
    }

    // For playing phase, check if we have MCCFR training data
    if (state.phase === 'playing' && this.mccfrStrategy) {
      if (this.mccfrStrategy.hasTrainingData(state)) {
        return this.mccfrStrategy.chooseAction(state, validActions);
      }
      // Fallback: random for untrained states
      return this.randomPlayAction(validActions);
    }

    // Use MCCFR strategy for bidding/trump (uses heuristics internally)
    if (this.mccfrStrategy) {
      return this.mccfrStrategy.chooseAction(state, validActions);
    }

    return validActions[0]!;
  }

  private randomPlayAction(validActions: ValidAction[]): ValidAction {
    const playActions = validActions.filter(a => a.action.type === 'play');
    if (playActions.length === 0) {
      return validActions[0]!;
    }
    const idx = Math.floor(this.rng.random() * playActions.length);
    return playActions[idx]!;
  }
}
