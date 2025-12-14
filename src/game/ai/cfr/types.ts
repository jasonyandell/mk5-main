/**
 * MCCFR Type Definitions
 *
 * Core types for Monte Carlo Counterfactual Regret Minimization.
 * Uses count-centric abstraction for information set hashing.
 */

/**
 * Information set key - identifies a unique game situation from player's perspective.
 * Uses count-centric abstraction (32.5x compression over canonical states).
 */
export type InfoSetKey = string;

/**
 * Action key - identifies a legal action at an information set.
 * For play actions, this is the domino ID (e.g., "6-4", "5-5").
 */
export type ActionKey = string;

/**
 * CFR node storing regrets and strategy sums for an information set.
 * One node per unique information set encountered during training.
 */
export interface CFRNode {
  /** Cumulative regrets for each action. Updated during traversal. */
  regrets: Map<ActionKey, number>;

  /** Cumulative strategy weights for each action. Used for average strategy. */
  strategySum: Map<ActionKey, number>;

  /** Number of times this info set was visited during training. */
  visitCount: number;
}

/**
 * Configuration for MCCFR training.
 */
export interface MCCFRConfig {
  /** Number of training iterations (games to simulate). */
  iterations: number;

  /** Random seed for reproducibility. */
  seed: number;

  /** Discount factor for older regrets (0-1, 1 = no discount). */
  regretDiscount?: number;

  /** Minimum regret floor (prevents negative regrets from dominating). */
  regretFloor?: number;

  /** Progress callback frequency (every N iterations). */
  progressInterval?: number;

  /** Progress callback function. */
  onProgress?: (iteration: number, totalIterations: number, nodesCount: number) => void;
}

/**
 * Serialized format for persisting trained strategies.
 */
export interface SerializedStrategy {
  /** Version for forward compatibility. */
  version: number;

  /** Training configuration used. */
  config: Omit<MCCFRConfig, 'onProgress'>;

  /** Number of training iterations completed. */
  iterationsCompleted: number;

  /** Serialized nodes: infoSetKey -> { regrets, strategySum, visitCount }. */
  nodes: Array<{
    key: InfoSetKey;
    regrets: Array<[ActionKey, number]>;
    strategySum: Array<[ActionKey, number]>;
    visitCount: number;
  }>;

  /** Training timestamp. */
  trainedAt: string;

  /** Total training time in milliseconds. */
  trainingTimeMs: number;

  /** Seed range start (for partitioned training). */
  seedStart?: number;

  /** Seed range end (for partitioned training). */
  seedEnd?: number;

  /** Whether this is an incomplete checkpoint (vs final output). */
  isCheckpoint?: boolean;
}

/**
 * Result of training run.
 */
export interface TrainingResult {
  /** Number of iterations completed. */
  iterations: number;

  /** Number of unique information sets discovered. */
  infoSetCount: number;

  /** Total training time in milliseconds. */
  trainingTimeMs: number;

  /** Average iterations per second. */
  iterationsPerSecond: number;
}

/**
 * Strategy profile for a single information set.
 * Maps actions to their probabilities.
 */
export type ActionProbabilities = Map<ActionKey, number>;
