/**
 * Regret Table for MCCFR
 *
 * Stores cumulative regrets and strategy sums for each information set.
 * Provides strategy computation using regret matching.
 */

import type {
  InfoSetKey,
  ActionKey,
  CFRNode,
  ActionProbabilities,
  SerializedStrategy,
  MCCFRConfig
} from './types';

/**
 * Regret table storing all CFR nodes.
 * Central data structure for MCCFR training.
 */
export class RegretTable {
  /** Map from info set key to CFR node. */
  private nodes: Map<InfoSetKey, CFRNode> = new Map();

  /** Minimum regret value (floor to prevent over-negative regrets). */
  private regretFloor: number;

  constructor(regretFloor: number = -1e9) {
    this.regretFloor = regretFloor;
  }

  /**
   * Get or create a CFR node for an information set.
   */
  getNode(infoSetKey: InfoSetKey): CFRNode {
    let node = this.nodes.get(infoSetKey);
    if (!node) {
      node = {
        regrets: new Map(),
        strategySum: new Map(),
        visitCount: 0
      };
      this.nodes.set(infoSetKey, node);
    }
    return node;
  }

  /**
   * Check if a node exists for an information set.
   */
  hasNode(infoSetKey: InfoSetKey): boolean {
    return this.nodes.has(infoSetKey);
  }

  /**
   * Get the current strategy for an information set using regret matching.
   *
   * Regret matching: probability proportional to positive regret.
   * If all regrets are non-positive, use uniform distribution.
   *
   * @param infoSetKey - Information set identifier
   * @param legalActions - Array of legal action keys at this info set
   * @returns Map of action -> probability
   */
  getStrategy(infoSetKey: InfoSetKey, legalActions: ActionKey[]): ActionProbabilities {
    const node = this.getNode(infoSetKey);
    const strategy: ActionProbabilities = new Map();

    // Calculate sum of positive regrets
    let positiveRegretSum = 0;
    for (const action of legalActions) {
      const regret = node.regrets.get(action) ?? 0;
      if (regret > 0) {
        positiveRegretSum += regret;
      }
    }

    if (positiveRegretSum > 0) {
      // Regret matching: proportional to positive regret
      for (const action of legalActions) {
        const regret = node.regrets.get(action) ?? 0;
        strategy.set(action, regret > 0 ? regret / positiveRegretSum : 0);
      }
    } else {
      // Uniform distribution when all regrets are non-positive
      const uniformProb = 1 / legalActions.length;
      for (const action of legalActions) {
        strategy.set(action, uniformProb);
      }
    }

    return strategy;
  }

  /**
   * Get the average strategy for an information set.
   *
   * The average strategy converges to Nash equilibrium in two-player zero-sum games.
   * Uses cumulative strategy sums normalized to probabilities.
   *
   * @param infoSetKey - Information set identifier
   * @param legalActions - Array of legal action keys
   * @returns Map of action -> probability
   */
  getAverageStrategy(infoSetKey: InfoSetKey, legalActions: ActionKey[]): ActionProbabilities {
    const node = this.nodes.get(infoSetKey);
    const strategy: ActionProbabilities = new Map();

    if (!node) {
      // No training data - use uniform
      const uniformProb = 1 / legalActions.length;
      for (const action of legalActions) {
        strategy.set(action, uniformProb);
      }
      return strategy;
    }

    // Calculate sum of strategy weights
    let strategySum = 0;
    for (const action of legalActions) {
      strategySum += node.strategySum.get(action) ?? 0;
    }

    if (strategySum > 0) {
      // Normalize to probabilities
      for (const action of legalActions) {
        const weight = node.strategySum.get(action) ?? 0;
        strategy.set(action, weight / strategySum);
      }
    } else {
      // No strategy data - use uniform
      const uniformProb = 1 / legalActions.length;
      for (const action of legalActions) {
        strategy.set(action, uniformProb);
      }
    }

    return strategy;
  }

  /**
   * Update regrets for an information set after traversal.
   *
   * @param infoSetKey - Information set identifier
   * @param actionValues - Map of action -> counterfactual value
   * @param expectedValue - Expected value under current strategy
   * @param opponentReachProb - Product of opponent reach probabilities
   */
  updateRegrets(
    infoSetKey: InfoSetKey,
    actionValues: Map<ActionKey, number>,
    expectedValue: number,
    opponentReachProb: number
  ): void {
    const node = this.getNode(infoSetKey);

    for (const [action, value] of actionValues) {
      const regret = (value - expectedValue) * opponentReachProb;
      const currentRegret = node.regrets.get(action) ?? 0;
      const newRegret = Math.max(this.regretFloor, currentRegret + regret);
      node.regrets.set(action, newRegret);
    }
  }

  /**
   * Update strategy sum for an information set.
   * Called during traversal to accumulate strategy weights.
   *
   * @param infoSetKey - Information set identifier
   * @param strategy - Current strategy (action -> probability)
   * @param reachProb - Player's reach probability to this info set
   */
  updateStrategySum(
    infoSetKey: InfoSetKey,
    strategy: ActionProbabilities,
    reachProb: number
  ): void {
    const node = this.getNode(infoSetKey);
    node.visitCount++;

    for (const [action, prob] of strategy) {
      const current = node.strategySum.get(action) ?? 0;
      node.strategySum.set(action, current + prob * reachProb);
    }
  }

  /**
   * Apply discount to all regrets and strategy sums.
   * Used for CFR+ variant to down-weight older iterations.
   *
   * @param discount - Discount factor (0-1)
   */
  applyDiscount(discount: number): void {
    for (const node of this.nodes.values()) {
      for (const [action, regret] of node.regrets) {
        node.regrets.set(action, regret * discount);
      }
      for (const [action, stratSum] of node.strategySum) {
        node.strategySum.set(action, stratSum * discount);
      }
    }
  }

  /**
   * Get number of unique information sets.
   */
  get size(): number {
    return this.nodes.size;
  }

  /**
   * Serialize the regret table to JSON-compatible format.
   */
  serialize(config: MCCFRConfig, iterationsCompleted: number, trainingTimeMs: number): SerializedStrategy {
    const serializedNodes: SerializedStrategy['nodes'] = [];

    for (const [key, node] of this.nodes) {
      serializedNodes.push({
        key,
        regrets: Array.from(node.regrets.entries()),
        strategySum: Array.from(node.strategySum.entries()),
        visitCount: node.visitCount
      });
    }

    // Remove non-serializable callback
    const { onProgress, ...serializableConfig } = config;

    return {
      version: 1,
      config: serializableConfig,
      iterationsCompleted,
      nodes: serializedNodes,
      trainedAt: new Date().toISOString(),
      trainingTimeMs
    };
  }

  /**
   * Load a serialized strategy into this regret table.
   */
  static deserialize(data: SerializedStrategy): RegretTable {
    const table = new RegretTable(data.config.regretFloor);

    for (const nodeData of data.nodes) {
      const node: CFRNode = {
        regrets: new Map(nodeData.regrets),
        strategySum: new Map(nodeData.strategySum),
        visitCount: nodeData.visitCount
      };
      table.nodes.set(nodeData.key, node);
    }

    return table;
  }

  /**
   * Clear all data (for testing).
   */
  clear(): void {
    this.nodes.clear();
  }

  /**
   * Get all info set keys (for debugging/analysis).
   */
  getInfoSetKeys(): InfoSetKey[] {
    return Array.from(this.nodes.keys());
  }

  /**
   * Get stats about the regret table.
   */
  getStats(): {
    nodeCount: number;
    totalVisits: number;
    avgActionsPerNode: number;
  } {
    let totalVisits = 0;
    let totalActions = 0;

    for (const node of this.nodes.values()) {
      totalVisits += node.visitCount;
      totalActions += node.regrets.size;
    }

    return {
      nodeCount: this.nodes.size,
      totalVisits,
      avgActionsPerNode: this.nodes.size > 0 ? totalActions / this.nodes.size : 0
    };
  }
}
