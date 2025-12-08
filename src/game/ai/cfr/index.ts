/**
 * MCCFR (Monte Carlo Counterfactual Regret Minimization) Module
 *
 * Exports for training and using MCCFR strategies in Texas 42.
 *
 * Usage:
 * ```typescript
 * // Training
 * import { MCCFRTrainer } from './cfr';
 * const trainer = new MCCFRTrainer({ iterations: 100000 });
 * await trainer.train();
 * const serialized = trainer.serialize();
 *
 * // Using trained strategy
 * import { MCCFRStrategy } from './cfr';
 * const strategy = MCCFRStrategy.fromSerialized(serialized);
 * const action = strategy.chooseAction(state, validActions);
 * ```
 */

// Types
export type {
  InfoSetKey,
  ActionKey,
  CFRNode,
  MCCFRConfig,
  SerializedStrategy,
  TrainingResult,
  ActionProbabilities
} from './types';

// Core classes
export { RegretTable } from './regret-table';
export { MCCFRTrainer } from './mccfr-trainer';
export { MCCFRStrategy, HybridMCCFRStrategy } from './mccfr-strategy';
export type { MCCFRStrategyConfig } from './mccfr-strategy';

// Action utilities
export {
  actionToKey,
  getActionFromKey,
  getActionKeys,
  sampleAction,
  selectBestAction,
  isPlayAction,
  getDominoId
} from './action-abstraction';
