/**
 * Intermediate AI Strategy - Monte Carlo based play decisions.
 *
 * Uses Monte Carlo simulation to evaluate play decisions, while delegating
 * bidding and trump selection to the beginner strategy.
 *
 * Design principles:
 * - Same protocol as humans (no privileged state access)
 * - Layer-agnostic (uses ExecutionContext for all rule queries)
 * - Partnership-aware (evaluates team outcomes, not individual tricks)
 * - Configurable simulation budget
 */

import type { AIStrategy } from '../types';
import type { GameState } from '../../types';
import type { ValidAction } from '../../../multiplayer/types';
import type { ExecutionContext } from '../../types/execution';
import { BeginnerAIStrategy } from '../strategies';
import { buildConstraints } from '../constraint-tracker';
import { selectBestPlay, type MonteCarloConfig } from '../monte-carlo';
import { createExecutionContext } from '../../types/execution';

/** Default configuration for Monte Carlo evaluation */
const DEFAULT_CONFIG: MonteCarloConfig = {
  simulations: 50
};

/**
 * Intermediate AI Strategy
 *
 * Uses Monte Carlo simulation for play decisions, delegating other
 * phases to the beginner strategy.
 */
export class IntermediateAIStrategy implements AIStrategy {
  private readonly beginner: BeginnerAIStrategy;
  private readonly config: MonteCarloConfig;
  private ctx: ExecutionContext | null = null;

  constructor(config: Partial<MonteCarloConfig> = {}) {
    this.beginner = new BeginnerAIStrategy();
    this.config = { ...DEFAULT_CONFIG, ...config };
  }

  /**
   * Choose an action from available valid actions.
   *
   * For play phase: Use Monte Carlo evaluation
   * For other phases: Delegate to beginner strategy
   */
  chooseAction(state: GameState, validActions: ValidAction[]): ValidAction {
    if (validActions.length === 0) {
      throw new Error('IntermediateAIStrategy: No valid actions available');
    }

    // Prioritize consensus actions (same as beginner)
    const consensusAction = validActions.find(va =>
      va.action.type === 'agree-complete-trick' ||
      va.action.type === 'agree-score-hand'
    );
    if (consensusAction) {
      return consensusAction;
    }

    // Non-play phases: delegate to beginner
    if (state.phase !== 'playing') {
      return this.beginner.chooseAction(state, validActions);
    }

    // Play phase: Monte Carlo evaluation
    return this.choosePlayAction(state, validActions);
  }

  /**
   * Use Monte Carlo simulation to choose the best play.
   */
  private choosePlayAction(state: GameState, validActions: ValidAction[]): ValidAction {
    // Filter to play actions only
    const playActions = validActions.filter(va => va.action.type === 'play');

    if (playActions.length === 0) {
      // No play actions - might be complete-trick or similar
      // Fall back to beginner
      return this.beginner.chooseAction(state, validActions);
    }

    if (playActions.length === 1) {
      // Only one option - no need for simulation
      return playActions[0]!;
    }

    // Get or create execution context
    const ctx = this.getExecutionContext(state);

    // Build constraints from game history
    const myPlayerIndex = state.currentPlayer;
    const constraints = buildConstraints(state, myPlayerIndex, ctx.rules);

    // Evaluate and select best play
    const bestPlay = selectBestPlay(
      state,
      playActions,
      myPlayerIndex,
      constraints,
      ctx,
      this.config
    );

    // Return best play, or fall back to beginner if evaluation failed
    return bestPlay ?? this.beginner.chooseAction(state, validActions);
  }

  /**
   * Get or create execution context for the current game configuration.
   *
   * Caches the context since it's derived from config which doesn't change.
   */
  private getExecutionContext(state: GameState): ExecutionContext {
    // Create context from game config
    // In practice, we'd want to cache this, but for now create fresh
    // to ensure we have the right layers
    if (!this.ctx) {
      this.ctx = createExecutionContext(state.initialConfig);
    }
    return this.ctx;
  }
}

/**
 * Create an Intermediate AI strategy with custom configuration.
 *
 * @param simulations Number of simulations per action (default: 50)
 * @param seed Optional random seed for reproducibility
 */
export function createIntermediateStrategy(
  simulations: number = 50,
  seed?: number
): IntermediateAIStrategy {
  const config: Partial<MonteCarloConfig> = { simulations };
  if (seed !== undefined) {
    config.seed = seed;
  }
  return new IntermediateAIStrategy(config);
}
