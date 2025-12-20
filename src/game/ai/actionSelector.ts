import type { GameState } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import { BeginnerAIStrategy, RandomAIStrategy } from './strategies';
import type { AIStrategy } from './types';
import type { RandomGenerator } from './hand-sampler';
import type { MonteCarloConfig } from './monte-carlo';

/** Available AI strategy types */
export type AIStrategyType = 'beginner' | 'random';

/**
 * Configuration for AI strategy - all dependencies injected, no globals.
 */
export interface AIStrategyConfig {
  type: AIStrategyType;
  /** RNG for random strategy (defaults to Math.random) */
  rng?: RandomGenerator;
  /** Monte Carlo config for beginner strategy */
  monteCarloConfig?: Partial<MonteCarloConfig>;
}

/**
 * Create a strategy instance from config.
 * Pure function - no global state.
 */
export function createStrategy(config: AIStrategyConfig): AIStrategy {
  switch (config.type) {
    case 'random':
      return new RandomAIStrategy(config.rng);
    case 'beginner':
      return new BeginnerAIStrategy(config.monteCarloConfig);
  }
}

/**
 * Pure function - Select AI action for a player.
 * All configuration is passed explicitly - no global state.
 */
export function selectAIAction(
  state: GameState,
  playerId: number,
  validActions: ValidAction[],
  config: AIStrategyConfig = { type: 'beginner' }
): ValidAction | null {
  // Filter to only this player's actions
  const myActions = validActions.filter(va => {
    // Actions without a player field are available to everyone
    if (!('player' in va.action)) {
      return true;
    }
    // Actions with a player field are only for that player
    return va.action.player === playerId;
  });

  if (myActions.length === 0) return null;

  const strategy = createStrategy(config);
  return strategy.chooseAction(state, myActions);
}
