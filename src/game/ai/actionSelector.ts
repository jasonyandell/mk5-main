import type { GameState } from '../types';
import type { ValidAction } from '../../multiplayer/types';
import { BeginnerAIStrategy, RandomAIStrategy } from './strategies';
import { IntermediateAIStrategy } from './strategies/intermediate';
import type { AIStrategy } from './types';

/** Available AI strategy types */
export type AIStrategyType = 'beginner' | 'intermediate' | 'random';

// Strategy instances - reused for pure functions
// Note: IntermediateAIStrategy creates ExecutionContext lazily from state
const strategies: Record<AIStrategyType, AIStrategy> = {
  beginner: new BeginnerAIStrategy(),
  intermediate: new IntermediateAIStrategy(),
  random: new RandomAIStrategy()
};

// Current default strategy (can be changed for testing/configuration)
let defaultStrategy: AIStrategyType = 'beginner';

/**
 * Set the default AI strategy for all AI players.
 * Useful for testing or global configuration.
 */
export function setDefaultAIStrategy(strategy: AIStrategyType): void {
  defaultStrategy = strategy;
}

/**
 * Get the current default AI strategy.
 */
export function getDefaultAIStrategy(): AIStrategyType {
  return defaultStrategy;
}

/**
 * Get a strategy instance by type.
 */
export function getStrategy(type: AIStrategyType): AIStrategy {
  return strategies[type];
}

/**
 * Get the AI strategy for a player (can be extended to support per-player strategies)
 */
function getStrategyForPlayer(_playerId: number, _state: GameState): AIStrategy {
  // Use the default strategy
  // Could extend to check state-based per-player strategies
  return strategies[defaultStrategy];
}

/**
 * Pure function - Select AI action for a player.
 * This belongs in the AI layer, not core, because it makes AI decisions.
 */
export function selectAIAction(
  state: GameState,
  playerId: number,
  validActions: ValidAction[]
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

  // AI players should immediately agree to consensus actions
  const consensusAction = myActions.find(va =>
    va.action.type === 'agree-complete-trick' ||
    va.action.type === 'agree-score-hand'
  );
  if (consensusAction) {
    return consensusAction;
  }

  const strategy = getStrategyForPlayer(playerId, state);
  return strategy.chooseAction(state, myActions);
}
