import type { GameState, StateTransition } from '../types';
import { BeginnerAIStrategy, RandomAIStrategy } from './strategies';
import type { AIStrategy } from './types';

// Strategy instances - reused for pure functions
const strategies = {
  beginner: new BeginnerAIStrategy(),
  random: new RandomAIStrategy()
};

/**
 * Get the AI strategy for a player (can be extended to support per-player strategies)
 */
function getStrategyForPlayer(_playerId: number, _state: GameState): AIStrategy {
  // For now, all AI players use beginner strategy
  // Could extend to check state.playerStrategies or similar
  return strategies.beginner;
}

/**
 * Pure function - Select AI action for a player.
 * This belongs in the AI layer, not core, because it makes AI decisions.
 */
export function selectAIAction(
  state: GameState,
  playerId: number,
  availableTransitions: StateTransition[]
): StateTransition | null {
  // Filter to only this player's actions
  const myTransitions = availableTransitions.filter(t => {
    // Actions without a player field are available to everyone
    if (!('player' in t.action)) {
      return true;
    }
    // Actions with a player field are only for that player
    return t.action.player === playerId;
  });

  if (myTransitions.length === 0) return null;

  // AI players should immediately agree to consensus actions
  const consensusAction = myTransitions.find(t =>
    t.action.type === 'agree-complete-trick' ||
    t.action.type === 'agree-score-hand'
  );
  if (consensusAction) {
    return consensusAction;
  }

  const strategy = getStrategyForPlayer(playerId, state);
  return strategy.chooseAction(state, myTransitions);
}
