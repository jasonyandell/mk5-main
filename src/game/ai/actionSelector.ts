import type { GameState } from '../types';
import type { ValidAction } from '../../multiplayer/types';
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
