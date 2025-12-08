/**
 * Action Abstraction for MCCFR
 *
 * Maps game actions to abstract action keys and back.
 * For the trick-taking phase, actions are play actions identified by domino ID.
 */

// Note: GameAction and Domino imports are available for future use
import type { ValidAction } from '../../../multiplayer/types';
import type { ActionKey, ActionProbabilities } from './types';
import type { RandomGenerator } from '../hand-sampler';

/**
 * Convert a play action to an action key.
 * For play actions, the key is the domino ID.
 *
 * @param action - Valid action from the game
 * @returns Action key (domino ID for plays)
 */
export function actionToKey(action: ValidAction): ActionKey {
  if (action.action.type !== 'play') {
    throw new Error(`actionToKey only supports play actions, got: ${action.action.type}`);
  }
  return action.action.dominoId;
}

/**
 * Find the valid action matching an action key.
 *
 * @param actionKey - Action key (domino ID)
 * @param validActions - Array of valid actions
 * @returns Matching ValidAction or undefined
 */
export function getActionFromKey(
  actionKey: ActionKey,
  validActions: ValidAction[]
): ValidAction | undefined {
  return validActions.find(va => {
    if (va.action.type !== 'play') return false;
    return va.action.dominoId === actionKey;
  });
}

/**
 * Get all action keys from valid actions.
 *
 * @param validActions - Array of valid actions
 * @returns Array of action keys
 */
export function getActionKeys(validActions: ValidAction[]): ActionKey[] {
  return validActions
    .filter(va => va.action.type === 'play')
    .map(va => (va.action as { type: 'play'; dominoId: string }).dominoId);
}

/**
 * Sample an action according to a probability distribution.
 *
 * @param strategy - Map of action key -> probability
 * @param validActions - Array of valid actions
 * @param rng - Random number generator
 * @returns Selected ValidAction
 */
export function sampleAction(
  strategy: ActionProbabilities,
  validActions: ValidAction[],
  rng: RandomGenerator
): ValidAction {
  // Filter to play actions only
  const playActions = validActions.filter(va => va.action.type === 'play');

  if (playActions.length === 0) {
    throw new Error('No play actions available to sample');
  }

  if (playActions.length === 1) {
    return playActions[0]!;
  }

  // Build cumulative distribution
  const actions: ValidAction[] = [];
  const cumProbs: number[] = [];
  let cumSum = 0;

  for (const action of playActions) {
    const key = actionToKey(action);
    const prob = strategy.get(key) ?? 0;
    cumSum += prob;
    actions.push(action);
    cumProbs.push(cumSum);
  }

  // Normalize if needed (shouldn't be necessary but safety check)
  if (cumSum === 0) {
    // Uniform fallback
    const idx = Math.floor(rng.random() * playActions.length);
    return playActions[idx]!;
  }

  // Sample according to distribution
  const r = rng.random() * cumSum;
  for (let i = 0; i < cumProbs.length; i++) {
    if (r <= cumProbs[i]!) {
      return actions[i]!;
    }
  }

  // Fallback (shouldn't happen)
  return playActions[playActions.length - 1]!;
}

/**
 * Select the best action according to average strategy (greedy).
 *
 * @param strategy - Map of action key -> probability
 * @param validActions - Array of valid actions
 * @returns ValidAction with highest probability
 */
export function selectBestAction(
  strategy: ActionProbabilities,
  validActions: ValidAction[]
): ValidAction {
  const playActions = validActions.filter(va => va.action.type === 'play');

  if (playActions.length === 0) {
    throw new Error('No play actions available');
  }

  if (playActions.length === 1) {
    return playActions[0]!;
  }

  let bestAction = playActions[0]!;
  let bestProb = strategy.get(actionToKey(bestAction)) ?? 0;

  for (let i = 1; i < playActions.length; i++) {
    const action = playActions[i]!;
    const prob = strategy.get(actionToKey(action)) ?? 0;
    if (prob > bestProb) {
      bestProb = prob;
      bestAction = action;
    }
  }

  return bestAction;
}

/**
 * Check if an action is a play action.
 */
export function isPlayAction(action: ValidAction): boolean {
  return action.action.type === 'play';
}

/**
 * Get domino from a play action.
 */
export function getDominoId(action: ValidAction): string {
  if (action.action.type !== 'play') {
    throw new Error('Not a play action');
  }
  return action.action.dominoId;
}
