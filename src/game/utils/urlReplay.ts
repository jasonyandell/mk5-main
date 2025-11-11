import { createInitialState, getNextStates } from '../index';
import { decodeGameUrl } from '../core/url-compression';
import { createExecutionContext } from '../types/execution';
import type { GameState } from '../types';

export interface ReplayOptions {
  stopAt?: number;
  verbose?: boolean;
  actionRangeStart?: number;
  actionRangeEnd?: number;
  showTricks?: boolean;
  focusHand?: number;
  compact?: boolean;
}

export interface ReplayResult {
  state: GameState;
  actionCount: number;
  errors?: string[];
}

/**
 * Replay game actions from a URL containing encoded state
 */
export function replayFromUrl(url: string, options: ReplayOptions = {}): ReplayResult {
  // Extract query string from URL
  const queryString = url.includes('?') ? url.split('?')[1] : url;
  const { seed, actions } = decodeGameUrl(queryString || '');
  return replayActions(seed, actions, options);
}

/**
 * Replay a sequence of actions from a seed
 */
export function replayActions(
  seed: number,
  actions: string[],
  options: ReplayOptions = {}
): ReplayResult {
  let currentState = createInitialState({ shuffleSeed: seed });
  const errors: string[] = [];
  let actionCount = 0;
  let currentHand = 1;
  let handStartAction = 0;

  // Create execution context once for all state lookups
  const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });

  // If focusHand is set, skip to that hand
  if (options.focusHand) {
    for (let i = 0; i < actions.length; i++) {
      const actionId = actions[i];
      if (!actionId) continue;

      if (actionId === 'score-hand') {
        currentHand++;
        if (currentHand === options.focusHand) {
          handStartAction = i + 1;
          // Fast-forward to this point
          for (let j = 0; j <= i; j++) {
            const transitions = getNextStates(currentState, ctx);
            const actionJ = actions[j];
            const transition = actionJ ? transitions.find(t => t.id === actionJ) : undefined;
            if (transition) currentState = transition.newState;
          }
          break;
        }
      }
    }
  }

  const startIdx = options.focusHand ? handStartAction : 0;

  for (let i = startIdx; i < actions.length; i++) {
    if (options.stopAt !== undefined && i >= options.stopAt) {
      break;
    }

    // Stop if we're past the focused hand
    const currentActionForCheck = actions[i];
    if (options.focusHand && currentActionForCheck && currentActionForCheck === 'score-hand' && i > handStartAction) {
      break;
    }

    const actionId = actions[i];
    if (!actionId) continue;
    const availableTransitions = getNextStates(currentState, ctx);
    const matchingTransition = availableTransitions.find(t => t.id === actionId);

    if (!matchingTransition) {
      const availableIds = availableTransitions.map(t => t.id).join(', ');
      const error = `Action ${i}: Invalid action "${actionId}". ` +
                   `Phase: ${currentState.phase}, Available: [${availableIds}]`;
      errors.push(error);
      
      if (options.verbose || (options.actionRangeStart !== undefined && i >= options.actionRangeStart && i <= (options.actionRangeEnd || i))) {
        console.error(error);
      }
      break;
    }

    const prevPhase = currentState.phase;
    const prevScores = [...currentState.teamScores];
    currentState = matchingTransition.newState;
    actionCount++;

    // Determine if we should show this action
    const inRange = !options.actionRangeStart || (i >= options.actionRangeStart && i <= (options.actionRangeEnd || i));
    const shouldShow = options.verbose || inRange;

    if (options.compact && shouldShow) {
      // Compact one-line format
      const scoreChange = (prevScores[0] !== currentState.teamScores[0] || prevScores[1] !== currentState.teamScores[1])
        ? ` [${prevScores}] → [${currentState.teamScores}]`
        : '';
      const phaseChange = prevPhase !== currentState.phase ? ` ${currentState.phase.toUpperCase()}` : '';
      console.log(`${i}: ${actionId}${scoreChange}${phaseChange}`);
    } else if (shouldShow && !options.showTricks) {
      console.log(`Action ${i}: ${actionId} (Phase: ${prevPhase} → ${currentState.phase}`);
    }

    // Show tricks if requested
    if (options.showTricks && actionId === 'complete-trick') {
      const trickNum = currentState.tricks.length;
      const trick = currentState.tricks[trickNum - 1];
      if (trick) {
        const winner = trick.winner !== undefined ? currentState.players[trick.winner] : undefined;
      if (!winner) continue;
        console.log(`Trick ${trickNum}: Won by P${trick.winner} (team ${winner.teamId}) → ${trick.points + 1} points`);
      }
    }
  }

  return {
    state: currentState,
    actionCount,
    errors: errors.length > 0 ? errors : []
  };
}

/**
 * Replay actions up to a specific action index
 */
export function replayToAction(
  seed: number, 
  actions: string[], 
  stopAt: number
): GameState {
  const result = replayActions(seed, actions, { stopAt });
  return result.state;
}

