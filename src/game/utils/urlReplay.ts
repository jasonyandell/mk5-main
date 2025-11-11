import { decodeGameUrl } from '../core/url-compression';
import { HeadlessRoom } from '../../server/HeadlessRoom';
import type { GameState } from '../types';
import { getNextStates } from '../core/state';
import { createExecutionContext } from '../types/execution';

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
  // Create HeadlessRoom for replay
  const room = new HeadlessRoom(
    { playerTypes: ['human', 'human', 'human', 'human'] },
    seed
  );

  // Create execution context for transition matching
  const ctx = createExecutionContext({ playerTypes: ['human', 'human', 'human', 'human'] });

  const errors: string[] = [];
  let actionCount = 0;
  let currentHand = 1;
  let handStartAction = 0;

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
            const actionJ = actions[j];
            if (!actionJ) continue;

            // Get current state and transitions
            const currentState = room.getState();
            const transitions = getNextStates(currentState, ctx);
            const matchingTransition = transitions.find(t => t.id === actionJ);

            if (matchingTransition) {
              // Determine which player should execute
              const executingPlayer = 'player' in matchingTransition.action
                ? matchingTransition.action.player
                : 0;
              try {
                room.executeAction(executingPlayer, matchingTransition.action);
              } catch {
                // Skip errors during fast-forward
              }
            }
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

    const currentState = room.getState();

    // Stop if we're past the focused hand
    const currentActionForCheck = actions[i];
    if (options.focusHand && currentActionForCheck && currentActionForCheck === 'score-hand' && i > handStartAction) {
      break;
    }

    const actionId = actions[i];
    if (!actionId) continue;

    // Get available transitions (they have IDs)
    const transitions = getNextStates(currentState, ctx);
    const matchingTransition = transitions.find(t => t.id === actionId);

    if (!matchingTransition) {
      const availableIds = transitions.map(t => t.id).join(', ');
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

    // Execute the action
    try {
      const executingPlayer = 'player' in matchingTransition.action
        ? matchingTransition.action.player
        : 0;
      room.executeAction(executingPlayer, matchingTransition.action);
      actionCount++;
    } catch (e) {
      const error = `Action ${i}: Execution failed for "${actionId}": ${e instanceof Error ? e.message : 'Unknown error'}`;
      errors.push(error);

      if (options.verbose || (options.actionRangeStart !== undefined && i >= options.actionRangeStart && i <= (options.actionRangeEnd || i))) {
        console.error(error);
      }
      break;
    }

    const newState = room.getState();

    // Determine if we should show this action
    const inRange = !options.actionRangeStart || (i >= options.actionRangeStart && i <= (options.actionRangeEnd || i));
    const shouldShow = options.verbose || inRange;

    if (options.compact && shouldShow) {
      // Compact one-line format
      const scoreChange = (prevScores[0] !== newState.teamScores[0] || prevScores[1] !== newState.teamScores[1])
        ? ` [${prevScores}] → [${newState.teamScores}]`
        : '';
      const phaseChange = prevPhase !== newState.phase ? ` ${newState.phase.toUpperCase()}` : '';
      console.log(`${i}: ${actionId}${scoreChange}${phaseChange}`);
    } else if (shouldShow && !options.showTricks) {
      console.log(`Action ${i}: ${actionId} (Phase: ${prevPhase} → ${newState.phase}`);
    }

    // Show tricks if requested
    if (options.showTricks && actionId === 'complete-trick') {
      const trickNum = newState.tricks.length;
      const trick = newState.tricks[trickNum - 1];
      if (trick) {
        const winner = trick.winner !== undefined ? newState.players[trick.winner] : undefined;
        if (!winner) continue;
        console.log(`Trick ${trickNum}: Won by P${trick.winner} (team ${winner.teamId}) → ${trick.points + 1} points`);
      }
    }
  }

  return {
    state: room.getState(),
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

