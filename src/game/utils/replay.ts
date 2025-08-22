import { createInitialState, decompressActionId, getNextStates } from '../index';
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
  const base64Data = extractBase64FromUrl(url);
  const { seed, actions } = decodeUrlData(base64Data);
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

  // If focusHand is set, skip to that hand
  if (options.focusHand) {
    for (let i = 0; i < actions.length; i++) {
      const actionId = actions[i];
      if (!actionId) continue;
      const decompressedId = decompressActionId(actionId);
      
      if (decompressedId === 'score-hand') {
        currentHand++;
        if (currentHand === options.focusHand) {
          handStartAction = i + 1;
          // Fast-forward to this point
          for (let j = 0; j <= i; j++) {
            const transitions = getNextStates(currentState);
            const actionJ = actions[j];
            const transition = actionJ ? transitions.find(t => t.id === decompressActionId(actionJ)) : undefined;
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
    if (options.focusHand && currentActionForCheck && decompressActionId(currentActionForCheck) === 'score-hand' && i > handStartAction) {
      break;
    }

    const actionId = actions[i];
    if (!actionId) continue;
    const decompressedId = decompressActionId(actionId);
    const availableTransitions = getNextStates(currentState);
    const matchingTransition = availableTransitions.find(t => t.id === decompressedId);

    if (!matchingTransition) {
      const availableIds = availableTransitions.map(t => t.id).join(', ');
      const error = `Action ${i}: Invalid action "${decompressedId}" (from compressed: "${actionId}"). ` +
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
      console.log(`${i}: ${decompressedId}${scoreChange}${phaseChange}`);
    } else if (shouldShow && !options.showTricks) {
      console.log(`Action ${i}: ${decompressedId} (Phase: ${prevPhase} → ${currentState.phase})`);
    }

    // Show tricks if requested
    if (options.showTricks && decompressedId === 'complete-trick') {
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

/**
 * Extract base64 parameter from URL
 */
function extractBase64FromUrl(url: string): string {
  let urlObj: URL;
  
  if (url.includes('://') || url.startsWith('localhost')) {
    urlObj = new URL(url.startsWith('localhost') ? 'http://' + url : url);
  } else {
    // Assume it's already base64
    return url;
  }
  
  const base64Data = urlObj.searchParams.get('d');
  if (!base64Data) {
    throw new Error('No "d" parameter found in URL');
  }
  
  return base64Data;
}

/**
 * Decode base64 URL data into seed and actions
 */
function decodeUrlData(base64Data: string): { seed: number; actions: string[] } {
  const decoded = Buffer.from(base64Data, 'base64').toString('utf-8');
  const urlData = JSON.parse(decoded);
  
  const seed = urlData.s?.s;
  const actions = urlData.a?.map((a: { i: string }) => a.i) || [];
  
  if (!seed) {
    throw new Error('No seed found in URL data');
  }
  
  return { seed, actions };
}