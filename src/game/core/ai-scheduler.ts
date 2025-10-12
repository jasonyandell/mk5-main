import type { GameState, StateTransition } from '../types';
import { BeginnerAIStrategy, RandomAIStrategy } from '../ai/strategies';
import type { AIStrategy } from '../ai/types';

// Strategy instances - reused for pure functions
const strategies = {
  beginner: new BeginnerAIStrategy(),
  random: new RandomAIStrategy()
};

// Global AI speed profile
let speedProfile: 'instant' | 'fast' | 'normal' | 'slow' = 'normal';
export function setAISpeedProfile(profile: 'instant' | 'fast' | 'normal' | 'slow'): void {
  speedProfile = profile;
}
export function getAISpeedProfile(): 'instant' | 'fast' | 'normal' | 'slow' {
  return speedProfile;
}

/**
 * Get the AI strategy for a player (can be extended to support per-player strategies)
 */
function getStrategyForPlayer(_playerId: number, _state: GameState): AIStrategy {
  // For now, all AI players use beginner strategy
  // Could extend to check state.playerStrategies or similar
  return strategies.beginner;
}

/**
 * Pure function - Select AI action for a player
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

/**
 * Pure function - Calculate delay in ticks for an AI action
 */
export function getAIDelayTicks(action: StateTransition): number {
  // Speed profile overrides
  if (speedProfile === 'instant') return 0;
  
  // Consensus actions should be instant
  if (action.action.type === 'agree-complete-trick' || 
      action.action.type === 'agree-score-hand') {
    return 0;
  }

  // Base thinking time in milliseconds
  let base = 500;
  if (action.action.type === 'bid' || action.action.type === 'pass') {
    base = 500;
  } else if (action.action.type === 'select-trump') {
    base = 1000;
  } else if (action.action.type === 'play') {
    base = 500;
  }

  // Apply profile-specific ranges
  let thinkingMs = base;
  if (speedProfile === 'fast') {
    if (action.action.type === 'select-trump') thinkingMs = 500 + Math.random() * 500; // 500-1000ms
    else if (action.action.type === 'play') thinkingMs = 250 + Math.random() * 300; // 250-550ms
    else thinkingMs = 300 + Math.random() * 400; // 300-700ms
  } else if (speedProfile === 'slow') {
    if (action.action.type === 'select-trump') thinkingMs = 1500 + Math.random() * 1500; // 1.5-3.0s
    else if (action.action.type === 'play') thinkingMs = 800 + Math.random() * 800; // 0.8-1.6s
    else thinkingMs = 900 + Math.random() * 1200; // 0.9-2.1s
  } else {
    // normal
    if (action.action.type === 'select-trump') thinkingMs = 1000 + Math.random() * 1000; // 1.0-2.0s
    else if (action.action.type === 'play') thinkingMs = 500 + Math.random() * 500; // 0.5-1.0s
    else thinkingMs = 500 + Math.random() * 1000; // 0.5-1.5s
  }

  // Convert to ticks (assuming ~60fps)
  return Math.ceil(thinkingMs / 16.67);
}

/**
 * Pure function - Skip all AI delays (for compatibility)
 */
export function skipAIDelays(state: GameState): GameState {
  // Now handled by dispatcher.executeAllScheduled()
  return state;
}

// DEPRECATED: tickGame removed - ticking now handled via advance-tick action

/**
 * Pure function - Reset AI scheduling (for navigation/restore)
 */
export function resetAISchedule(state: GameState): GameState {
  // AI schedule reset no longer needed - handled by dispatcher
  return state;
}
