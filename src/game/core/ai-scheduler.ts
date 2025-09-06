import type { GameState, StateTransition } from '../types';
import { BeginnerAIStrategy, RandomAIStrategy } from '../controllers/strategies';
import type { AIStrategy } from '../controllers/types';
import { getNextStates } from './gameEngine';
import { GAME_PHASES } from '../constants';

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
  
  const strategy = getStrategyForPlayer(playerId, state);
  return strategy.chooseAction(state, myTransitions);
}

/**
 * Pure function - Calculate delay in ticks for an AI action
 */
export function getAIDelayTicks(action: StateTransition): number {
  // Speed profile overrides
  if (speedProfile === 'instant') return 0;

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
 * Pure function - Schedule AI decisions for all AI players
 */
export function scheduleAIDecisions(state: GameState): GameState {
  // Don't modify if game is over
  if (state.phase === GAME_PHASES.GAME_END) {
    return state;
  }
  
  const transitions = getNextStates(state);
  const newSchedule = { ...state.aiSchedule };
  
  // Check each player
  for (let playerId = 0; playerId < 4; playerId++) {
    // Skip if human player
    if (state.playerTypes[playerId] === 'human') continue;
    
    // Skip if already has scheduled action
    if (newSchedule[playerId]) continue;
    
    // Try to select an action
    const choice = selectAIAction(state, playerId, transitions);
    if (choice) {
      const delayTicks = getAIDelayTicks(choice);
      newSchedule[playerId] = {
        transition: choice,
        executeAtTick: state.currentTick + delayTicks
      };
    }
  }
  
  // Return new state if schedule changed
  if (Object.keys(newSchedule).length !== Object.keys(state.aiSchedule).length) {
    return { ...state, aiSchedule: newSchedule };
  }
  
  return state;
}

/**
 * Pure function - Execute scheduled AI actions that are ready
 * Returns the state and any action that should be executed
 */
export function executeScheduledAIActions(state: GameState): { state: GameState; action?: StateTransition } {
  // Find actions ready to execute
  const ready: Array<[number, typeof state.aiSchedule[number]]> = [];
  
  for (const [playerId, scheduled] of Object.entries(state.aiSchedule)) {
    if (scheduled.executeAtTick <= state.currentTick) {
      ready.push([Number(playerId), scheduled]);
    }
  }
  
  // Sort by executeAtTick to ensure deterministic order
  ready.sort((a, b) => a[1].executeAtTick - b[1].executeAtTick);
  
  // Return the first ready action to be executed
  if (ready.length > 0) {
    const firstReady = ready[0];
    if (!firstReady) return { state };
    
    const [playerId, scheduled] = firstReady;
    const newSchedule = { ...state.aiSchedule };
    delete newSchedule[playerId];
    
    // Return the state with updated schedule and the action to execute
    return {
      state: {
        ...state,
        aiSchedule: newSchedule
      },
      action: scheduled.transition
    };
  }
  
  return { state };
}

/**
 * Pure function - Skip all AI delays (execute immediately)
 */
export function skipAIDelays(state: GameState): GameState {
  const newSchedule = { ...state.aiSchedule };
  
  // Set all scheduled actions to execute now
  for (const playerId in newSchedule) {
    const scheduled = newSchedule[playerId];
    if (scheduled) {
      newSchedule[playerId] = {
        ...scheduled,
        executeAtTick: state.currentTick
      };
    }
  }
  
  return { ...state, aiSchedule: newSchedule };
}

/**
 * Pure function - Advance game by one tick
 * Returns the state and any action that should be executed
 */
export function tickGame(state: GameState): { state: GameState; action?: StateTransition } {
  // Increment tick
  let newState = { ...state, currentTick: state.currentTick + 1 };
  
  // Schedule new AI decisions
  newState = scheduleAIDecisions(newState);
  
  // Execute ready AI actions
  const result = executeScheduledAIActions(newState);
  
  return result;
}

/**
 * Pure function - Reset AI scheduling (for navigation/restore)
 */
export function resetAISchedule(state: GameState): GameState {
  return {
    ...state,
    aiSchedule: {},
    currentTick: 0
  };
}
