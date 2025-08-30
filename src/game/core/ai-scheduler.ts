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
  // Get thinking time in milliseconds
  let thinkingMs = 500; // Default
  
  if (action.action.type === 'bid' || action.action.type === 'pass') {
    thinkingMs = 500 + Math.random() * 1000; // 500-1500ms for bidding
  } else if (action.action.type === 'select-trump') {
    thinkingMs = 1000 + Math.random() * 1000; // 1000-2000ms for trump
  } else if (action.action.type === 'play') {
    thinkingMs = 500 + Math.random() * 500; // 500-1000ms for playing
  }
  
  // Convert to ticks (assuming 60fps)
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