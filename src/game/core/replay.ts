import type { GameState, GameAction } from '../types';
import type { GameConfig } from '../../shared/multiplayer/protocol';
import { createInitialState } from './state';
import { executeAction as baseExecuteAction } from './actions';

/**
 * Replay actions from initial config to reconstruct state.
 * This is the FUNDAMENTAL operation for event sourcing.
 *
 * State = replayActions(initialConfig, actionHistory)
 */
export function replayActions(
  config: GameConfig,
  actions: GameAction[]
): GameState {
  // Note: Variants are stored in state.initialConfig but not yet used in replay
  // They will be applied later when needed for getValidActions
  // Currently variants only affect getValidActions, not executeAction

  // Create initial state with proper optional handling
  let state = createInitialState({
    playerTypes: config.playerTypes,
    ...(config.shuffleSeed !== undefined && { shuffleSeed: config.shuffleSeed }),
    ...(config.theme !== undefined && { theme: config.theme }),
    ...(config.colorOverrides !== undefined && { colorOverrides: config.colorOverrides })
  });

  // Add initialConfig to state
  state = {
    ...state,
    initialConfig: config
  };

  // Replay all actions using base executor
  // (Variants only affect valid actions, not execution)
  for (const action of actions) {
    state = baseExecuteAction(state, action);
  }

  return state;
}

/**
 * Create initial state with variants applied.
 * Convenience wrapper for replayActions with empty history.
 */
export function createInitialStateWithVariants(config: GameConfig): GameState {
  return replayActions(config, []);
}

/**
 * Add an action to history and derive new state.
 * This maintains the event sourcing invariant:
 *   newState = replayActions(state.initialConfig, [...state.actionHistory, action])
 *
 * Note: For performance, we don't actually replay from scratch.
 * We execute the single action and append to history.
 */
export function applyActionWithHistory(
  state: GameState,
  action: GameAction
): GameState {
  // Execute action (updates actionHistory internally)
  const newState = baseExecuteAction(state, action);

  // Ensure initialConfig is preserved
  return {
    ...newState,
    initialConfig: state.initialConfig
  };
}
