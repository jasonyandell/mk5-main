import type { GameState, GameAction } from '../types';
import type { GameConfig } from '../types/config';
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
  // Action transformers do not change execution: we simply reapply the recorded actions.
  // We capture the active action transformer configuration so that any caller asking for
  // valid actions after replay can compose the same action transformer pipeline.

  const actionTransformerConfigs = config.variants ?? [];

  // Create initial state with proper optional handling
  let state = createInitialState({
    playerTypes: config.playerTypes,
    ...(config.shuffleSeed !== undefined && { shuffleSeed: config.shuffleSeed }),
    ...(config.theme !== undefined && { theme: config.theme }),
    ...(config.colorOverrides !== undefined && { colorOverrides: config.colorOverrides }),
    ...(actionTransformerConfigs.length ? { variants: actionTransformerConfigs } : {})
  });

  // Add initialConfig to state
  state = {
    ...state,
    initialConfig: {
      ...config,
      ...(actionTransformerConfigs.length ? { variants: actionTransformerConfigs } : {})
    }
  };

  // Replay all actions using base executor
  // (Action transformers only affect valid actions, not execution)
  for (const action of actions) {
    state = baseExecuteAction(state, action);
  }

  return state;
}

/**
 * Create initial state with action transformers applied.
 * Convenience wrapper for replayActions with empty history.
 */
export function createInitialStateWithActionTransformers(config: GameConfig): GameState {
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
