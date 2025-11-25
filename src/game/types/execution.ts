/**
 * Execution Context
 *
 * Unified container for the execution configuration that layers need.
 * This groups together related parameters that always travel together:
 * - layers: Configuration of which game rules are active
 * - rules: Composed implementations derived from layers
 * - getValidActions: Action generator function for action generation
 *
 * By grouping these, we reduce parameter passing and make it clear that
 * layers/rules/getValidActions are execution context, not data.
 */

import type { Layer, GameRules } from '../layers/types';
import type { GameConfig } from './config';
import type { GameState, GameAction } from '../types';
import { composeRules, baseLayer, getLayersByNames, composeGetValidActions } from '../layers';
import { generateStructuralActions } from '../layers/base';

export interface ExecutionContext {
  readonly layers: readonly Layer[];
  readonly rules: GameRules;
  readonly getValidActions: (state: GameState) => GameAction[];
}

/**
 * Create an ExecutionContext from a GameConfig.
 *
 * This is the single composition point for execution configuration:
 * 1. Get enabled layers from config
 * 2. Compose rules via composeRules()
 * 3. Compose getValidActions via function composition
 * 4. Freeze and return immutable context
 *
 * @param config - Game configuration
 * @returns Frozen ExecutionContext ready for use
 */
export function createExecutionContext(config: GameConfig): ExecutionContext {
  // Get enabled layers
  const enabledLayerNames = config.enabledLayers ?? [];
  const enabledLayers = enabledLayerNames.length > 0
    ? getLayersByNames(enabledLayerNames)
    : [];

  // Compose all layers: base + enabled layers
  const layers = [baseLayer, ...enabledLayers];

  // Compose rules
  const rules = composeRules(layers);

  // Compose getValidActions: base structural actions + layer transformations
  const base = (state: GameState) => generateStructuralActions(state, rules);
  const getValidActions = composeGetValidActions(layers, base);

  // Freeze and return immutable context
  return Object.freeze({
    layers: Object.freeze(layers),
    rules,
    getValidActions
  });
}
