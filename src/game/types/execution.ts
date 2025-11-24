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
import { composeRules, baseRuleSet, getLayersByNames, composeActionGenerators } from '../layers';
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
 * 2. Get action transformer layers (treated as Layers now)
 * 3. Compose rules via composeRules()
 * 4. Compose action generators via function composition (f(g(h(x))))
 *    - base: generateStructuralActions (pass, redeal, consensus, trump, plays)
 *    - Layers: add domain actions (bids) + annotate/script (autoExecute, hints)
 * 5. Freeze and return immutable context
 *
 * @param config - Game configuration
 * @returns Frozen ExecutionContext ready for use
 */
export function createExecutionContext(config: GameConfig): ExecutionContext {
  // Get action transformer configs (now treated as Layer names)
  const actionTransformerConfigs = config.actionTransformers ?? [];
  const transformerLayers = actionTransformerConfigs.length > 0
    ? getLayersByNames(actionTransformerConfigs.map(t => t.type))
    : [];

  // Get enabled layers
  const enabledLayerNames = config.enabledLayers ?? [];
  const enabledLayers = enabledLayerNames.length > 0
    ? getLayersByNames(enabledLayerNames)
    : [];

  // Compose all layers: base + enabled layers + transformer layers
  const layers = [baseRuleSet, ...enabledLayers, ...transformerLayers];

  // Compose rules via composeRules()
  const rules = composeRules(layers);

  // Compose action generators via function composition (not eager evaluation!)
  // Base: structural actions only
  const base = (state: GameState) => generateStructuralActions(state, rules);

  // Layers: add domain actions (bids) + annotate/script (autoExecute, hints)
  const getValidActions = composeActionGenerators(layers, base);

  // Freeze and return immutable context
  return Object.freeze({
    layers: Object.freeze(layers),
    rules,
    getValidActions
  });
}
