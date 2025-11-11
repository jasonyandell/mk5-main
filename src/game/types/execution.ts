/**
 * Execution Context
 *
 * Unified container for the execution configuration that rule sets need.
 * This groups together related parameters that always travel together:
 * - ruleSets: Configuration of which game rules are active
 * - rules: Composed implementations derived from rule sets
 * - getValidActions: Action transformer function for action generation
 *
 * By grouping these, we reduce parameter passing and make it clear that
 * ruleSets/rules/getValidActions are execution context, not data.
 */

import type { GameRuleSet, GameRules } from '../rulesets/types';
import type { StateMachine } from '../action-transformers/types';
import type { GameConfig } from './config';
import type { GameState } from '../types';
import { composeRules, baseRuleSet, getRuleSetsByNames, composeActionGenerators } from '../rulesets';
import { generateStructuralActions } from '../rulesets/base';
import { applyActionTransformers } from '../action-transformers/registry';

export interface ExecutionContext {
  readonly ruleSets: readonly GameRuleSet[];
  readonly rules: GameRules;
  readonly getValidActions: StateMachine;
}

/**
 * Create an ExecutionContext from a GameConfig.
 *
 * This is the single composition point for execution configuration:
 * 1. Get enabled rulesets from config
 * 2. Compose rules via composeRules()
 * 3. Compose action generators via function composition (f(g(h(x))))
 *    - base: generateStructuralActions (pass, redeal, consensus, trump, plays)
 *    - RuleSets: add domain actions (bids) via composeActionGenerators
 *    - ActionTransformers: annotate/script (autoExecute, hints) via applyActionTransformers
 * 4. Freeze and return immutable context
 *
 * @param config - Game configuration
 * @returns Frozen ExecutionContext ready for use
 */
export function createExecutionContext(config: GameConfig): ExecutionContext {
  // Get action transformer configs
  const actionTransformerConfigs = config.actionTransformers ?? [];

  // Get enabled rulesets
  const enabledRuleSetNames = config.enabledRuleSets ?? [];
  const enabledRuleSets = enabledRuleSetNames.length > 0
    ? getRuleSetsByNames(enabledRuleSetNames)
    : [];

  const ruleSets = [baseRuleSet, ...enabledRuleSets];

  // Compose rules via composeRules()
  const rules = composeRules(ruleSets);

  // Compose action generators via function composition (not eager evaluation!)
  // 1. Base: structural actions only
  const base: StateMachine = (state: GameState) => generateStructuralActions(state, rules);

  // 2. RuleSets: add domain actions (bids)
  const withRuleSets = composeActionGenerators(ruleSets, base);

  // 3. ActionTransformers: annotate/script actions
  const getValidActions = applyActionTransformers(withRuleSets, actionTransformerConfigs);

  // Freeze and return immutable context
  return Object.freeze({
    ruleSets: Object.freeze(ruleSets),
    rules,
    getValidActions
  });
}
