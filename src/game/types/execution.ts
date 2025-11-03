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

export interface ExecutionContext {
  readonly ruleSets: readonly GameRuleSet[];
  readonly rules: GameRules;
  readonly getValidActions: StateMachine;
}
