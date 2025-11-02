/**
 * Execution Context
 *
 * Unified container for the execution configuration that layers and rules need.
 * This groups together related parameters that always travel together:
 * - layers: Configuration of which game rules are active
 * - rules: Composed implementations derived from layers
 * - getValidActions: State machine function for action generation
 *
 * By grouping these, we reduce parameter passing and make it clear that
 * layers/rules/getValidActions are execution context, not data.
 */

import type { GameLayer, GameRules } from '../layers/types';
import type { StateMachine } from '../variants/types';

export interface ExecutionContext {
  readonly layers: readonly GameLayer[];
  readonly rules: GameRules;
  readonly getValidActions: StateMachine;
}
