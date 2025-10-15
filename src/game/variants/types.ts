// Single transformer surface - variants only transform the action state machine
import type { GameState, GameAction } from '../types';

// State machine: produces valid actions from state
export type StateMachine = (state: GameState) => GameAction[];

// Variant: transforms a state machine
export type Variant = (base: StateMachine) => StateMachine;

// Factory for parameterized variants
export type VariantFactory = (config?: any) => Variant;

// Serializable config (stored in GameState)
export interface VariantConfig {
  type: string;
  config?: Record<string, any>;
}
