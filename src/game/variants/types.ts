// Single transformer surface - variants only transform the action state machine
import type { GameState, GameAction } from '../types';
import type { VariantConfig } from '../types/config';

// State machine: produces valid actions from state
export type StateMachine = (state: GameState) => GameAction[];

// Variant: transforms a state machine
export type Variant = (base: StateMachine) => StateMachine;

// Factory for parameterized variants
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type VariantFactory = (config?: any) => Variant;

export type { VariantConfig };
