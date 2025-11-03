// Single transformer surface - action transformers only transform the action state machine
import type { GameState, GameAction } from '../types';
import type { ActionTransformerConfig } from '../types/config';

// State machine: produces valid actions from state
export type StateMachine = (state: GameState) => GameAction[];

// ActionTransformer: transforms a state machine
export type ActionTransformer = (base: StateMachine) => StateMachine;

// Factory for parameterized action transformers
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type ActionTransformerFactory = (config?: any) => ActionTransformer;

export type { ActionTransformerConfig };
