// Pure lookup - no registration, no side effects
import type { ActionTransformerFactory, ActionTransformer, ActionTransformerConfig, StateMachine } from './types';
import { tournamentActionTransformer } from './tournament';
import { oneHandActionTransformer } from './oneHand';
import { speedActionTransformer } from './speed';
import { hintsActionTransformer } from './hints';

// Registry will be populated as action transformers are implemented
const ACTION_TRANSFORMER_REGISTRY: Record<string, ActionTransformerFactory> = {
  'tournament': tournamentActionTransformer,
  'one-hand': oneHandActionTransformer,
  'speed': speedActionTransformer,
  'hints': hintsActionTransformer
};

/**
 * Get action transformer by type. Pure lookup - no side effects.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function getActionTransformer(type: string, config?: any): ActionTransformer {
  const factory = ACTION_TRANSFORMER_REGISTRY[type];
  if (!factory) {
    throw new Error(`Unknown action transformer: ${type}`);
  }
  return factory(config);
}

/**
 * Compose multiple action transformers into single state machine.
 * Action transformers apply left-to-right: f(g(h(base)))
 */
export function applyActionTransformers(
  base: StateMachine,
  actionTransformers: ActionTransformerConfig[]
): StateMachine {
  return actionTransformers
    .map(at => getActionTransformer(at.type, at.config))
    .reduce((machine, transformer) => transformer(machine), base);
}

export type { ActionTransformer, ActionTransformerFactory, ActionTransformerConfig, StateMachine } from './types';
