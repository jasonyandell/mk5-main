// Pure lookup - no registration, no side effects
import type { VariantFactory, Variant, VariantConfig, StateMachine } from './types';
import { tournamentVariant } from './tournament';
import { oneHandVariant } from './oneHand';
import { speedVariant } from './speed';
import { hintsVariant } from './hints';

// Registry will be populated as variants are implemented
const VARIANT_REGISTRY: Record<string, VariantFactory> = {
  'tournament': tournamentVariant,
  'one-hand': oneHandVariant,
  'speed': speedVariant,
  'hints': hintsVariant
};

/**
 * Get variant by type. Pure lookup - no side effects.
 */
// eslint-disable-next-line @typescript-eslint/no-explicit-any
export function getVariant(type: string, config?: any): Variant {
  const factory = VARIANT_REGISTRY[type];
  if (!factory) {
    throw new Error(`Unknown variant: ${type}`);
  }
  return factory(config);
}

/**
 * Compose multiple variants into single state machine.
 * Variants apply left-to-right: f(g(h(base)))
 */
export function applyVariants(
  base: StateMachine,
  variants: VariantConfig[]
): StateMachine {
  return variants
    .map(v => getVariant(v.type, v.config))
    .reduce((machine, variant) => variant(machine), base);
}

export type { Variant, VariantFactory, VariantConfig, StateMachine } from './types';
