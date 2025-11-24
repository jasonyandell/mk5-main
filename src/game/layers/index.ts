/**
 * Layers module - Pure functional game rule composition.
 *
 * Exports all layers, composition utilities, and types for the
 * threaded rules architecture.
 */

// Types
export * from './types';

// Helpers
export * from './helpers';

// Layers
export { baseRuleSet } from './base';
export { nelloRuleSet } from './nello';
export { plungeRuleSet } from './plunge';
export { splashRuleSet } from './splash';
export { sevensRuleSet } from './sevens';
export { tournamentRuleSet } from './tournament';
export { oneHandRuleSet } from './oneHand';

// Composition
export { composeRules, composeActions, composeActionGenerators } from './compose';
export { LAYER_REGISTRY, getLayerByName, getLayersByNames } from './registry';
