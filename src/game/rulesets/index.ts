/**
 * RuleSets module - Pure functional game rule composition.
 *
 * Exports all rule sets, composition utilities, and types for the
 * threaded rules architecture.
 */

// Types
export * from './types';

// Helpers
export * from './helpers';

// RuleSets
export { baseRuleSet } from './base';
export { nelloRuleSet } from './nello';
export { plungeRuleSet } from './plunge';
export { splashRuleSet } from './splash';
export { sevensRuleSet } from './sevens';

// Composition
export { composeRules, composeActions } from './compose';
export { RULESET_REGISTRY, getRuleSetByName, getRuleSetsByNames } from './registry';
