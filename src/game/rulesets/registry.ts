/**
 * RuleSet registry - Central location for all available game rule sets.
 *
 * RuleSets can be enabled/disabled by configuration to compose different
 * game rule sets (standard, with special contracts, tournament mode, etc.)
 */

import type { GameRuleSet } from './types';
import { baseRuleSet } from './base';
import { nelloRuleSet } from './nello';
import { plungeRuleSet } from './plunge';
import { splashRuleSet } from './splash';
import { sevensRuleSet } from './sevens';

/**
 * Registry of all available rule sets.
 *
 * Note: Base rule set should always be included first in composition.
 * Other rule sets can be enabled/disabled via configuration.
 */
export const RULESET_REGISTRY: Record<string, GameRuleSet> = {
  'base': baseRuleSet,
  'nello': nelloRuleSet,
  'plunge': plungeRuleSet,
  'splash': splashRuleSet,
  'sevens': sevensRuleSet
};

/**
 * Get a rule set by name from the registry.
 *
 * @param name RuleSet name (e.g., 'nello', 'plunge')
 * @returns The requested rule set
 * @throws Error if rule set not found
 */
export function getRuleSetByName(name: string): GameRuleSet {
  const ruleSet = RULESET_REGISTRY[name];
  if (!ruleSet) {
    throw new Error(`Unknown ruleSet: ${name}`);
  }
  return ruleSet;
}

/**
 * Get multiple rule sets by names.
 *
 * @param names Array of rule set names
 * @returns Array of rule sets
 */
export function getRuleSetsByNames(names: string[]): GameRuleSet[] {
  return names.map(getRuleSetByName);
}
