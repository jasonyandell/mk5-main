/**
 * Layer registry - Central location for all available game layers.
 *
 * Layers can be enabled/disabled by configuration to compose different
 * game layers (standard, with special contracts, tournament mode, etc.)
 */

import type { Layer } from './types';
import { baseRuleSet } from './base';
import { nelloRuleSet } from './nello';
import { plungeRuleSet } from './plunge';
import { splashRuleSet } from './splash';
import { sevensRuleSet } from './sevens';
import { tournamentRuleSet } from './tournament';
import { oneHandRuleSet } from './oneHand';
import { speedRuleSet } from './speed';
import { hintsRuleSet } from './hints';

/**
 * Registry of all available layers.
 *
 * Note: Base layer should always be included first in composition.
 * Other layers can be enabled/disabled via configuration.
 *
 * Action transformer layers (one-hand, speed, hints) use hyphenated names
 * for backward compatibility with the old action-transformers system.
 */
export const LAYER_REGISTRY: Record<string, Layer> = {
  'base': baseRuleSet,
  'nello': nelloRuleSet,
  'plunge': plungeRuleSet,
  'splash': splashRuleSet,
  'sevens': sevensRuleSet,
  'tournament': tournamentRuleSet,
  'oneHand': oneHandRuleSet,
  'one-hand': oneHandRuleSet,  // Alias for backward compatibility
  'speed': speedRuleSet,
  'hints': hintsRuleSet
};

/**
 * Get a layer by name from the registry.
 *
 * @param name Layer name (e.g., 'nello', 'plunge')
 * @returns The requested layer
 * @throws Error if layer not found
 */
export function getLayerByName(name: string): Layer {
  const layer = LAYER_REGISTRY[name];
  if (!layer) {
    throw new Error(`Unknown layer: ${name}`);
  }
  return layer;
}

/**
 * Get multiple layers by names.
 *
 * @param names Array of layer names
 * @returns Array of layers
 */
export function getLayersByNames(names: string[]): Layer[] {
  return names.map(getLayerByName);
}
