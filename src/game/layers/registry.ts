/**
 * Layer registry - Central location for all available game layers.
 *
 * Layers can be enabled/disabled by configuration to compose different
 * game layers (standard, with special contracts, tournament mode, etc.)
 */

import type { Layer } from './types';
import { baseLayer } from './base';
import { nelloLayer } from './nello';
import { plungeLayer } from './plunge';
import { splashLayer } from './splash';
import { sevensLayer } from './sevens';
import { tournamentLayer } from './tournament';
import { oneHandLayer } from './oneHand';
import { speedLayer } from './speed';
import { hintsLayer } from './hints';
import { consensusLayer } from './consensus';

/**
 * Registry of all available layers.
 *
 * Note: Base layer should always be included first in composition.
 * Other layers can be enabled/disabled via configuration.
 */
export const LAYER_REGISTRY: Record<string, Layer> = {
  'base': baseLayer,
  'nello': nelloLayer,
  'plunge': plungeLayer,
  'splash': splashLayer,
  'sevens': sevensLayer,
  'tournament': tournamentLayer,
  'oneHand': oneHandLayer,
  'speed': speedLayer,
  'hints': hintsLayer,
  'consensus': consensusLayer
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
