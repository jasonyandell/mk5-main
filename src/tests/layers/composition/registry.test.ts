/**
 * Tests for layer registry and lookup.
 *
 * Verifies registry functions work correctly:
 * - getLayerByName returns correct layer
 * - getLayerByName throws on unknown layer
 * - getLayersByNames returns array in correct order
 * - Registry contains all expected layers
 */

import { describe, it, expect } from 'vitest';
import {
  LAYER_REGISTRY,
  getLayerByName,
  getLayersByNames,
  baseLayer,
  nelloLayer,
  plungeLayer,
  splashLayer,
  sevensLayer,
  tournamentLayer
} from '../../../game/layers';

describe('Layer Registry', () => {
  describe('LAYER_REGISTRY', () => {
    it('should contain all expected layers', () => {
      const keys = Object.keys(LAYER_REGISTRY);
      expect(keys).toHaveLength(10);
      expect(keys.sort()).toEqual([
        'base',
        'consensus',
        'hints',
        'nello',
        'oneHand',
        'plunge',
        'sevens',
        'speed',
        'splash',
        'tournament'
      ]);
    });

    it('should map keys to correct layer references', () => {
      expect(LAYER_REGISTRY['base']).toBe(baseLayer);
      expect(LAYER_REGISTRY['nello']).toBe(nelloLayer);
      expect(LAYER_REGISTRY['plunge']).toBe(plungeLayer);
      expect(LAYER_REGISTRY['splash']).toBe(splashLayer);
      expect(LAYER_REGISTRY['sevens']).toBe(sevensLayer);
      expect(LAYER_REGISTRY['tournament']).toBe(tournamentLayer);
    });
  });

  describe('getLayerByName', () => {
    it('should return correct layer for valid name', () => {
      expect(getLayerByName('base')).toBe(baseLayer);
      expect(getLayerByName('nello')).toBe(nelloLayer);
      expect(getLayerByName('plunge')).toBe(plungeLayer);
    });

    it('should return same reference for repeated lookups', () => {
      const layer1 = getLayerByName('base');
      const layer2 = getLayerByName('base');
      expect(layer1).toBe(layer2);
    });

    it('should throw for unknown layer name', () => {
      expect(() => getLayerByName('unknown')).toThrow('Unknown layer: unknown');
      expect(() => getLayerByName('')).toThrow('Unknown layer: ');
    });

    it('should throw for case mismatch', () => {
      expect(() => getLayerByName('BASE')).toThrow();
      expect(() => getLayerByName('Nello')).toThrow();
    });
  });

  describe('getLayersByNames', () => {
    it('should return single layer as array', () => {
      const layers = getLayersByNames(['base']);
      expect(layers).toHaveLength(1);
      expect(layers[0]).toBe(baseLayer);
    });

    it('should return multiple layers in correct order', () => {
      const layers = getLayersByNames(['base', 'nello', 'plunge']);
      expect(layers).toHaveLength(3);
      expect(layers[0]).toBe(baseLayer);
      expect(layers[1]).toBe(nelloLayer);
      expect(layers[2]).toBe(plungeLayer);
    });

    it('should preserve order regardless of alphabetical sorting', () => {
      const layers = getLayersByNames(['sevens', 'base', 'splash']);
      expect(layers[0]).toBe(sevensLayer);
      expect(layers[1]).toBe(baseLayer);
      expect(layers[2]).toBe(splashLayer);
    });

    it('should return empty array for empty input', () => {
      const layers = getLayersByNames([]);
      expect(layers).toEqual([]);
    });

    it('should throw for unknown layer name in array', () => {
      expect(() => getLayersByNames(['base', 'invalid'])).toThrow('Unknown layer: invalid');
    });

    it('should handle duplicate layer names', () => {
      const layers = getLayersByNames(['base', 'base', 'nello']);
      expect(layers).toHaveLength(3);
      expect(layers[0]).toBe(baseLayer);
      expect(layers[1]).toBe(baseLayer);
      expect(layers[2]).toBe(nelloLayer);
    });
  });
});
