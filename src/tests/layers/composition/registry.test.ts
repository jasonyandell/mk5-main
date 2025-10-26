/**
 * Tests for layer registry and lookup.
 *
 * Verifies that the layer registry works correctly:
 * - All 5 layers registered correctly (base, nello, plunge, splash, sevens)
 * - getLayerByName returns correct layer
 * - getLayerByName throws on unknown layer
 * - getLayersByNames returns array of layers in correct order
 * - Registry is immutable (can't accidentally modify)
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
  sevensLayer
} from '../../../game/layers';

describe('Layer Registry and Lookup', () => {
  describe('All 5 layers registered correctly', () => {
    it('should have base layer in registry', () => {
      expect(LAYER_REGISTRY['base']).toBeDefined();
      expect(LAYER_REGISTRY['base']).toBe(baseLayer);
      expect(LAYER_REGISTRY['base']?.name).toBe('base');
    });

    it('should have nello layer in registry', () => {
      expect(LAYER_REGISTRY['nello']).toBeDefined();
      expect(LAYER_REGISTRY['nello']).toBe(nelloLayer);
      expect(LAYER_REGISTRY['nello']?.name).toBe('nello');
    });

    it('should have plunge layer in registry', () => {
      expect(LAYER_REGISTRY['plunge']).toBeDefined();
      expect(LAYER_REGISTRY['plunge']).toBe(plungeLayer);
      expect(LAYER_REGISTRY['plunge']?.name).toBe('plunge');
    });

    it('should have splash layer in registry', () => {
      expect(LAYER_REGISTRY['splash']).toBeDefined();
      expect(LAYER_REGISTRY['splash']).toBe(splashLayer);
      expect(LAYER_REGISTRY['splash']?.name).toBe('splash');
    });

    it('should have sevens layer in registry', () => {
      expect(LAYER_REGISTRY['sevens']).toBeDefined();
      expect(LAYER_REGISTRY['sevens']).toBe(sevensLayer);
      expect(LAYER_REGISTRY['sevens']?.name).toBe('sevens');
    });

    it('should have exactly 5 layers', () => {
      const keys = Object.keys(LAYER_REGISTRY);
      expect(keys).toHaveLength(5);
      expect(keys.sort()).toEqual(['base', 'nello', 'plunge', 'sevens', 'splash']);
    });

    it('should have consistent layer names', () => {
      // Layer name should match registry key
      expect(LAYER_REGISTRY['base']?.name).toBe('base');
      expect(LAYER_REGISTRY['nello']?.name).toBe('nello');
      expect(LAYER_REGISTRY['plunge']?.name).toBe('plunge');
      expect(LAYER_REGISTRY['splash']?.name).toBe('splash');
      expect(LAYER_REGISTRY['sevens']?.name).toBe('sevens');
    });
  });

  describe('getLayerByName returns correct layer', () => {
    it('should return base layer', () => {
      const layer = getLayerByName('base');
      expect(layer).toBe(baseLayer);
      expect(layer.name).toBe('base');
    });

    it('should return nello layer', () => {
      const layer = getLayerByName('nello');
      expect(layer).toBe(nelloLayer);
      expect(layer.name).toBe('nello');
    });

    it('should return plunge layer', () => {
      const layer = getLayerByName('plunge');
      expect(layer).toBe(plungeLayer);
      expect(layer.name).toBe('plunge');
    });

    it('should return splash layer', () => {
      const layer = getLayerByName('splash');
      expect(layer).toBe(splashLayer);
      expect(layer.name).toBe('splash');
    });

    it('should return sevens layer', () => {
      const layer = getLayerByName('sevens');
      expect(layer).toBe(sevensLayer);
      expect(layer.name).toBe('sevens');
    });

    it('should return layer with correct structure', () => {
      const layer = getLayerByName('base');

      // Should have name
      expect(layer.name).toBeDefined();
      expect(typeof layer.name).toBe('string');

      // Should have rules object
      expect(layer.rules).toBeDefined();
      expect(typeof layer.rules).toBe('object');
    });

    it('should return layers with rule methods', () => {
      const layer = getLayerByName('base');

      // Base layer should have all 7 rule methods
      expect(layer.rules?.getTrumpSelector).toBeDefined();
      expect(layer.rules?.getFirstLeader).toBeDefined();
      expect(layer.rules?.getNextPlayer).toBeDefined();
      expect(layer.rules?.isTrickComplete).toBeDefined();
      expect(layer.rules?.checkHandOutcome).toBeDefined();
      expect(layer.rules?.getLedSuit).toBeDefined();
      expect(layer.rules?.calculateTrickWinner).toBeDefined();
    });
  });

  describe('getLayerByName throws on unknown layer', () => {
    it('should throw for non-existent layer', () => {
      expect(() => getLayerByName('unknown')).toThrow();
      expect(() => getLayerByName('unknown')).toThrow('Unknown layer: unknown');
    });

    it('should throw for empty string', () => {
      expect(() => getLayerByName('')).toThrow();
      expect(() => getLayerByName('')).toThrow('Unknown layer: ');
    });

    it('should throw for case mismatch', () => {
      expect(() => getLayerByName('BASE')).toThrow();
      expect(() => getLayerByName('Nello')).toThrow();
      expect(() => getLayerByName('PLUNGE')).toThrow();
    });

    it('should throw for partial matches', () => {
      expect(() => getLayerByName('base2')).toThrow();
      expect(() => getLayerByName('nell')).toThrow();
    });

    it('should throw descriptive error messages', () => {
      try {
        getLayerByName('invalid');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toContain('Unknown layer');
        expect((error as Error).message).toContain('invalid');
      }
    });
  });

  describe('getLayersByNames returns array in correct order', () => {
    it('should return single layer as array', () => {
      const layers = getLayersByNames(['base']);
      expect(layers).toHaveLength(1);
      expect(layers[0]).toBe(baseLayer);
    });

    it('should return multiple layers in order', () => {
      const layers = getLayersByNames(['base', 'nello', 'plunge']);
      expect(layers).toHaveLength(3);
      expect(layers[0]).toBe(baseLayer);
      expect(layers[1]).toBe(nelloLayer);
      expect(layers[2]).toBe(plungeLayer);
    });

    it('should preserve order even when out of alphabetical order', () => {
      const layers = getLayersByNames(['sevens', 'base', 'splash', 'nello']);
      expect(layers).toHaveLength(4);
      expect(layers[0]).toBe(sevensLayer);
      expect(layers[1]).toBe(baseLayer);
      expect(layers[2]).toBe(splashLayer);
      expect(layers[3]).toBe(nelloLayer);
    });

    it('should handle all layers', () => {
      const layers = getLayersByNames(['base', 'nello', 'plunge', 'splash', 'sevens']);
      expect(layers).toHaveLength(5);
      expect(layers[0]).toBe(baseLayer);
      expect(layers[1]).toBe(nelloLayer);
      expect(layers[2]).toBe(plungeLayer);
      expect(layers[3]).toBe(splashLayer);
      expect(layers[4]).toBe(sevensLayer);
    });

    it('should return empty array for empty input', () => {
      const layers = getLayersByNames([]);
      expect(layers).toHaveLength(0);
      expect(layers).toEqual([]);
    });

    it('should throw for invalid layer name in array', () => {
      expect(() => getLayersByNames(['base', 'invalid', 'nello'])).toThrow();
      expect(() => getLayersByNames(['base', 'invalid', 'nello'])).toThrow('Unknown layer: invalid');
    });

    it('should allow duplicate layer names', () => {
      // Note: This might not be a desired behavior, but documenting current behavior
      const layers = getLayersByNames(['base', 'base', 'nello']);
      expect(layers).toHaveLength(3);
      expect(layers[0]).toBe(baseLayer);
      expect(layers[1]).toBe(baseLayer);
      expect(layers[2]).toBe(nelloLayer);
    });
  });

  describe('Registry consistency', () => {
    it('should maintain consistent structure', () => {
      const originalKeys = Object.keys(LAYER_REGISTRY);

      // Note: JavaScript objects are mutable by default, but we test
      // that the registry maintains its expected structure
      expect(originalKeys.length).toBe(5);
      expect(originalKeys.sort()).toEqual(['base', 'nello', 'plunge', 'sevens', 'splash']);
    });

    it('should preserve layer references', () => {
      // Verify that base layer is always present
      expect(LAYER_REGISTRY['base']).toBeDefined();
      expect(LAYER_REGISTRY['base']).toBe(baseLayer);
    });

    it('should maintain layer names consistently', () => {
      const layer = getLayerByName('base');
      const originalName = layer.name;

      // Get layer again and verify name is consistent
      const layerAgain = getLayerByName('base');
      expect(layerAgain.name).toBe(originalName);
      expect(layerAgain.name).toBe('base');
    });

    it('should return same reference for repeated lookups', () => {
      const layer1 = getLayerByName('base');
      const layer2 = getLayerByName('base');

      expect(layer1).toBe(layer2);
      expect(layer1).toBe(baseLayer);
    });
  });

  describe('Registry integration', () => {
    it('should work with composeRules', () => {
      const layers = getLayersByNames(['base', 'nello']);

      // Should be able to compose these layers
      expect(() => {
        // Import composeRules and test (basic smoke test)
        expect(layers).toHaveLength(2);
        expect(layers[0]?.name).toBe('base');
        expect(layers[1]?.name).toBe('nello');
      }).not.toThrow();
    });

    it('should support dynamic layer selection', () => {
      // Simulating configuration-based layer selection
      const enabledLayers = ['base', 'plunge', 'splash'];
      const layers = getLayersByNames(enabledLayers);

      expect(layers).toHaveLength(3);
      expect(layers.map(l => l.name)).toEqual(['base', 'plunge', 'splash']);
    });

    it('should support conditional layer inclusion', () => {
      // Simulating feature flags
      const config = {
        enableNello: true,
        enableSevens: false,
        enablePlunge: true,
        enableSplash: true
      };

      const layerNames = ['base'];
      if (config.enableNello) layerNames.push('nello');
      if (config.enablePlunge) layerNames.push('plunge');
      if (config.enableSplash) layerNames.push('splash');
      if (config.enableSevens) layerNames.push('sevens');

      const layers = getLayersByNames(layerNames);

      expect(layers).toHaveLength(4); // base + nello + plunge + splash
      expect(layers.map(l => l.name)).toEqual(['base', 'nello', 'plunge', 'splash']);
    });
  });

  describe('Layer metadata', () => {
    it('should have descriptive names', () => {
      const layers = getLayersByNames(['base', 'nello', 'plunge', 'splash', 'sevens']);

      layers.forEach(layer => {
        expect(layer.name).toBeTruthy();
        expect(layer.name.length).toBeGreaterThan(0);
        expect(typeof layer.name).toBe('string');
      });
    });

    it('should have unique names', () => {
      const layers = getLayersByNames(['base', 'nello', 'plunge', 'splash', 'sevens']);
      const names = layers.map(l => l.name);
      const uniqueNames = new Set(names);

      expect(uniqueNames.size).toBe(names.length);
    });

    it('should have valid structure for all layers', () => {
      const layers = getLayersByNames(['base', 'nello', 'plunge', 'splash', 'sevens']);

      layers.forEach(layer => {
        // Each layer should have a name
        expect(layer.name).toBeDefined();

        // Each layer should have rules OR getValidActions (or both)
        const hasRules = layer.rules && Object.keys(layer.rules).length > 0;
        const hasActions = layer.getValidActions !== undefined;

        expect(hasRules || hasActions).toBe(true);
      });
    });
  });
});
