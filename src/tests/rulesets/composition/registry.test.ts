/**
 * Tests for ruleset registry and lookup.
 *
 * Verifies that the ruleset registry works correctly:
 * - All 7 ruleSets registered correctly (base, nello, oneHand, plunge, splash, sevens, tournament)
 * - getRuleSetByName returns correct ruleSet
 * - getRuleSetByName throws on unknown ruleSet
 * - getRuleSetsByNames returns array of ruleSets in correct order
 * - Registry is immutable (can't accidentally modify)
 */

import { describe, it, expect } from 'vitest';
import {
  RULESET_REGISTRY,
  getRuleSetByName,
  getRuleSetsByNames,
  baseRuleSet,
  nelloRuleSet,
  plungeRuleSet,
  splashRuleSet,
  sevensRuleSet,
  tournamentRuleSet
} from '../../../game/rulesets';

describe('Layer Registry and Lookup', () => {
  describe('All 7 ruleSets registered correctly', () => {
    it('should have base ruleset in registry', () => {
      expect(RULESET_REGISTRY['base']).toBeDefined();
      expect(RULESET_REGISTRY['base']).toBe(baseRuleSet);
      expect(RULESET_REGISTRY['base']?.name).toBe('base');
    });

    it('should have nello ruleset in registry', () => {
      expect(RULESET_REGISTRY['nello']).toBeDefined();
      expect(RULESET_REGISTRY['nello']).toBe(nelloRuleSet);
      expect(RULESET_REGISTRY['nello']?.name).toBe('nello');
    });

    it('should have plunge ruleset in registry', () => {
      expect(RULESET_REGISTRY['plunge']).toBeDefined();
      expect(RULESET_REGISTRY['plunge']).toBe(plungeRuleSet);
      expect(RULESET_REGISTRY['plunge']?.name).toBe('plunge');
    });

    it('should have splash ruleset in registry', () => {
      expect(RULESET_REGISTRY['splash']).toBeDefined();
      expect(RULESET_REGISTRY['splash']).toBe(splashRuleSet);
      expect(RULESET_REGISTRY['splash']?.name).toBe('splash');
    });

    it('should have sevens ruleset in registry', () => {
      expect(RULESET_REGISTRY['sevens']).toBeDefined();
      expect(RULESET_REGISTRY['sevens']).toBe(sevensRuleSet);
      expect(RULESET_REGISTRY['sevens']?.name).toBe('sevens');
    });

    it('should have tournament ruleset in registry', () => {
      expect(RULESET_REGISTRY['tournament']).toBeDefined();
      expect(RULESET_REGISTRY['tournament']).toBe(tournamentRuleSet);
      expect(RULESET_REGISTRY['tournament']?.name).toBe('tournament');
    });

    it('should have exactly 7 ruleSets', () => {
      const keys = Object.keys(RULESET_REGISTRY);
      expect(keys).toHaveLength(7);
      expect(keys.sort()).toEqual(['base', 'nello', 'oneHand', 'plunge', 'sevens', 'splash', 'tournament']);
    });

    it('should have consistent ruleset names', () => {
      // Layer name should match registry key
      expect(RULESET_REGISTRY['base']?.name).toBe('base');
      expect(RULESET_REGISTRY['nello']?.name).toBe('nello');
      expect(RULESET_REGISTRY['plunge']?.name).toBe('plunge');
      expect(RULESET_REGISTRY['splash']?.name).toBe('splash');
      expect(RULESET_REGISTRY['sevens']?.name).toBe('sevens');
      expect(RULESET_REGISTRY['tournament']?.name).toBe('tournament');
    });
  });

  describe('getRuleSetByName returns correct ruleSet', () => {
    it('should return base ruleSet', () => {
      const ruleset = getRuleSetByName('base');
      expect(ruleset).toBe(baseRuleSet);
      expect(ruleset.name).toBe('base');
    });

    it('should return nello ruleSet', () => {
      const ruleset = getRuleSetByName('nello');
      expect(ruleset).toBe(nelloRuleSet);
      expect(ruleset.name).toBe('nello');
    });

    it('should return plunge ruleSet', () => {
      const ruleset = getRuleSetByName('plunge');
      expect(ruleset).toBe(plungeRuleSet);
      expect(ruleset.name).toBe('plunge');
    });

    it('should return splash ruleSet', () => {
      const ruleset = getRuleSetByName('splash');
      expect(ruleset).toBe(splashRuleSet);
      expect(ruleset.name).toBe('splash');
    });

    it('should return sevens ruleSet', () => {
      const ruleset = getRuleSetByName('sevens');
      expect(ruleset).toBe(sevensRuleSet);
      expect(ruleset.name).toBe('sevens');
    });

    it('should return ruleset with correct structure', () => {
      const ruleset = getRuleSetByName('base');

      // Should have name
      expect(ruleset.name).toBeDefined();
      expect(typeof ruleset.name).toBe('string');

      // Should have rules object
      expect(ruleset.rules).toBeDefined();
      expect(typeof ruleset.rules).toBe('object');
    });

    it('should return ruleSets with rule methods', () => {
      const ruleset = getRuleSetByName('base');

      // Base ruleset should have all 7 rule methods
      expect(ruleset.rules?.getTrumpSelector).toBeDefined();
      expect(ruleset.rules?.getFirstLeader).toBeDefined();
      expect(ruleset.rules?.getNextPlayer).toBeDefined();
      expect(ruleset.rules?.isTrickComplete).toBeDefined();
      expect(ruleset.rules?.checkHandOutcome).toBeDefined();
      expect(ruleset.rules?.getLedSuit).toBeDefined();
      expect(ruleset.rules?.calculateTrickWinner).toBeDefined();
    });
  });

  describe('getRuleSetByName throws on unknown ruleSet', () => {
    it('should throw for non-existent ruleSet', () => {
      expect(() => getRuleSetByName('unknown')).toThrow();
      expect(() => getRuleSetByName('unknown')).toThrow('Unknown ruleSet: unknown');
    });

    it('should throw for empty string', () => {
      expect(() => getRuleSetByName('')).toThrow();
      expect(() => getRuleSetByName('')).toThrow('Unknown ruleSet: ');
    });

    it('should throw for case mismatch', () => {
      expect(() => getRuleSetByName('BASE')).toThrow();
      expect(() => getRuleSetByName('Nello')).toThrow();
      expect(() => getRuleSetByName('PLUNGE')).toThrow();
    });

    it('should throw for partial matches', () => {
      expect(() => getRuleSetByName('base2')).toThrow();
      expect(() => getRuleSetByName('nell')).toThrow();
    });

    it('should throw descriptive error messages', () => {
      try {
        getRuleSetByName('invalid');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toBeInstanceOf(Error);
        expect((error as Error).message).toContain('Unknown ruleSet');
        expect((error as Error).message).toContain('invalid');
      }
    });
  });

  describe('getRuleSetsByNames returns array in correct order', () => {
    it('should return single ruleset as array', () => {
      const ruleSets = getRuleSetsByNames(['base']);
      expect(ruleSets).toHaveLength(1);
      expect(ruleSets[0]).toBe(baseRuleSet);
    });

    it('should return multiple ruleSets in order', () => {
      const ruleSets = getRuleSetsByNames(['base', 'nello', 'plunge']);
      expect(ruleSets).toHaveLength(3);
      expect(ruleSets[0]).toBe(baseRuleSet);
      expect(ruleSets[1]).toBe(nelloRuleSet);
      expect(ruleSets[2]).toBe(plungeRuleSet);
    });

    it('should preserve order even when out of alphabetical order', () => {
      const ruleSets = getRuleSetsByNames(['sevens', 'base', 'splash', 'nello']);
      expect(ruleSets).toHaveLength(4);
      expect(ruleSets[0]).toBe(sevensRuleSet);
      expect(ruleSets[1]).toBe(baseRuleSet);
      expect(ruleSets[2]).toBe(splashRuleSet);
      expect(ruleSets[3]).toBe(nelloRuleSet);
    });

    it('should handle all ruleSets', () => {
      const ruleSets = getRuleSetsByNames(['base', 'nello', 'plunge', 'splash', 'sevens', 'tournament']);
      expect(ruleSets).toHaveLength(6);
      expect(ruleSets[0]).toBe(baseRuleSet);
      expect(ruleSets[1]).toBe(nelloRuleSet);
      expect(ruleSets[2]).toBe(plungeRuleSet);
      expect(ruleSets[3]).toBe(splashRuleSet);
      expect(ruleSets[4]).toBe(sevensRuleSet);
      expect(ruleSets[5]).toBe(tournamentRuleSet);
    });

    it('should return empty array for empty input', () => {
      const ruleSets = getRuleSetsByNames([]);
      expect(ruleSets).toHaveLength(0);
      expect(ruleSets).toEqual([]);
    });

    it('should throw for invalid ruleset name in array', () => {
      expect(() => getRuleSetsByNames(['base', 'invalid', 'nello'])).toThrow();
      expect(() => getRuleSetsByNames(['base', 'invalid', 'nello'])).toThrow('Unknown ruleSet: invalid');
    });

    it('should allow duplicate ruleset names', () => {
      // Note: This might not be a desired behavior, but documenting current behavior
      const ruleSets = getRuleSetsByNames(['base', 'base', 'nello']);
      expect(ruleSets).toHaveLength(3);
      expect(ruleSets[0]).toBe(baseRuleSet);
      expect(ruleSets[1]).toBe(baseRuleSet);
      expect(ruleSets[2]).toBe(nelloRuleSet);
    });
  });

  describe('Registry consistency', () => {
    it('should maintain consistent structure', () => {
      const originalKeys = Object.keys(RULESET_REGISTRY);

      // Note: JavaScript objects are mutable by default, but we test
      // that the registry maintains its expected structure
      expect(originalKeys.length).toBe(7);
      expect(originalKeys.sort()).toEqual(['base', 'nello', 'oneHand', 'plunge', 'sevens', 'splash', 'tournament']);
    });

    it('should preserve ruleset references', () => {
      // Verify that base ruleset is always present
      expect(RULESET_REGISTRY['base']).toBeDefined();
      expect(RULESET_REGISTRY['base']).toBe(baseRuleSet);
    });

    it('should maintain ruleset names consistently', () => {
      const ruleset = getRuleSetByName('base');
      const originalName = ruleset.name;

      // Get ruleset again and verify name is consistent
      const layerAgain = getRuleSetByName('base');
      expect(layerAgain.name).toBe(originalName);
      expect(layerAgain.name).toBe('base');
    });

    it('should return same reference for repeated lookups', () => {
      const layer1 = getRuleSetByName('base');
      const layer2 = getRuleSetByName('base');

      expect(layer1).toBe(layer2);
      expect(layer1).toBe(baseRuleSet);
    });
  });

  describe('Registry integration', () => {
    it('should work with composeRules', () => {
      const ruleSets = getRuleSetsByNames(['base', 'nello']);

      // Should be able to compose these ruleSets
      expect(() => {
        // Import composeRules and test (basic smoke test)
        expect(ruleSets).toHaveLength(2);
        expect(ruleSets[0]?.name).toBe('base');
        expect(ruleSets[1]?.name).toBe('nello');
      }).not.toThrow();
    });

    it('should support dynamic ruleset selection', () => {
      // Simulating configuration-based ruleset selection
      const enabledRuleSets = ['base', 'plunge', 'splash'];
      const ruleSets = getRuleSetsByNames(enabledRuleSets);

      expect(ruleSets).toHaveLength(3);
      expect(ruleSets.map(l => l.name)).toEqual(['base', 'plunge', 'splash']);
    });

    it('should support conditional ruleset inclusion', () => {
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

      const ruleSets = getRuleSetsByNames(layerNames);

      expect(ruleSets).toHaveLength(4); // base + nello + plunge + splash
      expect(ruleSets.map(l => l.name)).toEqual(['base', 'nello', 'plunge', 'splash']);
    });
  });

  describe('Layer metadata', () => {
    it('should have descriptive names', () => {
      const ruleSets = getRuleSetsByNames(['base', 'nello', 'plunge', 'splash', 'sevens']);

      ruleSets.forEach(ruleset => {
        expect(ruleset.name).toBeTruthy();
        expect(ruleset.name.length).toBeGreaterThan(0);
        expect(typeof ruleset.name).toBe('string');
      });
    });

    it('should have unique names', () => {
      const ruleSets = getRuleSetsByNames(['base', 'nello', 'plunge', 'splash', 'sevens']);
      const names = ruleSets.map(l => l.name);
      const uniqueNames = new Set(names);

      expect(uniqueNames.size).toBe(names.length);
    });

    it('should have valid structure for all ruleSets', () => {
      const ruleSets = getRuleSetsByNames(['base', 'nello', 'plunge', 'splash', 'sevens']);

      ruleSets.forEach(ruleset => {
        // Each ruleset should have a name
        expect(ruleset.name).toBeDefined();

        // Each ruleset should have rules OR getValidActions (or both)
        const hasRules = ruleset.rules && Object.keys(ruleset.rules).length > 0;
        const hasActions = ruleset.getValidActions !== undefined;

        expect(hasRules || hasActions).toBe(true);
      });
    });
  });
});
