/**
 * Tests for OneHand Layer
 *
 * Verifies that oneHandRuleSet properly:
 * - Overrides phase transition logic to create terminal state
 * - Provides action automation for bidding/trump selection
 */

import { describe, it, expect } from 'vitest';
import { oneHandRuleSet } from '../../../game/layers/oneHand';
import { baseRuleSet } from '../../../game/layers/base';
import { composeRules } from '../../../game/layers/compose';
import type { GameState } from '../../../game/types';

describe('OneHand RuleSet', () => {
  describe('getPhaseAfterHandComplete', () => {
    it('should return one-hand-complete phase', () => {
      // Create minimal state (actual state doesn't matter for this rule)
      const state = {} as GameState;

      // Get the rule from oneHandRuleSet
      const rule = oneHandRuleSet.rules?.getPhaseAfterHandComplete;
      expect(rule).toBeDefined();

      // Call rule (prev parameter not used in oneHand implementation)
      const phase = rule!(state, 'bidding');

      expect(phase).toBe('one-hand-complete');
    });

    it('should override base behavior when composed', () => {
      // Compose base + oneHand
      const composed = composeRules([baseRuleSet, oneHandRuleSet]);

      const state = {} as GameState;

      // Base alone would return 'bidding'
      const basePhase = baseRuleSet.rules!.getPhaseAfterHandComplete!(state, 'bidding');
      expect(basePhase).toBe('bidding');

      // Composed with oneHand should return 'one-hand-complete'
      const composedPhase = composed.getPhaseAfterHandComplete(state);
      expect(composedPhase).toBe('one-hand-complete');
    });

    it('should work in composition with other rulesets', () => {
      // Verify oneHand can compose with base (no conflicts)
      const ruleSets = [baseRuleSet, oneHandRuleSet];
      const composed = composeRules(ruleSets);

      const state = {} as GameState;

      // Should get oneHand's phase
      expect(composed.getPhaseAfterHandComplete(state)).toBe('one-hand-complete');

      // All other rules should still work (delegate to base)
      expect(typeof composed.getTrumpSelector).toBe('function');
      expect(typeof composed.isTrickComplete).toBe('function');
      expect(typeof composed.calculateScore).toBe('function');
    });
  });

  describe('ruleset structure', () => {
    it('should have correct name', () => {
      expect(oneHandRuleSet.name).toBe('oneHand');
    });

    it('should override getPhaseAfterHandComplete rule', () => {
      expect(oneHandRuleSet.rules).toBeDefined();
      expect(Object.keys(oneHandRuleSet.rules!)).toEqual(['getPhaseAfterHandComplete']);
    });

    it('should provide getValidActions for automation', () => {
      expect(oneHandRuleSet.getValidActions).toBeDefined();
      expect(typeof oneHandRuleSet.getValidActions).toBe('function');
    });
  });
});
