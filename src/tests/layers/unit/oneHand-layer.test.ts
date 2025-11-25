/**
 * Tests for OneHand Layer
 *
 * Verifies that oneHandLayer properly:
 * - Overrides phase transition logic to create terminal state
 * - Provides action automation for bidding/trump selection
 */

import { describe, it, expect } from 'vitest';
import { oneHandLayer } from '../../../game/layers/oneHand';
import { baseLayer } from '../../../game/layers/base';
import { composeRules } from '../../../game/layers/compose';
import type { GameState } from '../../../game/types';

describe('OneHand Layer', () => {
  describe('getPhaseAfterHandComplete', () => {
    it('should return one-hand-complete phase', () => {
      // Create minimal state (actual state doesn't matter for this rule)
      const state = {} as GameState;

      // Get the rule from oneHandLayer
      const rule = oneHandLayer.rules?.getPhaseAfterHandComplete;
      expect(rule).toBeDefined();

      // Call rule (prev parameter not used in oneHand implementation)
      const phase = rule!(state, 'bidding');

      expect(phase).toBe('one-hand-complete');
    });

    it('should override base behavior when composed', () => {
      // Compose base + oneHand
      const composed = composeRules([baseLayer, oneHandLayer]);

      const state = {} as GameState;

      // Base alone would return 'bidding'
      const basePhase = baseLayer.rules!.getPhaseAfterHandComplete!(state, 'bidding');
      expect(basePhase).toBe('bidding');

      // Composed with oneHand should return 'one-hand-complete'
      const composedPhase = composed.getPhaseAfterHandComplete(state);
      expect(composedPhase).toBe('one-hand-complete');
    });

    it('should work in composition with other layers', () => {
      // Verify oneHand can compose with base (no conflicts)
      const layers = [baseLayer, oneHandLayer];
      const composed = composeRules(layers);

      const state = {} as GameState;

      // Should get oneHand's phase
      expect(composed.getPhaseAfterHandComplete(state)).toBe('one-hand-complete');

      // All other rules should still work (delegate to base)
      expect(typeof composed.getTrumpSelector).toBe('function');
      expect(typeof composed.isTrickComplete).toBe('function');
      expect(typeof composed.calculateScore).toBe('function');
    });
  });

  describe('layer structure', () => {
    it('should have correct name', () => {
      expect(oneHandLayer.name).toBe('oneHand');
    });

    it('should override getPhaseAfterHandComplete rule', () => {
      expect(oneHandLayer.rules).toBeDefined();
      expect(Object.keys(oneHandLayer.rules!)).toEqual(['getPhaseAfterHandComplete']);
    });

    it('should provide getValidActions for automation', () => {
      expect(oneHandLayer.getValidActions).toBeDefined();
      expect(typeof oneHandLayer.getValidActions).toBe('function');
    });
  });
});
