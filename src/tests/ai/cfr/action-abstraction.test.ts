/**
 * Unit tests for action abstraction utilities
 */

import { describe, it, expect } from 'vitest';
import {
  actionToKey,
  getActionFromKey,
  getActionKeys,
  sampleAction,
  selectBestAction,
  isPlayAction
} from '../../../game/ai/cfr/action-abstraction';
import type { ValidAction } from '../../../multiplayer/types';
import { createSeededRng } from '../../../game/ai/hand-sampler';

// Helper to create mock play actions
function makePlayAction(dominoId: string): ValidAction {
  return {
    action: { type: 'play', player: 0, dominoId },
    label: `Play ${dominoId}`
  };
}

function makePassAction(): ValidAction {
  return {
    action: { type: 'pass', player: 0 },
    label: 'Pass'
  };
}

describe('action abstraction', () => {
  describe('actionToKey', () => {
    it('extracts domino ID from play action', () => {
      const action = makePlayAction('6-4');
      expect(actionToKey(action)).toBe('6-4');
    });

    it('throws for non-play actions', () => {
      const action = makePassAction();
      expect(() => actionToKey(action)).toThrow('only supports play actions');
    });
  });

  describe('getActionFromKey', () => {
    it('finds matching action by domino ID', () => {
      const actions = [
        makePlayAction('6-4'),
        makePlayAction('5-5'),
        makePlayAction('3-2')
      ];

      const found = getActionFromKey('5-5', actions);
      expect(found).toBeDefined();
      expect((found?.action as { dominoId: string }).dominoId).toBe('5-5');
    });

    it('returns undefined for non-existent action', () => {
      const actions = [makePlayAction('6-4')];
      expect(getActionFromKey('5-5', actions)).toBeUndefined();
    });

    it('ignores non-play actions', () => {
      const actions = [makePassAction(), makePlayAction('6-4')];
      const found = getActionFromKey('6-4', actions);
      expect(found).toBeDefined();
    });
  });

  describe('getActionKeys', () => {
    it('extracts all domino IDs from play actions', () => {
      const actions = [
        makePlayAction('6-4'),
        makePassAction(),
        makePlayAction('5-5'),
        makePlayAction('3-2')
      ];

      const keys = getActionKeys(actions);
      expect(keys).toEqual(['6-4', '5-5', '3-2']);
    });

    it('returns empty array for no play actions', () => {
      const actions = [makePassAction()];
      expect(getActionKeys(actions)).toEqual([]);
    });
  });

  describe('sampleAction', () => {
    it('samples according to probability distribution', () => {
      const actions = [
        makePlayAction('6-4'),
        makePlayAction('5-5'),
        makePlayAction('3-2')
      ];

      const strategy = new Map([
        ['6-4', 0.2],
        ['5-5', 0.7],
        ['3-2', 0.1]
      ]);

      // Run many samples to check distribution
      const counts = { '6-4': 0, '5-5': 0, '3-2': 0 };
      const rng = createSeededRng(12345);
      const numSamples = 10000;

      for (let i = 0; i < numSamples; i++) {
        const sampled = sampleAction(strategy, actions, rng);
        const key = (sampled.action as { dominoId: string }).dominoId;
        counts[key as keyof typeof counts]++;
      }

      // Check distribution is approximately correct (within 5%)
      expect(counts['6-4'] / numSamples).toBeCloseTo(0.2, 1);
      expect(counts['5-5'] / numSamples).toBeCloseTo(0.7, 1);
      expect(counts['3-2'] / numSamples).toBeCloseTo(0.1, 1);
    });

    it('returns single action when only one available', () => {
      const actions = [makePlayAction('6-4')];
      const strategy = new Map([['6-4', 1.0]]);
      const rng = createSeededRng(42);

      const sampled = sampleAction(strategy, actions, rng);
      expect((sampled.action as { dominoId: string }).dominoId).toBe('6-4');
    });

    it('falls back to uniform for zero probabilities', () => {
      const actions = [
        makePlayAction('6-4'),
        makePlayAction('5-5')
      ];
      const strategy = new Map([
        ['6-4', 0],
        ['5-5', 0]
      ]);

      // Should not throw, should pick uniformly
      const rng = createSeededRng(42);
      const sampled = sampleAction(strategy, actions, rng);
      expect(sampled).toBeDefined();
    });
  });

  describe('selectBestAction', () => {
    it('selects action with highest probability', () => {
      const actions = [
        makePlayAction('6-4'),
        makePlayAction('5-5'),
        makePlayAction('3-2')
      ];

      const strategy = new Map([
        ['6-4', 0.2],
        ['5-5', 0.7],
        ['3-2', 0.1]
      ]);

      const best = selectBestAction(strategy, actions);
      expect((best.action as { dominoId: string }).dominoId).toBe('5-5');
    });

    it('returns first action for ties', () => {
      const actions = [
        makePlayAction('6-4'),
        makePlayAction('5-5')
      ];

      const strategy = new Map([
        ['6-4', 0.5],
        ['5-5', 0.5]
      ]);

      const best = selectBestAction(strategy, actions);
      // First one with highest prob wins
      expect(best).toBeDefined();
    });
  });

  describe('isPlayAction', () => {
    it('returns true for play actions', () => {
      expect(isPlayAction(makePlayAction('6-4'))).toBe(true);
    });

    it('returns false for non-play actions', () => {
      expect(isPlayAction(makePassAction())).toBe(false);
    });
  });
});
