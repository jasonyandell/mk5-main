/**
 * Tests for the rule composition mechanics.
 *
 * Verifies that the monadic composition via reduce works correctly:
 * - Base layer produces correct results alone
 * - Base + single layer composes correctly
 * - Base + multiple layers compose correctly
 * - Layer order matters (left-to-right application)
 * - All 7 rule methods compose correctly
 * - Composition produces valid GameRules interface
 */

import { describe, it, expect } from 'vitest';
import { composeRules, baseLayer, nelloLayer, plungeLayer, sevensLayer } from '../../../game/layers';
import type { GameState, Bid, TrumpSelection, Domino, Play } from '../../../game/types';
import { createInitialState } from '../../../game/core/state';

describe('Rule Composition Mechanics', () => {
  // Helper to create a minimal test state
  function createTestState(overrides: Partial<GameState> = {}): GameState {
    const base = createInitialState();
    return {
      ...base,
      ...overrides
    };
  }

  describe('Base layer alone', () => {
    it('should produce correct getTrumpSelector result', () => {
      const rules = composeRules([baseLayer]);
      const state = createTestState();
      const bid: Bid = { type: 'marks', value: 2, player: 1 };

      expect(rules.getTrumpSelector(state, bid)).toBe(1);
    });

    it('should produce correct getFirstLeader result', () => {
      const rules = composeRules([baseLayer]);
      const state = createTestState();
      const trump: TrumpSelection = { type: 'suit', suit: 1 };

      expect(rules.getFirstLeader(state, 2, trump)).toBe(2);
    });

    it('should produce correct getNextPlayer result', () => {
      const rules = composeRules([baseLayer]);
      const state = createTestState();

      expect(rules.getNextPlayer(state, 0)).toBe(1);
      expect(rules.getNextPlayer(state, 1)).toBe(2);
      expect(rules.getNextPlayer(state, 2)).toBe(3);
      expect(rules.getNextPlayer(state, 3)).toBe(0);
    });

    it('should produce correct isTrickComplete result', () => {
      const rules = composeRules([baseLayer]);

      const state3Plays = createTestState({
        currentTrick: [
          { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
          { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
          { player: 2, domino: { id: '2-3', high: 2, low: 3, points: 0 } }
        ]
      });
      expect(rules.isTrickComplete(state3Plays)).toBe(false);

      const state4Plays = createTestState({
        currentTrick: [
          { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
          { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
          { player: 2, domino: { id: '2-3', high: 2, low: 3, points: 0 } },
          { player: 3, domino: { id: '3-4', high: 3, low: 4, points: 0 } }
        ]
      });
      expect(rules.isTrickComplete(state4Plays)).toBe(true);
    });

    it('should produce correct checkHandOutcome result', () => {
      const rules = composeRules([baseLayer]);

      const state6Tricks = createTestState({ tricks: new Array(6) });
      expect(rules.checkHandOutcome(state6Tricks)).toBeNull();

      const state7Tricks = createTestState({ tricks: new Array(7) });
      const outcome = rules.checkHandOutcome(state7Tricks);
      expect(outcome).not.toBeNull();
      expect(outcome?.isDetermined).toBe(true);
      expect(outcome?.reason).toBe('All tricks played');
    });

    it('should produce correct getLedSuit result for non-trump', () => {
      const rules = composeRules([baseLayer]);
      const state = createTestState({ trump: { type: 'suit', suit: 1 } });
      const domino: Domino = { id: '5-6', high: 5, low: 6, points: 0 };

      expect(rules.getLedSuit(state, domino)).toBe(5);
    });

    it('should produce correct calculateTrickWinner result', () => {
      const rules = composeRules([baseLayer]);
      const state = createTestState({
        trump: { type: 'suit', suit: 1 },
        currentSuit: 2
      });
      const trick: Play[] = [
        { player: 0, domino: { id: '2-3', high: 2, low: 3, points: 0 } },
        { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } }, // Trump
        { player: 2, domino: { id: '2-4', high: 2, low: 4, points: 0 } }
      ];

      expect(rules.calculateTrickWinner(state, trick)).toBe(1); // Trump wins
    });
  });

  describe('Base + single special layer', () => {
    it('should compose base + nello correctly for getNextPlayer', () => {
      const rules = composeRules([baseLayer, nelloLayer]);
      const state = createTestState({
        trump: { type: 'nello' },
        winningBidder: 0
      });

      // Partner of bidder (0) is 2, so should skip from 1 to 3
      expect(rules.getNextPlayer(state, 1)).toBe(3);
      // Normal progression elsewhere
      expect(rules.getNextPlayer(state, 0)).toBe(1);
    });

    it('should compose base + nello correctly for isTrickComplete', () => {
      const rules = composeRules([baseLayer, nelloLayer]);
      const state = createTestState({
        trump: { type: 'nello' },
        currentTrick: [
          { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
          { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
          { player: 3, domino: { id: '3-4', high: 3, low: 4, points: 0 } }
        ]
      });

      expect(rules.isTrickComplete(state)).toBe(true); // 3 plays for nello
    });

    it('should compose base + plunge correctly for getTrumpSelector', () => {
      const rules = composeRules([baseLayer, plungeLayer]);
      const state = createTestState();
      const bid: Bid = { type: 'plunge', value: 4, player: 1 };

      // Partner of player 1 is player 3
      expect(rules.getTrumpSelector(state, bid)).toBe(3);
    });

    it('should compose base + sevens correctly for calculateTrickWinner', () => {
      const rules = composeRules([baseLayer, sevensLayer]);
      const state = createTestState({ trump: { type: 'sevens' } });
      const trick: Play[] = [
        { player: 0, domino: { id: '3-3', high: 3, low: 3, points: 0 } }, // 6 total (distance 1)
        { player: 1, domino: { id: '2-5', high: 2, low: 5, points: 0 } }, // 7 total (distance 0) - wins
        { player: 2, domino: { id: '0-0', high: 0, low: 0, points: 0 } }  // 0 total (distance 7)
      ];

      expect(rules.calculateTrickWinner(state, trick)).toBe(1); // Closest to 7
    });
  });

  describe('Base + multiple layers', () => {
    it('should compose base + nello + sevens correctly', () => {
      const rules = composeRules([baseLayer, nelloLayer, sevensLayer]);

      // Nello behavior when trump is nello
      const nelloState = createTestState({
        trump: { type: 'nello' },
        winningBidder: 0,
        currentTrick: [
          { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } },
          { player: 1, domino: { id: '1-2', high: 1, low: 2, points: 0 } },
          { player: 3, domino: { id: '3-4', high: 3, low: 4, points: 0 } }
        ]
      });
      expect(rules.isTrickComplete(nelloState)).toBe(true);

      // Sevens behavior when trump is sevens
      const sevensState = createTestState({ trump: { type: 'sevens' } });
      const trick: Play[] = [
        { player: 0, domino: { id: '3-4', high: 3, low: 4, points: 0 } }, // 7 total
        { player: 1, domino: { id: '2-3', high: 2, low: 3, points: 0 } }  // 5 total
      ];
      expect(rules.calculateTrickWinner(sevensState, trick)).toBe(0);
    });
  });

  describe('Layer order matters', () => {
    it('should apply layers left-to-right', () => {
      // Both layers can affect the same rule
      const rules1 = composeRules([baseLayer, nelloLayer]);
      const rules2 = composeRules([baseLayer, sevensLayer]);

      const nelloState = createTestState({
        trump: { type: 'nello' },
        winningBidder: 0
      });

      // nelloLayer is included, so getLedSuit should use nello rules
      const domino: Domino = { id: '2-2', high: 2, low: 2, points: 0 };
      expect(rules1.getLedSuit(nelloState, domino)).toBe(7); // Doubles = suit 7

      // sevensLayer doesn't override getLedSuit for nello
      const baseState = createTestState({
        trump: { type: 'suit', suit: 1 }
      });
      expect(rules2.getLedSuit(baseState, domino)).toBe(2); // Base behavior
    });

    it('should allow later layers to override earlier ones', () => {
      // Create a custom layer that always returns a fixed value
      const mockLayer = {
        name: 'mock',
        rules: {
          getNextPlayer: (_state: GameState, _current: number, _prev: number) => 99
        }
      };

      const rules = composeRules([baseLayer, nelloLayer, mockLayer]);
      const state = createTestState();

      // Mock layer is last, so it wins
      expect(rules.getNextPlayer(state, 0)).toBe(99);
    });
  });

  describe('All 7 rule methods compose correctly', () => {
    it('should compose all methods without error', () => {
      const rules = composeRules([baseLayer, nelloLayer, plungeLayer, sevensLayer]);

      expect(rules.getTrumpSelector).toBeDefined();
      expect(rules.getFirstLeader).toBeDefined();
      expect(rules.getNextPlayer).toBeDefined();
      expect(rules.isTrickComplete).toBeDefined();
      expect(rules.checkHandOutcome).toBeDefined();
      expect(rules.getLedSuit).toBeDefined();
      expect(rules.calculateTrickWinner).toBeDefined();
    });

    it('should handle empty layers gracefully', () => {
      const emptyLayer = { name: 'empty', rules: {} };
      const rules = composeRules([baseLayer, emptyLayer]);

      const state = createTestState();
      const bid: Bid = { type: 'marks', value: 2, player: 1 };

      // Should fall through to base
      expect(rules.getTrumpSelector(state, bid)).toBe(1);
      expect(rules.getNextPlayer(state, 0)).toBe(1);
    });
  });

  describe('Composition produces valid GameRules interface', () => {
    it('should return GameRules with all required methods', () => {
      const rules = composeRules([baseLayer]);

      // Type check - should have all methods
      expect(typeof rules.getTrumpSelector).toBe('function');
      expect(typeof rules.getFirstLeader).toBe('function');
      expect(typeof rules.getNextPlayer).toBe('function');
      expect(typeof rules.isTrickComplete).toBe('function');
      expect(typeof rules.checkHandOutcome).toBe('function');
      expect(typeof rules.getLedSuit).toBe('function');
      expect(typeof rules.calculateTrickWinner).toBe('function');
    });

    it('should work with no layers (edge case)', () => {
      const rules = composeRules([]);

      const state = createTestState();
      const bid: Bid = { type: 'marks', value: 2, player: 1 };

      // Should use base identity values from composeRules
      expect(rules.getTrumpSelector(state, bid)).toBe(1); // bid.player
      expect(rules.getFirstLeader(state, 2, { type: 'suit', suit: 1 })).toBe(2); // selector
    });

    it('should preserve function signatures across composition', () => {
      const rules = composeRules([baseLayer, nelloLayer]);

      const state = createTestState();
      const bid: Bid = { type: 'marks', value: 2, player: 1 };
      const trump: TrumpSelection = { type: 'suit', suit: 1 };
      const domino: Domino = { id: '0-1', high: 0, low: 1, points: 0 };
      const trick: Play[] = [
        { player: 0, domino: { id: '0-1', high: 0, low: 1, points: 0 } }
      ];

      // All methods should be callable with correct signatures
      expect(() => rules.getTrumpSelector(state, bid)).not.toThrow();
      expect(() => rules.getFirstLeader(state, 0, trump)).not.toThrow();
      expect(() => rules.getNextPlayer(state, 0)).not.toThrow();
      expect(() => rules.isTrickComplete(state)).not.toThrow();
      expect(() => rules.checkHandOutcome(state)).not.toThrow();
      expect(() => rules.getLedSuit(state, domino)).not.toThrow();
      expect(() => rules.calculateTrickWinner(state, trick)).not.toThrow();
    });
  });
});
