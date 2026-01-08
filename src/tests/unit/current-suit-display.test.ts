import { describe, test, expect } from 'vitest';
import { getCurrentSuit } from '../../game/core/rules';
import { createInitialState } from '../../game/core/state';
import type { TrumpSelection, LedSuitOrNone } from '../../game/types';
import {
  BLANKS, FOURS, FIVES, SIXES,
  CALLED, NO_LEAD_SUIT
} from '../../game/types';

describe('Current Suit Display', () => {
  const doublesAreTrump = { type: 'doubles' } as const;
  const fivesAreTrump = { type: 'suit', suit: FIVES } as const;
  const sixesAreTrump = { type: 'suit', suit: SIXES } as const;
  const noTrump = { type: 'no-trump' } as const;

  test('should return "None" when no domino is led', () => {
    const state = createInitialState();
    state.currentSuit = NO_LEAD_SUIT;
    state.trump = doublesAreTrump;
    expect(getCurrentSuit(state)).toBe('None (no domino led)');
  });

  test('should return "None" when trump is not set', () => {
    const state = createInitialState();
    state.currentSuit = SIXES; // Some suit was led
    state.trump = { type: 'not-selected' };
    expect(getCurrentSuit(state)).toBe('None (no trump set)');
  });

  describe('When doubles are trump', () => {
    test('6-6 leads -> Doubles (Trump)', () => {
      const state = createInitialState();
      state.trump = doublesAreTrump;
      state.currentSuit = CALLED; // Doubles led
      expect(getCurrentSuit(state)).toBe('Doubles (Trump)');
    });

    test('3-3 leads -> Doubles (Trump)', () => {
      const state = createInitialState();
      state.trump = doublesAreTrump;
      state.currentSuit = CALLED; // Doubles led
      expect(getCurrentSuit(state)).toBe('Doubles (Trump)');
    });

    test('6-5 leads -> Sixes (higher end)', () => {
      const state = createInitialState();
      state.trump = doublesAreTrump;
      state.currentSuit = SIXES; // Sixes led (higher end)
      expect(getCurrentSuit(state)).toBe('Sixes');
    });

    test('4-2 leads -> Fours (higher end)', () => {
      const state = createInitialState();
      state.trump = doublesAreTrump;
      state.currentSuit = FOURS; // Fours led (higher end)
      expect(getCurrentSuit(state)).toBe('Fours');
    });
  });

  describe('When specific suit is trump (not doubles)', () => {
    test('6-6 leads with 6s trump -> Sixes (Trump)', () => {
      const state = createInitialState();
      state.trump = sixesAreTrump;
      state.currentSuit = SIXES; // Sixes led (trump)
      expect(getCurrentSuit(state)).toBe('Sixes (Trump)');
    });

    test('5-5 leads with 6s trump -> Fives', () => {
      const state = createInitialState();
      state.trump = sixesAreTrump;
      state.currentSuit = FIVES; // Fives led (not trump)
      expect(getCurrentSuit(state)).toBe('Fives');
    });

    test('6-5 leads with 6s trump -> Sixes (Trump)', () => {
      const state = createInitialState();
      state.trump = sixesAreTrump;
      state.currentSuit = SIXES; // Sixes led (trump)
      expect(getCurrentSuit(state)).toBe('Sixes (Trump)');
    });

    test('6-5 leads with 5s trump -> Fives (Trump)', () => {
      const state = createInitialState();
      state.trump = fivesAreTrump;
      state.currentSuit = FIVES; // Fives led (trump)
      expect(getCurrentSuit(state)).toBe('Fives (Trump)');
    });

    test('4-2 leads with 5s trump -> Fours', () => {
      const state = createInitialState();
      state.trump = fivesAreTrump;
      state.currentSuit = FOURS; // Fours led (not trump)
      expect(getCurrentSuit(state)).toBe('Fours');
    });
  });

  describe('No trump (follow-me)', () => {
    test('6-6 leads -> Sixes', () => {
      const state = createInitialState();
      state.trump = noTrump;
      state.currentSuit = SIXES; // Sixes led
      expect(getCurrentSuit(state)).toBe('Sixes');
    });

    test('6-5 leads -> Sixes (higher end)', () => {
      const state = createInitialState();
      state.trump = noTrump;
      state.currentSuit = SIXES; // Sixes led
      expect(getCurrentSuit(state)).toBe('Sixes');
    });

    test('0-0 leads -> Blanks', () => {
      const state = createInitialState();
      state.trump = noTrump;
      state.currentSuit = BLANKS; // Blanks led
      expect(getCurrentSuit(state)).toBe('Blanks');
    });
  });

  describe('All suit names', () => {
    const testCases = [
      { domino: { high: 0, low: 0, id: "0-0" }, expected: 'Blanks' },
      { domino: { high: 1, low: 1, id: "1-1" }, expected: 'Ones' },
      { domino: { high: 2, low: 2, id: "2-2" }, expected: 'Twos' },
      { domino: { high: 3, low: 3, id: "3-3" }, expected: 'Threes' },
      { domino: { high: 4, low: 4, id: "4-4" }, expected: 'Fours' },
      { domino: { high: 5, low: 5, id: "5-5" }, expected: 'Fives' },
      { domino: { high: 6, low: 6, id: "6-6" }, expected: 'Sixes' }
    ];

    testCases.forEach(({ domino, expected }) => {
      test(`${domino.id} with no trump -> ${expected}`, () => {
        const state = createInitialState();
        state.trump = noTrump;
        state.currentSuit = domino.high as LedSuitOrNone; // Use the domino's suit
        expect(getCurrentSuit(state)).toBe(expected);
      });
    });
  });

  describe('Edge cases', () => {
    test('Complex scenario - 6-4 led with 4s trump should show Fours (Trump)', () => {
      const state = createInitialState();
      state.trump = { type: 'suit', suit: FOURS } as TrumpSelection; // 4s are trump
      state.currentSuit = FOURS; // 4s were led (trump)
      expect(getCurrentSuit(state)).toBe('Fours (Trump)');
    });

    test('Domino with trump number led should show trump', () => {
      const state = createInitialState();
      state.trump = { type: 'suit', suit: FIVES } as TrumpSelection; // 5s are trump
      state.currentSuit = FIVES; // 5s were led (trump)
      expect(getCurrentSuit(state)).toBe('Fives (Trump)');
    });
  });
});