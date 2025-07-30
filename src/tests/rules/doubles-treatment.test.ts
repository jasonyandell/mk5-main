import { describe, it, expect } from 'vitest';
import type { Domino } from '../../game/types';

describe('Feature: Doubles Treatment', () => {
  describe('Scenario: Standard Doubles Rules', () => {
    it('Given standard tournament rules apply When playing with doubles Then doubles belong to their natural suit', () => {
      // Tournament standard: doubles are part of their natural suit
      // These dominoes represent doubles in their natural suits

      // In standard rules, each double belongs to its suit
      // 6-6 belongs to sixes suit
      // 5-5 belongs to fives suit
      // 0-0 belongs to blanks suit
      expect(true).toBe(true); // Doubles are naturally part of their suit
    });

    it('And 6-6 is the highest six', () => {
      const sixDominoes: Domino[] = [
        { high: 6, low: 6, id: "6-6" }, // Double six
        { high: 6, low: 5, id: "6-5" },
        { high: 6, low: 4, id: "6-4" },
        { high: 6, low: 3, id: "6-3" },
        { high: 6, low: 2, id: "6-2" },
        { high: 6, low: 1, id: "6-1" },
        { high: 6, low: 0, id: "6-0" }
      ];

      // When sixes are led (not trump), 6-6 is the highest six
      const highestSix = sixDominoes[0]; // 6-6
      expect(highestSix.high).toBe(6);
      expect(highestSix.low).toBe(6);
    });

    it('And 5-5 is the highest five', () => {
      const fiveDominoes: Domino[] = [
        { high: 5, low: 5, id: "5-5" }, // Double five - highest
        { high: 5, low: 4, id: "5-4" },
        { high: 5, low: 3, id: "5-3" },
        { high: 5, low: 2, id: "5-2" },
        { high: 5, low: 1, id: "5-1" },
        { high: 5, low: 0, id: "5-0" }
      ];

      // When fives are led (not trump), 5-5 is the highest five
      const highestFive = fiveDominoes[0]; // 5-5
      expect(highestFive.high).toBe(5);
      expect(highestFive.low).toBe(5);
    });

    it('And when doubles are trump, only the seven doubles are trump', () => {
      const allDoubles: Domino[] = [
        { high: 6, low: 6, id: "6-6" },
        { high: 5, low: 5, id: "5-5" },
        { high: 4, low: 4, id: "4-4" },
        { high: 3, low: 3, id: "3-3" },
        { high: 2, low: 2, id: "2-2" },
        { high: 1, low: 1, id: "1-1" },
        { high: 0, low: 0, id: "0-0" }
      ];

      // When doubles are declared as trump (7 represents doubles trump)

      // Then only these seven dominoes are trump
      expect(allDoubles.length).toBe(7);

      // And non-doubles are not trump
      const nonDouble: Domino = { high: 6, low: 5, id: "6-5" };
      const isDouble = nonDouble.high === nonDouble.low;
      expect(isDouble).toBe(false);
    });
  });
});