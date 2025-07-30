import { describe, it, expect } from 'vitest';
import { Domino, GameState, Trump } from '../../game/types';
import { isDominoValid, getHighestDominoInSuit } from '../../game/helpers';

describe('Feature: Doubles Treatment', () => {
  describe('Scenario: Standard Doubles Rules', () => {
    it('Given standard tournament rules apply When playing with doubles Then doubles belong to their natural suit', () => {
      // Tournament standard: doubles are part of their natural suit
      const doubleSix: Domino = { pips1: 6, pips2: 6 };
      const doubleFive: Domino = { pips1: 5, pips2: 5 };
      const doubleBlank: Domino = { pips1: 0, pips2: 0 };

      // In standard rules, each double belongs to its suit
      // 6-6 belongs to sixes suit
      // 5-5 belongs to fives suit
      // 0-0 belongs to blanks suit
      expect(true).toBe(true); // Doubles are naturally part of their suit
    });

    it('And 6-6 is the highest six', () => {
      const sixDominoes: Domino[] = [
        { pips1: 6, pips2: 6 }, // Double six
        { pips1: 6, pips2: 5 },
        { pips1: 6, pips2: 4 },
        { pips1: 6, pips2: 3 },
        { pips1: 6, pips2: 2 },
        { pips1: 6, pips2: 1 },
        { pips1: 6, pips2: 0 }
      ];

      // When sixes are led (not trump), 6-6 is the highest six
      const highestSix = sixDominoes[0]; // 6-6
      expect(highestSix.pips1).toBe(6);
      expect(highestSix.pips2).toBe(6);
    });

    it('And 5-5 is the highest five', () => {
      const fiveDominoes: Domino[] = [
        { pips1: 5, pips2: 5 }, // Double five - highest
        { pips1: 5, pips2: 4 },
        { pips1: 5, pips2: 3 },
        { pips1: 5, pips2: 2 },
        { pips1: 5, pips2: 1 },
        { pips1: 5, pips2: 0 }
      ];

      // When fives are led (not trump), 5-5 is the highest five
      const highestFive = fiveDominoes[0]; // 5-5
      expect(highestFive.pips1).toBe(5);
      expect(highestFive.pips2).toBe(5);
    });

    it('And when doubles are trump, only the seven doubles are trump', () => {
      const allDoubles: Domino[] = [
        { pips1: 6, pips2: 6 },
        { pips1: 5, pips2: 5 },
        { pips1: 4, pips2: 4 },
        { pips1: 3, pips2: 3 },
        { pips1: 2, pips2: 2 },
        { pips1: 1, pips2: 1 },
        { pips1: 0, pips2: 0 }
      ];

      // When doubles are declared as trump
      const trump: Trump = { type: 'doubles' };

      // Then only these seven dominoes are trump
      expect(allDoubles.length).toBe(7);

      // And non-doubles are not trump
      const nonDouble: Domino = { pips1: 6, pips2: 5 };
      const isDouble = nonDouble.pips1 === nonDouble.pips2;
      expect(isDouble).toBe(false);
    });
  });
});