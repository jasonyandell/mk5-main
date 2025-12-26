import { describe, it, expect } from 'vitest';
import {
  createDominoes,
  isDouble
} from '../../game';
import { getLedSuitBase, rankInTrickBase } from '../../game/layers/rules-base';
import type { GameState } from '../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES, DOUBLES_AS_TRUMP } from '../../game/types';

describe('Feature: Doubles Treatment', () => {
  describe('Scenario: Standard Doubles Rules', () => {
    it('Given standard tournament rules apply When playing with doubles Then non-trump doubles belong to their natural suit', () => {
      const allDominoes = createDominoes();
      const doubles = allDominoes.filter(isDouble);

      // When trump is NOT doubles (e.g., trump is 3), doubles belong to their natural suit
      const trump = { type: 'suit', suit: TRES } as const;
      const state: GameState = { trump } as GameState;

      doubles.forEach(double => {
        const suit = getLedSuitBase(state, double);
        // Double's suit should be its pip value (unless it's the trump double)
        if (double.high === TRES) {
          expect(suit).toBe(DOUBLES_AS_TRUMP); // 3-3 is absorbed, leads suit 7
        } else {
          expect(suit).toBe(double.high); // Other doubles are their natural suit
        }
      });
    });

    it('And 6-6 is the highest six when sixes are not trump', () => {
      const trump = { type: 'suit', suit: TRES } as const; // Threes are trump, not sixes
      const state: GameState = { trump } as GameState;
      const sixDominoes = createDominoes().filter(domino =>
        getLedSuitBase(state, domino) === SIXES
      );

      // Get domino ranks for comparison (higher rank wins)
      const sixValues = sixDominoes.map(domino => ({
        domino,
        rank: rankInTrickBase(state, SIXES, domino)
      }));

      // Sort by rank descending
      sixValues.sort((a, b) => b.rank - a.rank);

      // 6-6 should be the highest six
      const highestSix = sixValues[0]?.domino;
      if (!highestSix) {
        throw new Error('No highest six found');
      }
      expect(highestSix.high).toBe(SIXES);
      expect(highestSix.low).toBe(SIXES);
      expect(isDouble(highestSix)).toBe(true);
    });

    it('And 5-5 is the highest five when fives are not trump', () => {
      const trump = { type: 'suit', suit: TRES } as const; // Threes are trump, not fives
      const state: GameState = { trump } as GameState;
      const fiveDominoes = createDominoes().filter(domino =>
        getLedSuitBase(state, domino) === FIVES
      );

      // Get domino ranks for comparison
      const fiveValues = fiveDominoes.map(domino => ({
        domino,
        rank: rankInTrickBase(state, FIVES, domino)
      }));

      // Sort by rank descending
      fiveValues.sort((a, b) => b.rank - a.rank);

      // 5-5 should be the highest five
      const highestFive = fiveValues[0]?.domino;
      if (!highestFive) {
        throw new Error('No highest five found');
      }
      expect(highestFive.high).toBe(FIVES);
      expect(highestFive.low).toBe(FIVES);
      expect(isDouble(highestFive)).toBe(true);
    });

    it('And when doubles are trump, only the seven doubles are trump', () => {
      const trump = { type: 'doubles' } as const; // Doubles are trump
      const state: GameState = { trump } as GameState;
      const allDominoes = createDominoes();

      // Find all trump dominoes
      const trumpDominoes = allDominoes.filter(domino =>
        getLedSuitBase(state, domino) === DOUBLES_AS_TRUMP
      );

      // Should be exactly 7 trump dominoes (all doubles)
      expect(trumpDominoes.length).toBe(7);

      // All trump dominoes should be doubles
      trumpDominoes.forEach(domino => {
        expect(isDouble(domino)).toBe(true);
      });

      // Non-doubles should not be trump
      const nonDoubles = allDominoes.filter(domino => !isDouble(domino));
      nonDoubles.forEach(domino => {
        const suit = getLedSuitBase(state, domino);
        expect(suit).not.toBe(DOUBLES_AS_TRUMP); // Should not be trump suit (doubles trump uses 7)
      });
    });

    it('And doubles are ranked 6-6 highest to 0-0 lowest when doubles are trump', () => {
      const trump = { type: 'doubles' } as const; // Doubles are trump
      const state: GameState = { trump } as GameState;
      const allDominoes = createDominoes();
      const doubles = allDominoes.filter(isDouble);

      // Get ranks and sort
      const doubleValues = doubles.map(domino => ({
        domino,
        rank: rankInTrickBase(state, DOUBLES_AS_TRUMP, domino)
      }));

      doubleValues.sort((a, b) => b.rank - a.rank);

      // Should be ordered from 6-6 down to 0-0
      const expectedOrder = [SIXES, FIVES, FOURS, TRES, DEUCES, ACES, BLANKS];
      doubleValues.forEach((item, index) => {
        expect(item.domino.high).toBe(expectedOrder[index]);
        expect(item.domino.low).toBe(expectedOrder[index]);
      });
    });
  });
});