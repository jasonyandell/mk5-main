import { describe, it, expect } from 'vitest';
import { 
  createDominoes, 
  getDominoSuit, 
  getDominoValue, 
  isDouble
} from '../../game';

describe('Feature: Doubles Treatment', () => {
  describe('Scenario: Standard Doubles Rules', () => {
    it('Given standard tournament rules apply When playing with doubles Then doubles belong to their natural suit', () => {
      const allDominoes = createDominoes();
      const doubles = allDominoes.filter(isDouble);
      
      // When trump is NOT doubles (e.g., trump is 3), doubles belong to their natural suit
      const trump = 3;
      
      doubles.forEach(double => {
        const suit = getDominoSuit(double, trump);
        // Double's suit should be its pip value (unless it's the trump double)
        if (double.high === trump) {
          expect(suit).toBe(trump); // 3-3 would be trump suit
        } else {
          expect(suit).toBe(double.high); // Other doubles are their natural suit
        }
      });
    });

    it('And 6-6 is the highest six when sixes are not trump', () => {
      const trump = 3; // Threes are trump, not sixes
      const sixDominoes = createDominoes().filter(domino => 
        getDominoSuit(domino, trump) === 6
      );
      
      // Get domino values for comparison (higher value wins)
      const sixValues = sixDominoes.map(domino => ({
        domino,
        value: getDominoValue(domino, trump)
      }));
      
      // Sort by value descending
      sixValues.sort((a, b) => b.value - a.value);
      
      // 6-6 should be the highest six
      const highestSix = sixValues[0].domino;
      expect(highestSix.high).toBe(6);
      expect(highestSix.low).toBe(6);
      expect(isDouble(highestSix)).toBe(true);
    });

    it('And 5-5 is the highest five when fives are not trump', () => {
      const trump = 3; // Threes are trump, not fives
      const fiveDominoes = createDominoes().filter(domino => 
        getDominoSuit(domino, trump) === 5
      );
      
      // Get domino values for comparison
      const fiveValues = fiveDominoes.map(domino => ({
        domino,
        value: getDominoValue(domino, trump)
      }));
      
      // Sort by value descending
      fiveValues.sort((a, b) => b.value - a.value);
      
      // 5-5 should be the highest five
      const highestFive = fiveValues[0].domino;
      expect(highestFive.high).toBe(5);
      expect(highestFive.low).toBe(5);
      expect(isDouble(highestFive)).toBe(true);
    });

    it('And when doubles are trump, only the seven doubles are trump', () => {
      const trump = 7; // Doubles are trump
      const allDominoes = createDominoes();
      
      // Find all trump dominoes
      const trumpDominoes = allDominoes.filter(domino => 
        getDominoSuit(domino, trump) === 7
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
        const suit = getDominoSuit(domino, trump);
        expect(suit).not.toBe(7); // Should not be trump suit
      });
    });

    it('And doubles are ranked 6-6 highest to 0-0 lowest when doubles are trump', () => {
      const trump = 7; // Doubles are trump
      const allDominoes = createDominoes();
      const doubles = allDominoes.filter(isDouble);
      
      // Get values and sort
      const doubleValues = doubles.map(domino => ({
        domino,
        value: getDominoValue(domino, trump)
      }));
      
      doubleValues.sort((a, b) => b.value - a.value);
      
      // Should be ordered from 6-6 down to 0-0
      const expectedOrder = [6, 5, 4, 3, 2, 1, 0];
      doubleValues.forEach((item, index) => {
        expect(item.domino.high).toBe(expectedOrder[index]);
        expect(item.domino.low).toBe(expectedOrder[index]);
      });
    });
  });
});