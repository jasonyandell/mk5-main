import { describe, it, expect } from 'vitest';
import { isValidPlay, getTrickWinner, getTrickPoints } from '../../game/core/rules';
import { Trump, Play, Domino } from '../../game/types';
import { TRUMP_SUITS } from '../../game/constants';

// Helper function to create dominoes for testing
function createDomino(high: number, low: number): Domino {
  return {
    id: `${high}-${low}`,
    high,
    low
  };
}

describe('Trick Validation', () => {
  const trump: Trump = { suit: TRUMP_SUITS.BLANKS, followsSuit: false };
  
  describe('isValidPlay', () => {
    it('should allow any domino for opening lead', () => {
      const hand = [createDomino(0, 0), createDomino(1, 2), createDomino(3, 4)];
      const currentTrick: Play[] = [];
      
      hand.forEach(domino => {
        expect(isValidPlay(domino, hand, currentTrick, trump)).toBe(true);
      });
    });

    it('should require following suit when possible', () => {
      const hand = [
        createDomino(0, 1), // has blank
        createDomino(2, 3),  // no blank
        createDomino(4, 5)   // no blank
      ];
      const currentTrick: Play[] = [
        { player: 0, domino: createDomino(0, 2) } // lead with blank
      ];
      
      expect(isValidPlay(hand[0], hand, currentTrick, trump)).toBe(true);  // must follow
      expect(isValidPlay(hand[1], hand, currentTrick, trump)).toBe(false); // can't play off-suit
      expect(isValidPlay(hand[2], hand, currentTrick, trump)).toBe(false); // can't play off-suit
    });

    it('should allow any domino when cannot follow suit', () => {
      const hand = [
        createDomino(1, 2), // no blank
        createDomino(3, 4), // no blank
        createDomino(5, 6)  // no blank
      ];
      const currentTrick: Play[] = [
        { player: 0, domino: createDomino(0, 0) } // lead with double blank
      ];
      
      hand.forEach(domino => {
        expect(isValidPlay(domino, hand, currentTrick, trump)).toBe(true);
      });
    });

    it('should handle trump suit correctly', () => {
      const trumpSuit: Trump = { suit: TRUMP_SUITS.ONES, followsSuit: false };
      const hand = [
        createDomino(1, 1), // trump
        createDomino(2, 3), // not trump
        createDomino(0, 4)  // not trump
      ];
      const currentTrick: Play[] = [
        { player: 0, domino: createDomino(1, 2) } // lead with trump suit
      ];
      
      expect(isValidPlay(hand[0], hand, currentTrick, trumpSuit)).toBe(true);  // can follow trump
      expect(isValidPlay(hand[1], hand, currentTrick, trumpSuit)).toBe(false); // must follow trump
      expect(isValidPlay(hand[2], hand, currentTrick, trumpSuit)).toBe(false); // must follow trump
    });
  });

  describe('getTrickWinner', () => {
    it('should identify winner of basic trick', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(1, 2) }, // leads with 2s
        { player: 1, domino: createDomino(1, 4) }, // 4s - can't follow suit
        { player: 2, domino: createDomino(1, 6) }, // 6s - can't follow suit  
        { player: 3, domino: createDomino(2, 3) }  // 3s - CAN follow suit with higher value
      ];
      
      const winner = getTrickWinner(trick, trump);
      expect(winner).toBe(3); // Player 3 wins by following suit with higher value
    });

    it('should handle trump plays correctly', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(1, 2) },
        { player: 1, domino: createDomino(0, 4) }, // trump (blank)
        { player: 2, domino: createDomino(1, 6) },
        { player: 3, domino: createDomino(2, 3) }
      ];
      
      const winner = getTrickWinner(trick, trump);
      expect(winner).toBe(1); // Player 1 with trump
    });

    it('should handle multiple trump plays', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(0, 1) }, // trump
        { player: 1, domino: createDomino(0, 4) }, // trump
        { player: 2, domino: createDomino(0, 6) }, // trump (highest)
        { player: 3, domino: createDomino(2, 3) }
      ];
      
      const winner = getTrickWinner(trick, trump);
      expect(winner).toBe(2); // Player 2 with highest trump
    });

    it('should handle double blank (highest trump)', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(0, 1) }, // trump
        { player: 1, domino: createDomino(0, 0) }, // double blank (highest trump)
        { player: 2, domino: createDomino(0, 6) }, // trump
        { player: 3, domino: createDomino(2, 3) }
      ];
      
      const winner = getTrickWinner(trick, trump);
      expect(winner).toBe(1); // Player 1 with double blank
    });
  });

  describe('getTrickPoints', () => {
    it('should calculate points correctly for basic dominoes', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(1, 4) }, // 5 points
        { player: 1, domino: createDomino(2, 3) }, // 5 points
        { player: 2, domino: createDomino(0, 0) }, // 0 points
        { player: 3, domino: createDomino(6, 6) }  // 0 points
      ];
      
      expect(getTrickPoints(trick)).toBe(10);
    });

    it('should handle special point dominoes', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(5, 0) }, // 5 points
        { player: 1, domino: createDomino(4, 1) }, // 5 points  
        { player: 2, domino: createDomino(3, 2) }, // 5 points
        { player: 3, domino: createDomino(6, 4) }  // 10 points
      ];
      
      expect(getTrickPoints(trick)).toBe(25);
    });

    it('should handle empty trick', () => {
      expect(getTrickPoints([])).toBe(0);
    });
  });
});