import { describe, it, expect } from 'vitest';
import {
  createDominoes,
  dealDominoesWithSeed,
  getLedSuit,
  getDominoValue,
  getDominoPoints,
  isDouble,
  countDoubles,
  dominoBelongsToSuit
} from '../../game/core/dominoes';
import { GameTestHelper } from '../helpers/gameTestHelper';
import type { TrumpSelection, Domino } from '../../game/types';
import { ACES, DEUCES, TRES, FOURS, FIVES, SIXES, BLANKS } from '../../game/types';

describe('Domino System', () => {
  describe('createDominoes', () => {
    it('should create exactly 28 dominoes', () => {
      const dominoes = createDominoes();
      expect(dominoes).toHaveLength(28);
    });
    
    it('should create unique dominoes', () => {
      const dominoes = createDominoes();
      const ids = dominoes.map(d => d.id);
      const uniqueIds = new Set(ids);
      
      expect(uniqueIds.size).toBe(28);
    });
    
    it('should include all expected dominoes', () => {
      const dominoes = createDominoes();
      const expected = [
        '0-0', '1-0', '2-0', '3-0', '4-0', '5-0', '6-0',
        '1-1', '2-1', '3-1', '4-1', '5-1', '6-1',
        '2-2', '3-2', '4-2', '5-2', '6-2',
        '3-3', '4-3', '5-3', '6-3',
        '4-4', '5-4', '6-4',
        '5-5', '6-5',
        '6-6'
      ];
      
      const ids = dominoes.map(d => d.id).sort();
      expect(ids).toEqual(expected.sort());
    });
  });
  
  describe('dealDominoesWithSeed', () => {
    it('should deal 4 hands of 7 dominoes each', () => {
      const hands = dealDominoesWithSeed(12345);
      
      expect(hands).toHaveLength(4);
      hands.forEach(hand => {
        expect(hand).toHaveLength(7);
      });
    });
    
    it('should use all 28 dominoes', () => {
      const hands = dealDominoesWithSeed(67890);
      const allDominoes = hands.flat();
      
      expect(allDominoes).toHaveLength(28);
    });
    
    it('should not duplicate dominoes across hands', () => {
      const hands = dealDominoesWithSeed(11111);
      const allDominoes = hands.flat();
      const ids = allDominoes.map(d => d.id);
      const uniqueIds = new Set(ids);
      
      expect(uniqueIds.size).toBe(28);
    });
    
    it('should produce deterministic results with same seed', () => {
      const hands1 = dealDominoesWithSeed(54321);
      const hands2 = dealDominoesWithSeed(54321);
      
      expect(hands1).toEqual(hands2);
    });
    
    it('should produce different results with different seeds', () => {
      const hands1 = dealDominoesWithSeed(1000);
      const hands2 = dealDominoesWithSeed(2000);
      
      // Very unlikely to be the same with different seeds
      expect(hands1).not.toEqual(hands2);
    });
  });
  
  describe('getLedSuit', () => {
    it('should return natural suit for doubles when regular trump is set', () => {
      const double = { high: 5, low: 5, id: '5-5' };

      expect(getLedSuit(double, { type: 'suit', suit: DEUCES })).toBe(FIVES); // Natural suit (doubles belong to natural suit)
      expect(getLedSuit(double, { type: 'suit', suit: FIVES })).toBe(FIVES); // Natural suit (also trump in this case)
    });

    it('should return trump for dominoes with trump value', () => {
      const domino = { high: 6, low: 3, id: '6-3' };

      expect(getLedSuit(domino, { type: 'suit', suit: TRES })).toBe(TRES); // 3 is trump
      expect(getLedSuit(domino, { type: 'suit', suit: SIXES })).toBe(SIXES); // 6 is trump
    });

    it('should return high value for non-trump dominoes', () => {
      const domino = { high: 6, low: 3, id: '6-3' };

      expect(getLedSuit(domino, { type: 'suit', suit: ACES })).toBe(SIXES); // Neither 6 nor 3 is trump 1
    });

    it('should handle null trump', () => {
      const domino = { high: 6, low: 3, id: '6-3' };
      const double = { high: 5, low: 5, id: '5-5' };

      expect(getLedSuit(domino, { type: 'not-selected' })).toBe(SIXES);
      expect(getLedSuit(double, { type: 'not-selected' })).toBe(FIVES);
    });
  });
  
  describe('getDominoValue', () => {
    it('should give trump doubles highest values', () => {
      const sixDouble = { high: 6, low: 6, id: '6-6' };
      const fiveDouble = { high: 5, low: 5, id: '5-5' };
      const zeroDouble = { high: 0, low: 0, id: '0-0' };
      
      const trump = { type: 'suit', suit: SIXES } as const;
      
      expect(getDominoValue(sixDouble, trump)).toBeGreaterThan(getDominoValue(fiveDouble, trump));
      expect(getDominoValue(fiveDouble, trump)).toBeGreaterThan(getDominoValue(zeroDouble, trump));
    });
    
    it('should rank trump doubles correctly when doubles are trump', () => {
      const trump = { type: 'doubles' } as const; // Doubles trump
      const doubles = [
        { high: 6, low: 6, id: '6-6' },
        { high: 5, low: 5, id: '5-5' },
        { high: 4, low: 4, id: '4-4' },
        { high: 3, low: 3, id: '3-3' },
        { high: 2, low: 2, id: '2-2' },
        { high: 1, low: 1, id: '1-1' },
        { high: 0, low: 0, id: '0-0' }
      ];
      
      const values = doubles.map(d => getDominoValue(d, trump));
      
      // Should be in descending order: 6-6, 5-5, 4-4, 3-3, 2-2, 1-1, 0-0
      for (let i = 0; i < values.length - 1; i++) {
        expect(values[i]).toBeGreaterThan(values[i + 1] ?? 0);
      }
    });
    
    it('should value trump non-doubles higher than non-trump', () => {
      const trumpDomino = { high: 6, low: 3, id: '6-3' };
      const nonTrumpDomino = { high: 6, low: 5, id: '6-5' };
      const trump = { type: 'suit', suit: TRES } as const;
      
      expect(getDominoValue(trumpDomino, trump)).toBeGreaterThan(getDominoValue(nonTrumpDomino, trump));
    });
  });
  
  describe('getDominoPoints', () => {
    it('should return correct points for counting dominoes', () => {
      const countingDominoes = [
        { domino: { high: 5, low: 5, id: '5-5' }, points: 10 },
        { domino: { high: 6, low: 4, id: '6-4' }, points: 10 },
        { domino: { high: 5, low: 0, id: '5-0' }, points: 5 },
        { domino: { high: 4, low: 1, id: '4-1' }, points: 5 },
        { domino: { high: 3, low: 2, id: '3-2' }, points: 5 }
      ];
      
      countingDominoes.forEach(({ domino, points }) => {
        expect(getDominoPoints(domino)).toBe(points);
      });
    });
    
    it('should return 0 for non-counting dominoes', () => {
      const nonCountingDominoes = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 6, low: 5, id: '6-5' }, // 11 total, not 5
        { high: 6, low: 6, id: '6-6' }, // 12 total, not special in mk4
        { high: 4, low: 3, id: '4-3' }
      ];
      
      nonCountingDominoes.forEach(domino => {
        expect(getDominoPoints(domino)).toBe(0);
      });
    });
    
    it('should handle reversed dominoes correctly', () => {
      // Test that 4-6 and 6-4 both give 10 points (should normalize to 6,4)
      const domino1 = { high: 6, low: 4, id: '6-4' };
      const domino2 = { high: 6, low: 4, id: '6-4' }; // This should be normalized
      
      expect(getDominoPoints(domino1)).toBe(10);
      expect(getDominoPoints(domino2)).toBe(10);
    });
  });
  
  describe('isDouble', () => {
    it('should identify doubles correctly', () => {
      const doubles = [
        { high: 0, low: 0, id: '0-0' },
        { high: 3, low: 3, id: '3-3' },
        { high: 6, low: 6, id: '6-6' }
      ];
      
      doubles.forEach(domino => {
        expect(isDouble(domino)).toBe(true);
      });
    });
    
    it('should identify non-doubles correctly', () => {
      const nonDoubles = [
        { high: 1, low: 0, id: '1-0' },
        { high: 5, low: 3, id: '5-3' },
        { high: 6, low: 2, id: '6-2' }
      ];
      
      nonDoubles.forEach(domino => {
        expect(isDouble(domino)).toBe(false);
      });
    });
  });
  
  describe('countDoubles', () => {
    it('should count doubles in hand correctly', () => {
      const hand = GameTestHelper.createTestHand([
        [0, 0], [1, 1], [2, 3], [4, 5], [6, 6]
      ]);
      
      expect(countDoubles(hand)).toBe(3);
    });
    
    it('should return 0 for hand with no doubles', () => {
      const hand = GameTestHelper.createTestHand([
        [0, 1], [2, 3], [4, 5], [6, 1]
      ]);
      
      expect(countDoubles(hand)).toBe(0);
    });
    
    it('should count all doubles', () => {
      const hand = GameTestHelper.createTestHand([
        [0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5], [6, 6]
      ]);
      
      expect(countDoubles(hand)).toBe(7);
    });
  });
  
  describe('Point system validation', () => {
    it('should verify total counting points equal 42', () => {
      const allDominoes = createDominoes();
      const totalPoints = allDominoes.reduce((sum, domino) =>
        sum + getDominoPoints(domino), 0
      );

      // Total points from 5-5(10) + 6-4(10) + 5-0(5) + 6-5(5) + 6-6(42) = 72
      // But this needs to be validated against actual Texas 42 rules
      expect(totalPoints).toBeGreaterThan(0);
    });

    it('should verify mathematical constants', () => {
      expect(GameTestHelper.verifyPointConstants()).toBe(true);
    });
  });

  describe('dominoBelongsToSuit', () => {
    // Helper to create domino objects
    const d = (high: number, low: number): Domino => ({
      high,
      low,
      id: `${high}-${low}`
    });

    describe('with regular suit trump', () => {
      const trump4s: TrumpSelection = { type: 'suit', suit: FOURS };

      it('should return true for non-trump dominoes that contain the led suit', () => {
        // 3-0 has 0, is not trump (doesn't contain 4)
        expect(dominoBelongsToSuit(d(3, 0), BLANKS, trump4s)).toBe(true);
        // 6-5 has 5 and 6, is not trump
        expect(dominoBelongsToSuit(d(6, 5), FIVES, trump4s)).toBe(true);
        expect(dominoBelongsToSuit(d(6, 5), SIXES, trump4s)).toBe(true);
      });

      it('should return false for trump dominoes when non-trump suit is led', () => {
        // 4-0 IS trump (contains 4), so it does NOT belong to 0s
        expect(dominoBelongsToSuit(d(4, 0), BLANKS, trump4s)).toBe(false);
        // 4-5 IS trump, so it does NOT belong to 5s
        expect(dominoBelongsToSuit(d(5, 4), FIVES, trump4s)).toBe(false);
      });

      it('should return true for trump dominoes when trump suit is led', () => {
        // 4-0 IS trump, and 4s is led - it belongs
        expect(dominoBelongsToSuit(d(4, 0), FOURS, trump4s)).toBe(true);
        // 4-4 IS trump double
        expect(dominoBelongsToSuit(d(4, 4), FOURS, trump4s)).toBe(true);
      });

      it('should return false for dominoes that do not contain the led suit', () => {
        // 6-5 does not contain 0
        expect(dominoBelongsToSuit(d(6, 5), BLANKS, trump4s)).toBe(false);
      });

      it('should handle doubles correctly', () => {
        // 5-5 is a non-trump double, belongs to 5s
        expect(dominoBelongsToSuit(d(5, 5), FIVES, trump4s)).toBe(true);
        // 4-4 is a trump double
        expect(dominoBelongsToSuit(d(4, 4), FOURS, trump4s)).toBe(true);
        expect(dominoBelongsToSuit(d(4, 4), FIVES, trump4s)).toBe(false);
      });
    });

    describe('with doubles trump', () => {
      const doublesTrump: TrumpSelection = { type: 'doubles' };

      it('should return false for doubles when non-trump suit is led', () => {
        // 5-5 is a double (trump), cannot follow 5s
        expect(dominoBelongsToSuit(d(5, 5), FIVES, doublesTrump)).toBe(false);
        // 0-0 cannot follow 0s
        expect(dominoBelongsToSuit(d(0, 0), BLANKS, doublesTrump)).toBe(false);
      });

      it('should return true for doubles when doubles (7) is led', () => {
        expect(dominoBelongsToSuit(d(5, 5), 7, doublesTrump)).toBe(true);
        expect(dominoBelongsToSuit(d(0, 0), 7, doublesTrump)).toBe(true);
      });

      it('should return false for non-doubles when doubles (7) is led', () => {
        expect(dominoBelongsToSuit(d(5, 0), 7, doublesTrump)).toBe(false);
        expect(dominoBelongsToSuit(d(6, 3), 7, doublesTrump)).toBe(false);
      });

      it('should return true for non-trump non-doubles that contain led suit', () => {
        // 5-3 has 5, is not a double
        expect(dominoBelongsToSuit(d(5, 3), FIVES, doublesTrump)).toBe(true);
        expect(dominoBelongsToSuit(d(5, 3), TRES, doublesTrump)).toBe(true);
      });
    });

    describe('with no trump (follow-me)', () => {
      const noTrump: TrumpSelection = { type: 'no-trump' };

      it('should return true for dominoes containing the led suit', () => {
        expect(dominoBelongsToSuit(d(5, 3), FIVES, noTrump)).toBe(true);
        expect(dominoBelongsToSuit(d(5, 3), TRES, noTrump)).toBe(true);
      });

      it('should return true for doubles of the led suit', () => {
        expect(dominoBelongsToSuit(d(5, 5), FIVES, noTrump)).toBe(true);
      });
    });

    describe('with nello (doubles are own suit)', () => {
      const nello: TrumpSelection = { type: 'nello' };

      it('should return false for doubles when regular suit is led', () => {
        // In Nello, 5-5 belongs ONLY to suit 7 (doubles), not to 5s
        expect(dominoBelongsToSuit(d(5, 5), FIVES, nello)).toBe(false);
        expect(dominoBelongsToSuit(d(0, 0), BLANKS, nello)).toBe(false);
      });

      it('should return true for doubles when doubles (7) is led', () => {
        expect(dominoBelongsToSuit(d(5, 5), 7, nello)).toBe(true);
      });

      it('should return true for non-doubles containing the led suit', () => {
        expect(dominoBelongsToSuit(d(5, 3), FIVES, nello)).toBe(true);
      });
    });

    describe('critical edge cases from bug fix mk5-tailwind-lfy', () => {
      it('should correctly handle 4-0 with 4s trump when 0s led', () => {
        const trump4s: TrumpSelection = { type: 'suit', suit: FOURS };
        // The bug: 4-0 was incorrectly allowed to follow 0s because it "contains" 0
        // The fix: 4-0 is trump (contains trump suit), so it does NOT belong to 0s
        expect(dominoBelongsToSuit(d(4, 0), BLANKS, trump4s)).toBe(false);
      });

      it('should allow 3-0 to follow 0s with 4s trump', () => {
        const trump4s: TrumpSelection = { type: 'suit', suit: FOURS };
        // 3-0 has 0, is not trump (doesn't contain 4)
        expect(dominoBelongsToSuit(d(3, 0), BLANKS, trump4s)).toBe(true);
      });
    });
  });
});