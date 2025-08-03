import { describe, it, expect } from 'vitest';
import { analyzeSuits, calculateSuitCount, calculateSuitRanking, getStrongestSuits } from '../../game/core/suit-analysis';
import type { Domino } from '../../game/types';

describe('Suit Analysis', () => {
  describe('calculateSuitCount', () => {
    it('should count suits correctly for basic hand', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' }, // double fives
        { high: 6, low: 5, id: '6-5' }, // contains 5 and 6
        { high: 3, low: 1, id: '3-1' }, // contains 1 and 3
      ];

      const count = calculateSuitCount(hand);
      
      expect(count[0]).toBe(0); // no blanks
      expect(count[1]).toBe(1); // one domino with 1 (3-1)
      expect(count[2]).toBe(0); // no twos
      expect(count[3]).toBe(1); // one domino with 3 (3-1)
      expect(count[4]).toBe(0); // no fours
      expect(count[5]).toBe(2); // two dominoes with 5 (5-5, 6-5)
      expect(count[6]).toBe(1); // one domino with 6 (6-5)
      expect(count.doubles).toBe(1); // one double (5-5)
      expect(count.trump).toBe(0); // no trump declared
    });

    it('should count trump correctly when trump is declared', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' }, // double fives (trump if 5s are trump)
        { high: 6, low: 5, id: '6-5' }, // contains trump 5
        { high: 3, low: 1, id: '3-1' }, // no trump
      ];

      const count = calculateSuitCount(hand, { type: 'suit', suit: 5 }); // 5s are trump
      
      expect(count[5]).toBe(2); // natural suit count stays same
      expect(count.trump).toBe(2); // two trump dominoes (5-5, 6-5)
    });

    it('should handle doubles trump correctly', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' }, // double (trump when doubles are trump)
        { high: 6, low: 5, id: '6-5' }, // not trump when doubles are trump
        { high: 3, low: 3, id: '3-3' }, // double (trump when doubles are trump)
      ];

      const count = calculateSuitCount(hand, { type: 'doubles' }); // doubles are trump
      
      expect(count.doubles).toBe(2); // two doubles
      expect(count.trump).toBe(2); // two trump dominoes (both doubles)
    });
  });

  describe('calculateSuitRanking', () => {
    it('should rank dominoes by suit correctly', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' }, // double fives
        { high: 6, low: 5, id: '6-5' }, // 6-5 (high pip count)
        { high: 3, low: 1, id: '3-1' }, // 3-1 (lower pip count)
        { high: 4, low: 1, id: '4-1' }, // 4-1 (higher pip count than 3-1)
      ];

      const ranking = calculateSuitRanking(hand);
      
      // Check that 5s are ranked correctly (double first, then by pip count)
      expect(ranking[5]).toHaveLength(2);
      expect(ranking[5][0]).toEqual({ high: 5, low: 5, id: '5-5' }); // double first
      expect(ranking[5][1]).toEqual({ high: 6, low: 5, id: '6-5' }); // then non-double
      
      // Check that 1s are ranked by pip count
      expect(ranking[1]).toHaveLength(2);
      expect(ranking[1][0]).toEqual({ high: 4, low: 1, id: '4-1' }); // higher pip count first
      expect(ranking[1][1]).toEqual({ high: 3, low: 1, id: '3-1' }); // lower pip count second
      
      // Check doubles ranking
      expect(ranking.doubles).toHaveLength(1);
      expect(ranking.doubles[0]).toEqual({ high: 5, low: 5, id: '5-5' });
    });

    it('should rank trump dominoes correctly when trump is declared', () => {
      const hand: Domino[] = [
        { high: 3, low: 3, id: '3-3' }, // double threes (highest trump if 3s are trump)
        { high: 6, low: 3, id: '6-3' }, // contains trump 3
        { high: 3, low: 1, id: '3-1' }, // contains trump 3
        { high: 4, low: 2, id: '4-2' }, // no trump
      ];

      const ranking = calculateSuitRanking(hand, { type: 'suit', suit: 3 }); // 3s are trump
      
      // Trump ranking should have double first, then by pip count
      expect(ranking.trump).toHaveLength(3);
      expect(ranking.trump[0]).toEqual({ high: 3, low: 3, id: '3-3' }); // double first
      expect(ranking.trump[1]).toEqual({ high: 6, low: 3, id: '6-3' }); // higher pip count
      expect(ranking.trump[2]).toEqual({ high: 3, low: 1, id: '3-1' }); // lower pip count
    });
  });

  describe('analyzeSuits', () => {
    it('should return complete analysis with count and rank', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' },
        { high: 6, low: 5, id: '6-5' },
        { high: 3, low: 1, id: '3-1' },
      ];

      const analysis = analyzeSuits(hand);
      
      expect(analysis.count).toBeDefined();
      expect(analysis.rank).toBeDefined();
      expect(analysis.count[5]).toBe(2);
      expect(analysis.rank[5]).toHaveLength(2);
    });

    it('should update trump analysis when trump is declared', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' },
        { high: 6, low: 5, id: '6-5' },
        { high: 3, low: 1, id: '3-1' },
      ];

      const analysis = analyzeSuits(hand, { type: 'suit', suit: 5 });
      
      expect(analysis.count.trump).toBe(2);
      expect(analysis.rank.trump).toHaveLength(2);
    });
  });

  describe('getStrongestSuits', () => {
    it('should rank suits by count first, then by highest domino', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' }, // suit 5: count 2
        { high: 6, low: 5, id: '6-5' }, // suit 5: count 2, suit 6: count 1
        { high: 3, low: 1, id: '3-1' }, // suit 3: count 1, suit 1: count 1
        { high: 4, low: 1, id: '4-1' }, // suit 4: count 1, suit 1: count 2
      ];

      const analysis = analyzeSuits(hand);
      const strongest = getStrongestSuits(analysis);
      
      // Suit 5 should be strongest (count 2)
      expect(strongest[0]).toBe(5);
      // Suit 1 should be second (count 2, but lower highest domino than 5)
      expect(strongest[1]).toBe(1);
      // Then suits with count 1, ranked by their highest domino
      // Suit 6: has 6-5 (total 11)
      // Suit 4: has 4-1 (total 5) 
      // Suit 3: has 3-1 (total 4)
      expect(strongest[2]).toBe(6); // highest total among count-1 suits
      expect(strongest[3]).toBe(4);
      expect(strongest[4]).toBe(3);
      // Suits 0 and 2 have count 0 and should be last (order doesn't matter)
      expect(strongest.slice(5)).toContain(0);
      expect(strongest.slice(5)).toContain(2);
    });
  });

  describe('Real game scenarios', () => {
    it('should handle the example from user: 5-5, 6-5, 3-1', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' },
        { high: 6, low: 5, id: '6-5' },
        { high: 3, low: 1, id: '3-1' },
      ];

      const analysis = analyzeSuits(hand);
      
      // Expected count: {1:1, 3:1, 5:2, 6:1}
      expect(analysis.count[1]).toBe(1);
      expect(analysis.count[3]).toBe(1);
      expect(analysis.count[5]).toBe(2);
      expect(analysis.count[6]).toBe(1);
      
      // Expected rank: {5:[5-5, 6-5], 6:[6-5], 3:[3-1], 1:[3-1]}
      expect(analysis.rank[5]).toEqual([
        { high: 5, low: 5, id: '5-5' },
        { high: 6, low: 5, id: '6-5' }
      ]);
      expect(analysis.rank[6]).toEqual([{ high: 6, low: 5, id: '6-5' }]);
      expect(analysis.rank[3]).toEqual([{ high: 3, low: 1, id: '3-1' }]);
      expect(analysis.rank[1]).toEqual([{ high: 3, low: 1, id: '3-1' }]);
    });

    it('should properly track trump when 3s are trump', () => {
      const hand: Domino[] = [
        { high: 5, low: 5, id: '5-5' },
        { high: 6, low: 5, id: '6-5' },
        { high: 3, low: 1, id: '3-1' }, // contains trump
      ];

      const analysis = analyzeSuits(hand, { type: 'suit', suit: 3 }); // 3s are trump
      
      // Trump count and rank should be identical to suit 3
      expect(analysis.count.trump).toBe(analysis.count[3]);
      expect(analysis.rank.trump).toEqual(analysis.rank[3]);
      expect(analysis.count.trump).toBe(1);
      expect(analysis.rank.trump).toEqual([{ high: 3, low: 1, id: '3-1' }]);
    });
  });
});