import { describe, it, expect } from 'vitest';
import { getDominoPoints, createDominoes } from '../../game';

describe('Scoring Systems - Counting Dominoes', () => {
  describe('When calculating point values', () => {
    it('should value 5-5 at 10 points', () => {
      const points = getDominoPoints({ high: 5, low: 5, id: '5-5' });
      expect(points).toBe(10);
    });

    it('should value 6-4 at 10 points', () => {
      const points = getDominoPoints({ high: 6, low: 4, id: '6-4' });
      expect(points).toBe(10);
    });

    it('should value 5-0 at 5 points', () => {
      const points = getDominoPoints({ high: 5, low: 0, id: '5-0' });
      expect(points).toBe(5);
    });

    it('should value 4-1 at 5 points', () => {
      const points = getDominoPoints({ high: 4, low: 1, id: '4-1' });
      expect(points).toBe(5);
    });

    it('should value 3-2 at 5 points', () => {
      const points = getDominoPoints({ high: 3, low: 2, id: '3-2' });
      expect(points).toBe(5);
    });

    it('should value all other dominoes at 0 points', () => {
      const nonCountingDominoes = createDominoes().filter(domino => {
        const points = getDominoPoints(domino);
        return points === 0;
      });
      
      // Should have 23 non-counting dominoes (28 total - 5 counting)
      expect(nonCountingDominoes).toHaveLength(23);
      
      nonCountingDominoes.forEach(domino => {
        const points = getDominoPoints(domino);
        expect(points).toBe(0);
      });
    });

    it('should have total count value of 35 points', () => {
      const allDominoes = createDominoes();
      let totalCount = 0;
      
      allDominoes.forEach(domino => {
        totalCount += getDominoPoints(domino);
      });
      
      expect(totalCount).toBe(35);
    });
  });
});