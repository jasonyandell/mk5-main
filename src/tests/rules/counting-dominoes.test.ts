import { describe, it, expect } from 'vitest';

describe('Scoring Systems - Counting Dominoes', () => {
  describe('When calculating point values', () => {
    it('should value 5-5 at 10 points', () => {
      const domino = [5, 5];
      const points = getDominoPointValue(domino);
      expect(points).toBe(10);
    });

    it('should value 6-4 at 10 points', () => {
      const domino = [6, 4];
      const points = getDominoPointValue(domino);
      expect(points).toBe(10);
    });

    it('should value 5-0 at 5 points', () => {
      const domino = [5, 0];
      const points = getDominoPointValue(domino);
      expect(points).toBe(5);
    });

    it('should value 4-1 at 5 points', () => {
      const domino = [4, 1];
      const points = getDominoPointValue(domino);
      expect(points).toBe(5);
    });

    it('should value 3-2 at 5 points', () => {
      const domino = [3, 2];
      const points = getDominoPointValue(domino);
      expect(points).toBe(5);
    });

    it('should value all other dominoes at 0 points', () => {
      const nonCountingDominoes = [
        [6, 6], [6, 5], [6, 3], [6, 2], [6, 1], [6, 0],
        [5, 4], [5, 3], [5, 2], [5, 1],
        [4, 4], [4, 3], [4, 2], [4, 0],
        [3, 3], [3, 1], [3, 0],
        [2, 2], [2, 1], [2, 0],
        [1, 1], [1, 0],
        [0, 0]
      ];
      
      nonCountingDominoes.forEach(domino => {
        const points = getDominoPointValue(domino);
        expect(points).toBe(0);
      });
    });

    it('should have total count value of 35 points', () => {
      const allDominoes = getAllDominoes();
      let totalCount = 0;
      
      allDominoes.forEach(domino => {
        totalCount += getDominoPointValue(domino);
      });
      
      expect(totalCount).toBe(35);
    });
  });
});

// Test-only implementation
function getDominoPointValue(domino: number[]): number {
  const [end1, end2] = domino;
  const sortedDomino = [Math.min(end1, end2), Math.max(end1, end2)];
  
  // 10-point dominoes
  if ((sortedDomino[0] === 5 && sortedDomino[1] === 5) ||
      (sortedDomino[0] === 4 && sortedDomino[1] === 6)) {
    return 10;
  }
  
  // 5-point dominoes
  if ((sortedDomino[0] === 0 && sortedDomino[1] === 5) ||
      (sortedDomino[0] === 1 && sortedDomino[1] === 4) ||
      (sortedDomino[0] === 2 && sortedDomino[1] === 3)) {
    return 5;
  }
  
  return 0;
}

function getAllDominoes(): number[][] {
  const dominoes: number[][] = [];
  for (let i = 0; i <= 6; i++) {
    for (let j = i; j <= 6; j++) {
      dominoes.push([i, j]);
    }
  }
  return dominoes;
}