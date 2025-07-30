import { describe, it, expect } from 'vitest';
import { 
  determineDealerFromDraw,
  createDominoForDraw 
} from '../../game/core/setup';
import type { DealerDrawResult } from '../../game/core/setup';

describe('Determining First Dealer', () => {
  describe('Given all four players are ready to start', () => {
    describe('When each player draws one domino face-down', () => {
      it('Then the player with the highest total pip count becomes the first shaker', () => {
        const draws: DealerDrawResult[] = [
          { playerId: 0, domino: createDominoForDraw(3, 2) }, // Total: 5
          { playerId: 1, domino: createDominoForDraw(6, 4) }, // Total: 10  
          { playerId: 2, domino: createDominoForDraw(2, 1) }, // Total: 3
          { playerId: 3, domino: createDominoForDraw(4, 3) }, // Total: 7
        ];

        const result = determineDealerFromDraw(draws);
        
        expect(result.dealer).toBe(1);
        expect(result.requiresRedraw).toBe(false);
        expect(result.tiedPlayers).toBeUndefined();
      });

      it('And if there is a tie, the affected players must redraw', () => {
        const draws: DealerDrawResult[] = [
          { playerId: 0, domino: createDominoForDraw(4, 3) }, // Total: 7
          { playerId: 1, domino: createDominoForDraw(5, 2) }, // Total: 7 - TIE!
          { playerId: 2, domino: createDominoForDraw(2, 1) }, // Total: 3
          { playerId: 3, domino: createDominoForDraw(3, 2) }, // Total: 5
        ];

        const result = determineDealerFromDraw(draws);
        
        expect(result.dealer).toBeNull();
        expect(result.requiresRedraw).toBe(true);
        expect(result.tiedPlayers).toEqual([0, 1]);
      });

      it('handles three-way ties correctly', () => {
        const draws: DealerDrawResult[] = [
          { playerId: 0, domino: createDominoForDraw(5, 3) }, // Total: 8
          { playerId: 1, domino: createDominoForDraw(6, 2) }, // Total: 8 - TIE!
          { playerId: 2, domino: createDominoForDraw(4, 4) }, // Total: 8 - TIE!
          { playerId: 3, domino: createDominoForDraw(3, 2) }, // Total: 5
        ];

        const result = determineDealerFromDraw(draws);
        
        expect(result.dealer).toBeNull();
        expect(result.requiresRedraw).toBe(true);
        expect(result.tiedPlayers).toEqual([0, 1, 2]);
      });

      it('handles four-way ties correctly', () => {
        const draws: DealerDrawResult[] = [
          { playerId: 0, domino: createDominoForDraw(3, 3) }, // Total: 6
          { playerId: 1, domino: createDominoForDraw(4, 2) }, // Total: 6
          { playerId: 2, domino: createDominoForDraw(5, 1) }, // Total: 6
          { playerId: 3, domino: createDominoForDraw(6, 0) }, // Total: 6
        ];

        const result = determineDealerFromDraw(draws);
        
        expect(result.dealer).toBeNull();
        expect(result.requiresRedraw).toBe(true);
        expect(result.tiedPlayers).toEqual([0, 1, 2, 3]);
      });

      it('correctly identifies the dealer with double six', () => {
        const draws: DealerDrawResult[] = [
          { playerId: 0, domino: createDominoForDraw(3, 3) }, // Total: 6
          { playerId: 1, domino: createDominoForDraw(6, 6) }, // Total: 12 - WINNER
          { playerId: 2, domino: createDominoForDraw(5, 5) }, // Total: 10
          { playerId: 3, domino: createDominoForDraw(4, 4) }, // Total: 8
        ];

        const result = determineDealerFromDraw(draws);
        
        expect(result.dealer).toBe(1);
        expect(result.requiresRedraw).toBe(false);
      });

      it('correctly handles blanks in the draw', () => {
        const draws: DealerDrawResult[] = [
          { playerId: 0, domino: createDominoForDraw(0, 0) }, // Total: 0
          { playerId: 1, domino: createDominoForDraw(1, 0) }, // Total: 1
          { playerId: 2, domino: createDominoForDraw(2, 0) }, // Total: 2
          { playerId: 3, domino: createDominoForDraw(3, 0) }, // Total: 3 - WINNER
        ];

        const result = determineDealerFromDraw(draws);
        
        expect(result.dealer).toBe(3);
        expect(result.requiresRedraw).toBe(false);
      });
    });
  });

  describe('Edge cases', () => {
    it('throws error if not exactly 4 players draw', () => {
      const tooFewDraws: DealerDrawResult[] = [
        { playerId: 0, domino: createDominoForDraw(3, 2) },
        { playerId: 1, domino: createDominoForDraw(6, 4) },
        { playerId: 2, domino: createDominoForDraw(2, 1) },
      ];

      expect(() => determineDealerFromDraw(tooFewDraws)).toThrow('Exactly 4 players must draw');

      const tooManyDraws: DealerDrawResult[] = [
        { playerId: 0, domino: createDominoForDraw(3, 2) },
        { playerId: 1, domino: createDominoForDraw(6, 4) },
        { playerId: 2, domino: createDominoForDraw(2, 1) },
        { playerId: 3, domino: createDominoForDraw(4, 3) },
        { playerId: 4, domino: createDominoForDraw(5, 2) },
      ];

      expect(() => determineDealerFromDraw(tooManyDraws)).toThrow('Exactly 4 players must draw');
    });

    it('throws error if player IDs are not 0-3', () => {
      const invalidPlayerIds: DealerDrawResult[] = [
        { playerId: 1, domino: createDominoForDraw(3, 2) },
        { playerId: 2, domino: createDominoForDraw(6, 4) },
        { playerId: 3, domino: createDominoForDraw(2, 1) },
        { playerId: 4, domino: createDominoForDraw(4, 3) }, // Invalid ID
      ];

      expect(() => determineDealerFromDraw(invalidPlayerIds)).toThrow('Invalid player IDs');
    });

    it('throws error if domino values are invalid', () => {
      // Test that createDominoForDraw throws on invalid values
      expect(() => createDominoForDraw(7, 2)).toThrow('Invalid domino values');
      expect(() => createDominoForDraw(-1, 2)).toThrow('Invalid domino values');
      expect(() => createDominoForDraw(2, 7)).toThrow('Invalid domino values');
      expect(() => createDominoForDraw(2, -1)).toThrow('Invalid domino values');

      // Test that determineDealerFromDraw validates domino values
      const invalidDomino: DealerDrawResult[] = [
        { playerId: 0, domino: { high: 7, low: 2, id: '7-2' } }, // 7 is invalid
        { playerId: 1, domino: createDominoForDraw(6, 4) },
        { playerId: 2, domino: createDominoForDraw(2, 1) },
        { playerId: 3, domino: createDominoForDraw(4, 3) },
      ];

      expect(() => determineDealerFromDraw(invalidDomino)).toThrow('Invalid domino values');
    });
  });
});