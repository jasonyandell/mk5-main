import { describe, it, expect } from 'vitest';
import { 
  createDominoForDraw, 
  determineDealerFromDraw,
  type DealerDrawResult
} from '../../game/core/setup';

describe('Feature: Game Setup - Determining First Dealer', () => {
  describe('Scenario: Determining First Dealer', () => {
    it('Given all four players are ready to start', () => {
      // Setup is implicit - we have 4 players (0, 1, 2, 3)
    });

    it('When each player draws one domino face-down', () => {
      // The draw action is represented by creating DealerDrawResult objects
      const draws: DealerDrawResult[] = [
        { playerId: 0, domino: createDominoForDraw(3, 2) }, // Total: 5
        { playerId: 1, domino: createDominoForDraw(6, 4) }, // Total: 10
        { playerId: 2, domino: createDominoForDraw(1, 1) }, // Total: 2
        { playerId: 3, domino: createDominoForDraw(5, 3) }  // Total: 8
      ];

      // Then the player with the highest total pip count becomes the first shaker
      const result = determineDealerFromDraw(draws);
      
      expect(result.dealer).toBe(1); // Player 1 has highest total (10)
      expect(result.requiresRedraw).toBe(false);
      expect(result.tiedPlayers).toBeUndefined();
    });

    it('And if there is a tie, the affected players must redraw', () => {
      // Create a tie scenario
      const draws: DealerDrawResult[] = [
        { playerId: 0, domino: createDominoForDraw(5, 5) }, // Total: 10
        { playerId: 1, domino: createDominoForDraw(6, 4) }, // Total: 10
        { playerId: 2, domino: createDominoForDraw(1, 1) }, // Total: 2
        { playerId: 3, domino: createDominoForDraw(3, 2) }  // Total: 5
      ];

      const result = determineDealerFromDraw(draws);
      
      expect(result.dealer).toBeNull();
      expect(result.requiresRedraw).toBe(true);
      expect(result.tiedPlayers).toEqual([0, 1]); // Players 0 and 1 are tied
    });

    // Additional test cases for completeness
    it('handles multiple players tied for highest', () => {
      const draws: DealerDrawResult[] = [
        { playerId: 0, domino: createDominoForDraw(4, 4) }, // Total: 8
        { playerId: 1, domino: createDominoForDraw(5, 3) }, // Total: 8
        { playerId: 2, domino: createDominoForDraw(6, 2) }, // Total: 8
        { playerId: 3, domino: createDominoForDraw(1, 1) }  // Total: 2
      ];

      const result = determineDealerFromDraw(draws);
      
      expect(result.dealer).toBeNull();
      expect(result.requiresRedraw).toBe(true);
      expect(result.tiedPlayers).toEqual([0, 1, 2]);
    });

    it('validates exactly 4 players must draw', () => {
      const invalidDraws: DealerDrawResult[] = [
        { playerId: 0, domino: createDominoForDraw(3, 2) },
        { playerId: 1, domino: createDominoForDraw(6, 4) },
        { playerId: 2, domino: createDominoForDraw(1, 1) }
        // Missing player 3
      ];

      expect(() => determineDealerFromDraw(invalidDraws)).toThrow('Exactly 4 players must draw for dealer determination');
    });

    it('validates player IDs are 0-3', () => {
      const invalidDraws: DealerDrawResult[] = [
        { playerId: 0, domino: createDominoForDraw(3, 2) },
        { playerId: 1, domino: createDominoForDraw(6, 4) },
        { playerId: 2, domino: createDominoForDraw(1, 1) },
        { playerId: 5, domino: createDominoForDraw(5, 3) } // Invalid player ID
      ];

      expect(() => determineDealerFromDraw(invalidDraws)).toThrow('Invalid player IDs: must be 0, 1, 2, 3');
    });

    it('validates domino pip values are 0-6', () => {
      const invalidDraws: DealerDrawResult[] = [
        { playerId: 0, domino: createDominoForDraw(3, 2) },
        { playerId: 1, domino: createDominoForDraw(7, 4) }, // Invalid high value
        { playerId: 2, domino: createDominoForDraw(1, 1) },
        { playerId: 3, domino: createDominoForDraw(5, 3) }
      ];

      expect(() => createDominoForDraw(7, 4)).toThrow('Invalid domino values: must be between 0 and 6');
    });
  });
});