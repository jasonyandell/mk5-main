import { describe, it, expect } from 'vitest';
import { getDominoValue, getDominoSuit } from '../../game/core/dominoes';
import { calculateTrickWinner } from '../../game/core/scoring';
import type { PlayedDomino } from '../../game/types';

describe('Doubles Trump Rules', () => {
  describe('when a regular suit (0-6) is trump', () => {
    it('should only treat dominoes containing that number as trump', () => {
      // Test with 5s as trump
      const trump = 5;
      
      const testCases = [
        { domino: { high: 5, low: 5, id: '5-5' }, shouldBeTrump: true, reason: 'contains 5' },
        { domino: { high: 6, low: 5, id: '6-5' }, shouldBeTrump: true, reason: 'contains 5' },
        { domino: { high: 5, low: 4, id: '5-4' }, shouldBeTrump: true, reason: 'contains 5' },
        { domino: { high: 5, low: 3, id: '5-3' }, shouldBeTrump: true, reason: 'contains 5' },
        { domino: { high: 5, low: 2, id: '5-2' }, shouldBeTrump: true, reason: 'contains 5' },
        { domino: { high: 5, low: 1, id: '5-1' }, shouldBeTrump: true, reason: 'contains 5' },
        { domino: { high: 5, low: 0, id: '5-0' }, shouldBeTrump: true, reason: 'contains 5' },
        { domino: { high: 6, low: 6, id: '6-6' }, shouldBeTrump: false, reason: 'does not contain 5' },
        { domino: { high: 4, low: 4, id: '4-4' }, shouldBeTrump: false, reason: 'does not contain 5' },
        { domino: { high: 3, low: 3, id: '3-3' }, shouldBeTrump: false, reason: 'does not contain 5' },
        { domino: { high: 2, low: 2, id: '2-2' }, shouldBeTrump: false, reason: 'does not contain 5' },
        { domino: { high: 1, low: 1, id: '1-1' }, shouldBeTrump: false, reason: 'does not contain 5' },
        { domino: { high: 0, low: 0, id: '0-0' }, shouldBeTrump: false, reason: 'does not contain 5' },
        { domino: { high: 6, low: 4, id: '6-4' }, shouldBeTrump: false, reason: 'does not contain 5' }
      ];
      
      testCases.forEach(({ domino, shouldBeTrump }) => {
        const value = getDominoValue(domino, trump);
        const suit = getDominoSuit(domino, trump);
        
        if (shouldBeTrump) {
          // Trump dominoes should have values > 100
          expect(value).toBeGreaterThan(100);
          expect(suit).toBe(trump);
        } else {
          // Non-trump dominoes should have values < 100
          expect(value).toBeLessThan(100);
          expect(suit).not.toBe(trump);
        }
      });
    });
    
    it('should correctly determine trick winners with proper trump rules', () => {
      const trump = 3; // 3s are trump
      
      // Test case: 4-4 should NOT beat 3-2 when 3s are trump
      const trick: PlayedDomino[] = [
        { domino: { high: 3, low: 2, id: '3-2' }, player: 0 }, // Trump (contains 3)
        { domino: { high: 4, low: 4, id: '4-4' }, player: 1 }, // Not trump (no 3)
        { domino: { high: 6, low: 2, id: '6-2' }, player: 2 }, // Not trump
        { domino: { high: 5, low: 1, id: '5-1' }, player: 3 }  // Not trump
      ];
      
      const winner = calculateTrickWinner(trick, trump);
      
      // Player 0 should win because 3-2 is trump and 4-4 is not
      expect(winner).toBe(0);
    });
  });
  
  describe('when doubles are trump (trump = 7)', () => {
    it('should treat all doubles as trump', () => {
      const trump = 7; // Doubles are trump
      
      const doubles = [
        { high: 6, low: 6, id: '6-6' },
        { high: 5, low: 5, id: '5-5' },
        { high: 4, low: 4, id: '4-4' },
        { high: 3, low: 3, id: '3-3' },
        { high: 2, low: 2, id: '2-2' },
        { high: 1, low: 1, id: '1-1' },
        { high: 0, low: 0, id: '0-0' }
      ];
      
      doubles.forEach(domino => {
        const value = getDominoValue(domino, trump);
        const suit = getDominoSuit(domino, trump);
        
        // All doubles should be trump with values >= 200
        expect(value).toBeGreaterThanOrEqual(200);
        expect(suit).toBe(7);
      });
      
      // Non-doubles should not be trump
      const nonDouble = { high: 6, low: 5, id: '6-5' };
      const value = getDominoValue(nonDouble, trump);
      const suit = getDominoSuit(nonDouble, trump);
      
      expect(value).toBeLessThan(100);
      expect(suit).not.toBe(7);
    });
  });
});