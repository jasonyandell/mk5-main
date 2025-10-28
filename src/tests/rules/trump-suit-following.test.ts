import { describe, it, expect } from 'vitest';
import { createTestState } from '../helpers/gameTestHelper';
import { composeRules } from '../../game/layers/compose';
import { baseLayer } from '../../game/layers';
import { analyzeSuits } from '../../game/core/suit-analysis';
import type { Domino } from '../../game/types';
import { DEUCES, TRES, FOURS, FIVES, SIXES } from '../../game/types';

const rules = composeRules([baseLayer]);

describe('Trump Suit Following Rules', () => {
  describe('Cannot Play Trump When Can Follow Suit', () => {
    it('prevents playing 6-5 (trump) when player has 6-2 (non-trump) and 6s are led', () => {
      // This is the exact bug scenario from the user's example
      const playerHand: Domino[] = [
        { id: '6-2', high: 6, low: 2, points: 0 }, // Can follow suit with this
        { id: '6-5', high: 6, low: 5, points: 0 }, // This is trump (5s are trump)
        { id: '3-1', high: 3, low: 1, points: 0 }  // Random other domino
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: FIVES }, // Fives are trump
        currentTrick: [{
          player: 0,
          domino: { id: '6-6', high: 6, low: 6, points: 0 } // 6-6 leads (sixes suit)
        }],
        currentSuit: SIXES, // Sixes were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: FIVES }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // Check that 6-5 is NOT valid because player can follow suit with 6-2
      expect(rules.isValidPlay(state, { id: '6-5', high: 6, low: 5, points: 0 }, 1)).toBe(false);
      expect(rules.isValidPlay(state, { id: '6-2', high: 6, low: 2, points: 0 }, 1)).toBe(true);
      
      // Valid plays should only include 6-2, not 6-5
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      const firstValidPlay = validPlays[0];
      if (!firstValidPlay) throw new Error('First valid play is undefined');
      expect(firstValidPlay.id).toBe('6-2');
    });

    it('allows playing trump domino when it is the only way to follow suit', () => {
      // Player only has trump dominoes that can follow suit
      const playerHand: Domino[] = [
        { id: '6-5', high: 6, low: 5, points: 0 }, // This is trump (5s are trump) but also follows 6s
        { id: '5-4', high: 5, low: 4, points: 0 }, // Pure trump
        { id: '3-1', high: 3, low: 1, points: 0 }  // Cannot follow suit
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: FIVES }, // Fives are trump
        currentTrick: [{
          player: 0,
          domino: { id: '6-6', high: 6, low: 6, points: 0 } // 6-6 leads (sixes suit)
        }],
        currentSuit: SIXES, // Sixes were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: FIVES }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // Since player has no non-trump 6s, they can play any domino
      const validPlays = rules.getValidPlays(state, 1);
      expect(validPlays).toHaveLength(3); // All dominoes are valid
      
      // All plays should be valid since player cannot follow suit without trump
      expect(rules.isValidPlay(state, { id: '6-5', high: 6, low: 5, points: 0 }, 1)).toBe(true);
      expect(rules.isValidPlay(state, { id: '5-4', high: 5, low: 4, points: 0 }, 1)).toBe(true);
      expect(rules.isValidPlay(state, { id: '3-1', high: 3, low: 1, points: 0 }, 1)).toBe(true);
    });

    it('prevents playing trump when multiple non-trump options exist to follow suit', () => {
      const playerHand: Domino[] = [
        { id: '4-2', high: 4, low: 2, points: 0 }, // Can follow 4s
        { id: '4-1', high: 4, low: 1, points: 5 }, // Can follow 4s  
        { id: '4-3', high: 4, low: 3, points: 0 }, // This is trump (3s are trump) but also 4s
        { id: '3-2', high: 3, low: 2, points: 5 }  // Pure trump
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: TRES }, // Threes are trump
        currentTrick: [{
          player: 0,
          domino: { id: '4-4', high: 4, low: 4, points: 0 } // 4-4 leads (fours suit)
        }],
        currentSuit: FOURS, // Fours were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: TRES }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      const validPlays = rules.getValidPlays(state, 1);
      
      // Should only be able to play the non-trump 4s
      expect(validPlays).toHaveLength(2);
      expect(validPlays.map(d => d.id)).toContain('4-2');
      expect(validPlays.map(d => d.id)).toContain('4-1');
      expect(validPlays.map(d => d.id)).not.toContain('4-3'); // Trump, cannot play
      expect(validPlays.map(d => d.id)).not.toContain('3-2'); // Trump, cannot play
    });
  });

  describe('Suit Analysis with Trump', () => {
    it('correctly includes trump dominoes in natural suits for reference', () => {
      const hand: Domino[] = [
        { id: '2-1', high: 2, low: 1, points: 0 }, // Has 2, not trump
        { id: '2-3', high: 2, low: 3, points: 5 }, // This is trump (3s are trump) but also in suit 2
        { id: '3-3', high: 3, low: 3, points: 0 }, // Double 3 (trump)
        { id: '4-5', high: 4, low: 5, points: 0 }  // Neither 2 nor trump
      ];

      const analysis = analyzeSuits(hand, { type: 'suit', suit: TRES });
      
      // Check suit 2: should have both dominoes containing 2 (including trump)
      expect(analysis.rank[DEUCES]).toHaveLength(2);
      expect(analysis.rank[DEUCES].map(d => d.id)).toContain('2-1');
      expect(analysis.rank[DEUCES].map(d => d.id)).toContain('2-3');
      
      // Check trump: should have both dominoes containing 3
      expect(analysis.rank.trump).toHaveLength(2);
      expect(analysis.rank.trump.map(d => d.id)).toContain('3-3');
      expect(analysis.rank.trump.map(d => d.id)).toContain('2-3');
      
      // Note: 2-3 is in BOTH suit 2 and trump (filtering happens at validation time)
    });

    it('handles doubles as trump correctly', () => {
      const hand: Domino[] = [
        { id: '3-3', high: 3, low: 3, points: 0 }, // Double (trump when doubles are trump)
        { id: '3-4', high: 3, low: 4, points: 0 }, // Has 3, not trump when doubles are trump
        { id: '2-2', high: 2, low: 2, points: 0 }, // Another double (trump)
        { id: '3-5', high: 3, low: 5, points: 0 }  // Has 3, not trump
      ];

      const analysis = analyzeSuits(hand, { type: 'doubles' });
      
      // When doubles are trump, non-double 3s should be in suit 3
      expect(analysis.rank[TRES]).toHaveLength(2);
      expect(analysis.rank[TRES].map(d => d.id)).toContain('3-4');
      expect(analysis.rank[TRES].map(d => d.id)).toContain('3-5');
      
      // Doubles should only be in trump
      expect(analysis.rank.trump).toHaveLength(2);
      expect(analysis.rank.trump.map(d => d.id)).toContain('3-3');
      expect(analysis.rank.trump.map(d => d.id)).toContain('2-2');
      
      // Doubles should NOT be in their natural suits when doubles are trump
      expect(analysis.rank[TRES].map(d => d.id)).not.toContain('3-3');
      expect(analysis.rank[DEUCES].map(d => d.id)).not.toContain('2-2');
    });
  });
});