import { describe, it, expect } from 'vitest';
import { createTestState, createTestHand } from '../helpers/gameTestHelper';
import { isValidPlay, canFollowSuit, getValidPlays } from '../../game/core/rules';
import { getDominoSuit } from '../../game/core/dominoes';
import type { Domino, Trump } from '../../game/types';

describe('Renege Detection and Prevention', () => {
  describe('Must Follow Suit When Able', () => {
    it('requires following suit when player has matching suit', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 2, low: 4, points: 0 }, // 2-4 (twos suit - must play)
        { id: 'trump-1', high: 1, low: 3, points: 0 },  // 1-3 (trump)
        { id: 'other-1', high: 3, low: 5, points: 0 }   // 3-5 (threes suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 1, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }],
        players: [
          undefined, // player 0 (use default)
          { hand: playerHand }, // player 1 gets the test hand
          undefined, // player 2 (use default)
          undefined  // player 3 (use default)
        ]
      });

      // Can only play twos suit when able to follow
      const validPlays = getValidPlays(playerHand, state.currentTrick, state.trump!);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe('follow-1');
      
      // Validate each domino individually
      expect(isValidPlay(state, playerHand[0], 1)).toBe(true);  // Following suit
      expect(isValidPlay(state, playerHand[1], 1)).toBe(false); // Trump when can follow
      expect(isValidPlay(state, playerHand[2], 1)).toBe(false); // Other suit when can follow
    });

    it('allows any play when cannot follow suit', () => {
      const playerHand: Domino[] = [
        { id: 'trump-1', high: 1, low: 3, points: 0 },  // 1-3 (trump)
        { id: 'other-1', high: 3, low: 5, points: 0 },  // 3-5 (threes suit)
        { id: 'other-2', high: 4, low: 6, points: 0 }   // 4-6 (fours suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 1, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }],
        players: [
          undefined, // player 0 (use default)
          { hand: playerHand }, // player 1 gets the test hand
          undefined, // player 2 (use default)
          undefined  // player 3 (use default)
        ]
      });

      // All plays valid when can't follow suit
      const validPlays = getValidPlays(playerHand, state.currentTrick, state.trump!);
      expect(validPlays).toHaveLength(3);
      
      // All dominoes should be valid when can't follow suit
      playerHand.forEach(domino => {
        expect(isValidPlay(state, domino, 1)).toBe(true);
      });
    });

    it('correctly identifies trump vs non-trump suits', () => {
      const trump: Trump = 2; // Twos trump
      
      const trumpDomino: Domino = { id: 'trump', high: 2, low: 5, points: 0 }; // 2-5
      const nonTrumpDomino: Domino = { id: 'non-trump', high: 3, low: 4, points: 0 }; // 3-4

      expect(getDominoSuit(trumpDomino, trump)).toBe(trump);
      expect(getDominoSuit(nonTrumpDomino, trump)).not.toBe(trump);
    });
  });

  describe('Trump Play Rules', () => {
    it('allows trump when cannot follow suit', () => {
      const playerHand: Domino[] = [
        { id: 'trump-1', high: 1, low: 3, points: 0 },  // 1-3 (trump)
        { id: 'other-1', high: 3, low: 5, points: 0 }   // 3-5 (threes suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 1, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }],
        players: [
          undefined, // player 0 (use default)
          { hand: playerHand }, // player 1 gets the test hand
          undefined, // player 2 (use default)
          undefined  // player 3 (use default)
        ]
      });

      expect(canFollowSuit(playerHand, 2, state.trump!)).toBe(false);
      expect(isValidPlay(state, playerHand[0], 1)).toBe(true); // Trump play allowed
      expect(isValidPlay(state, playerHand[1], 1)).toBe(true); // Any play allowed
    });

    it('prevents trump when can follow suit', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 2, low: 4, points: 0 }, // 2-4 (twos suit - must play)
        { id: 'trump-1', high: 1, low: 3, points: 0 }   // 1-3 (trump - cannot play)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 1, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }],
        players: [
          undefined, // player 0 (use default)
          { hand: playerHand }, // player 1 gets the test hand
          undefined, // player 2 (use default)
          undefined  // player 3 (use default)
        ]
      });

      expect(canFollowSuit(playerHand, 2, state.trump!)).toBe(true);
      expect(isValidPlay(state, playerHand[0], 1)).toBe(true);  // Following suit
      expect(isValidPlay(state, playerHand[1], 1)).toBe(false); // Trump when can follow
    });
  });

  describe('Doubles Trump Special Cases', () => {
    it('handles doubles trump correctly', () => {
      const state = createTestState({
        phase: 'playing',
        trump: 7, // Doubles trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (threes suit)
        }]
      });

      const playerHand: Domino[] = [
        { id: 'follow-1', high: 3, low: 4, points: 0 }, // 3-4 (threes suit)
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (double/trump)
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (other suit)
      ];

      // Must follow threes suit, not doubles trump
      const validPlays = getValidPlays(playerHand, state.currentTrick, state.trump!);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe('follow-1');
    });

    it('allows doubles trump when cannot follow natural suit', () => {
      const playerHand: Domino[] = [
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (double/trump)
        { id: 'double-2', high: 2, low: 2, points: 0 }, // 2-2 (double/trump) 
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (other suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 7, // Doubles trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (threes suit)
        }],
        players: [
          undefined, // player 0 (use default)
          { hand: playerHand }, // player 1 gets the test hand
          undefined, // player 2 (use default)
          undefined  // player 3 (use default)
        ]
      });

      // Cannot follow threes suit, all plays valid
      const validPlays = getValidPlays(playerHand, state.currentTrick, state.trump!);
      expect(validPlays).toHaveLength(3);
      
      playerHand.forEach(domino => {
        expect(isValidPlay(state, domino, 1)).toBe(true);
      });
    });
  });

  describe('No Trump (Follow-Me) Special Cases', () => {
    it('enforces natural suit following in no-trump', () => {
      const state = createTestState({
        phase: 'playing',
        trump: 8, // No trump (follow-me)
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (higher pip = threes)
        }]
      });

      const playerHand: Domino[] = [
        { id: 'follow-1', high: 3, low: 4, points: 0 }, // 3-4 (threes suit)
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (ones suit in no-trump)
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (fours suit)
      ];

      const validPlays = getValidPlays(playerHand, state.currentTrick, state.trump!);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe('follow-1');
    });
  });

  describe('First Play of Trick', () => {
    it('allows any domino on opening lead', () => {
      const playerHand: Domino[] = [
        { id: 'any-1', high: 2, low: 3, points: 0 },
        { id: 'any-2', high: 1, low: 4, points: 0 }, // Trump
        { id: 'any-3', high: 5, low: 6, points: 0 }
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 1, // Ones trump
        currentTrick: [], // Empty trick
        players: [
          { hand: playerHand }, // player 0 gets the test hand
          undefined, // player 1 (use default)
          undefined, // player 2 (use default)
          undefined  // player 3 (use default)
        ]
      });

      const validPlays = getValidPlays(playerHand, state.currentTrick, state.trump!);
      expect(validPlays).toHaveLength(3);
      
      playerHand.forEach(domino => {
        expect(isValidPlay(state, domino, 0)).toBe(true);
      });
    });
  });

  describe('Edge Cases and Error Conditions', () => {
    it('handles invalid game phases', () => {
      const state = createTestState({
        phase: 'bidding' // Wrong phase
      });

      const domino: Domino = { id: 'test', high: 1, low: 2, points: 0 };
      expect(isValidPlay(state, domino, 0)).toBe(false);
    });

    it('handles missing trump declaration', () => {
      const state = createTestState({
        phase: 'playing',
        trump: null // No trump set
      });

      const domino: Domino = { id: 'test', high: 1, low: 2, points: 0 };
      expect(isValidPlay(state, domino, 0)).toBe(false);
    });

    it('prevents playing dominoes not in hand', () => {
      const state = createTestState({
        phase: 'playing',
        trump: 1,
        currentTrick: []
      });

      const notInHand: Domino = { id: 'not-owned', high: 1, low: 2, points: 0 };
      expect(isValidPlay(state, notInHand, 0)).toBe(false);
    });

    it('validates player bounds', () => {
      const state = createTestState({
        phase: 'playing',
        trump: 1,
        currentTrick: []
      });

      const domino: Domino = { id: 'test', high: 1, low: 2, points: 0 };
      expect(isValidPlay(state, domino, -1)).toBe(false); // Invalid player
      expect(isValidPlay(state, domino, 4)).toBe(false);  // Invalid player
    });
  });

  describe('Complex Renege Scenarios', () => {
    it('handles multiple trumps in trick with suit following', () => {
      const state = createTestState({
        phase: 'playing',
        trump: 1, // Ones trump
        currentTrick: [
          { player: 0, domino: { id: 'lead', high: 2, low: 2, points: 0 } }, // 2-2 (twos suit)
          { player: 1, domino: { id: 'trump1', high: 1, low: 3, points: 0 } }, // 1-3 (trump)
          { player: 2, domino: { id: 'trump2', high: 1, low: 4, points: 0 } }  // 1-4 (trump)
        ]
      });

      const playerHand: Domino[] = [
        { id: 'follow-1', high: 2, low: 5, points: 0 }, // 2-5 (twos suit - must play)
        { id: 'trump-3', high: 1, low: 6, points: 0 }   // 1-6 (trump)
      ];

      // Must still follow original suit despite trumps played
      const validPlays = getValidPlays(playerHand, state.currentTrick, state.trump!);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe('follow-1');
    });

    it('prevents renege in complex trick scenarios', () => {
      const state = createTestState({
        phase: 'playing',
        trump: 3, // Threes trump
        currentTrick: [{
          player: 0,
          domino: { id: 'lead', high: 4, low: 5, points: 0 } // 4-5 (fives suit)
        }]
      });

      const playerHand: Domino[] = [
        { id: 'follow-1', high: 5, low: 6, points: 0 }, // 5-6 (fives suit)
        { id: 'follow-2', high: 0, low: 5, points: 5 }, // 0-5 (fives suit, counting)
        { id: 'trump-1', high: 3, low: 6, points: 0 },  // 3-6 (trump)
        { id: 'other-1', high: 1, low: 2, points: 0 }   // 1-2 (ones suit)
      ];

      const validPlays = getValidPlays(playerHand, state.currentTrick, state.trump!);
      
      // Must play one of the fives suit dominoes
      expect(validPlays).toHaveLength(2);
      expect(validPlays.map(d => d.id)).toContain('follow-1');
      expect(validPlays.map(d => d.id)).toContain('follow-2');
      expect(validPlays.map(d => d.id)).not.toContain('trump-1');
      expect(validPlays.map(d => d.id)).not.toContain('other-1');
    });
  });
});