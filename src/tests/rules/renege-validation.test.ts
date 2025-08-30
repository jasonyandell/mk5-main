import { describe, it, expect } from 'vitest';
import { createTestState } from '../helpers/gameTestHelper';
import { isValidPlay, canFollowSuit, getValidPlays } from '../../game/core/rules';
import { getDominoSuit } from '../../game/core/dominoes';
import { analyzeSuits } from '../../game/core/suit-analysis';
import type { Domino, TrumpSelection } from '../../game/types';
import { ACES, DEUCES, TRES, FIVES, NO_LEAD_SUIT } from '../../game/types';

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
        trump: { type: 'suit', suit: ACES }, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }],
        currentSuit: DEUCES, // Twos were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: ACES }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // Can only play twos suit when able to follow
      const validPlays = getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe('follow-1');
      
      // Validate each domino individually
      const domino0 = playerHand[0];
      const domino1 = playerHand[1];
      const domino2 = playerHand[2];
      if (!domino0 || !domino1 || !domino2) {
        throw new Error('Missing dominoes in player hand');
      }
      expect(isValidPlay(state, domino0, 1)).toBe(true);  // Following suit
      expect(isValidPlay(state, domino1, 1)).toBe(false); // Trump when can follow
      expect(isValidPlay(state, domino2, 1)).toBe(false); // Other suit when can follow
    });

    it('allows any play when cannot follow suit', () => {
      const playerHand: Domino[] = [
        { id: 'trump-1', high: 1, low: 3, points: 0 },  // 1-3 (trump)
        { id: 'other-1', high: 3, low: 5, points: 0 },  // 3-5 (threes suit)
        { id: 'other-2', high: 4, low: 6, points: 0 }   // 4-6 (fours suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: ACES }, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }],
        currentSuit: DEUCES, // Twos were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: ACES }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // All plays valid when can't follow suit
      const validPlays = getValidPlays(state, 1);
      expect(validPlays).toHaveLength(3);
      
      // All dominoes should be valid when can't follow suit
      playerHand.forEach(domino => {
        expect(isValidPlay(state, domino, 1)).toBe(true);
      });
    });

    it('correctly identifies trump vs non-trump suits', () => {
      const trump: TrumpSelection = { type: 'suit', suit: DEUCES }; // Twos trump
      
      const trumpDomino: Domino = { id: 'trump', high: 2, low: 5, points: 0 }; // 2-5
      const nonTrumpDomino: Domino = { id: 'non-trump', high: 3, low: 4, points: 0 }; // 3-4

      expect(getDominoSuit(trumpDomino, trump)).toBe(DEUCES); // Should be suit 2 (trump)
      expect(getDominoSuit(nonTrumpDomino, trump)).not.toBe(DEUCES); // Should not be suit 2
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
        trump: { type: 'suit', suit: ACES }, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }],
        currentSuit: DEUCES, // Twos were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: ACES }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      expect(canFollowSuit(state.players[1]!, DEUCES)).toBe(false);
      expect(isValidPlay(state, playerHand[0]!, 1)).toBe(true); // Trump play allowed
      expect(isValidPlay(state, playerHand[1]!, 1)).toBe(true); // Any play allowed
    });

    it('prevents trump when can follow suit', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 2, low: 4, points: 0 }, // 2-4 (twos suit - must play)
        { id: 'trump-1', high: 1, low: 3, points: 0 }   // 1-3 (trump - cannot play)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: ACES }, // Ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 2, points: 0 } // 2-2 leads (twos suit)
        }],
        currentSuit: DEUCES, // Twos were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: ACES }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      expect(canFollowSuit(state.players[1]!, DEUCES)).toBe(true);
      expect(isValidPlay(state, playerHand[0]!, 1)).toBe(true);  // Following suit
      expect(isValidPlay(state, playerHand[1]!, 1)).toBe(false); // Trump when can follow
    });
  });

  describe('Doubles Trump Special Cases', () => {
    it('handles doubles trump correctly', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 3, low: 4, points: 0 }, // 3-4 (threes suit)
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (double/trump)
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (other suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'doubles' }, // Doubles trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (threes suit)
        }],
        currentSuit: TRES, // Threes were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'doubles' }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // Must follow threes suit, not doubles trump
      const validPlays = getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe('follow-1');
    });

    it('allows doubles trump when cannot follow natural suit', () => {
      const playerHand: Domino[] = [
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (double/trump)
        { id: 'double-2', high: 2, low: 2, points: 0 }, // 2-2 (double/trump) 
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (other suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'doubles' }, // Doubles trump
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (threes suit)
        }],
        currentSuit: TRES, // Threes were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'doubles' }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      // Cannot follow threes suit, all plays valid
      const validPlays = getValidPlays(state, 1);
      expect(validPlays).toHaveLength(3);
      
      playerHand.forEach(domino => {
        expect(isValidPlay(state, domino, 1)).toBe(true);
      });
    });
  });

  describe('No Trump (Follow-Me) Special Cases', () => {
    it('enforces natural suit following in no-trump', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 3, low: 4, points: 0 }, // 3-4 (threes suit)
        { id: 'double-1', high: 1, low: 1, points: 0 }, // 1-1 (ones suit in no-trump)
        { id: 'other-1', high: 4, low: 5, points: 0 }   // 4-5 (fours suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'no-trump' }, // No trump (follow-me)
        currentTrick: [{
          player: 0,
          domino: { id: 'test-lead', high: 2, low: 3, points: 0 } // 2-3 leads (higher pip = threes)
        }],
        currentSuit: TRES, // Threes were led
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'no-trump' }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      const validPlays = getValidPlays(state, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe('follow-1');
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
        trump: { type: 'suit', suit: ACES }, // Ones trump
        currentTrick: [], // Empty trick
        currentSuit: NO_LEAD_SUIT,
        currentPlayer: 0,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: ACES }) },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      const validPlays = getValidPlays(state, 0);
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
        trump: { type: 'not-selected' } // No trump set
      });

      const domino: Domino = { id: 'test', high: 1, low: 2, points: 0 };
      expect(isValidPlay(state, domino, 0)).toBe(false);
    });

    it('prevents playing dominoes not in hand', () => {
      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: ACES },
        currentTrick: []
      });

      const notInHand: Domino = { id: 'not-owned', high: 1, low: 2, points: 0 };
      expect(isValidPlay(state, notInHand, 0)).toBe(false);
    });

    it('validates player bounds', () => {
      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: ACES },
        currentTrick: []
      });

      const domino: Domino = { id: 'test', high: 1, low: 2, points: 0 };
      expect(isValidPlay(state, domino, -1)).toBe(false); // Invalid player
      expect(isValidPlay(state, domino, 4)).toBe(false);  // Invalid player
    });
  });

  describe('Complex Renege Scenarios', () => {
    it('handles multiple trumps in trick with suit following', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 2, low: 5, points: 0 }, // 2-5 (twos suit - must play)
        { id: 'trump-3', high: 1, low: 6, points: 0 }   // 1-6 (trump)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: ACES }, // Ones trump
        currentTrick: [
          { player: 0, domino: { id: 'lead', high: 2, low: 2, points: 0 } }, // 2-2 (twos suit)
          { player: 1, domino: { id: 'trump1', high: 1, low: 3, points: 0 } }, // 1-3 (trump)
          { player: 2, domino: { id: 'trump2', high: 1, low: 4, points: 0 } }  // 1-4 (trump)
        ],
        currentSuit: DEUCES, // Twos were led
        currentPlayer: 3,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: [] },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: ACES }) }
        ]
      });

      // Must still follow original suit despite trumps played
      const validPlays = getValidPlays(state, 3);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0]?.id).toBe('follow-1');
    });

    it('prevents renege in complex trick scenarios', () => {
      const playerHand: Domino[] = [
        { id: 'follow-1', high: 5, low: 6, points: 0 }, // 5-6 (fives suit)
        { id: 'follow-2', high: 0, low: 5, points: 5 }, // 0-5 (fives suit, counting)
        { id: 'trump-1', high: 3, low: 6, points: 0 },  // 3-6 (trump)
        { id: 'other-1', high: 1, low: 2, points: 0 }   // 1-2 (ones suit)
      ];

      const state = createTestState({
        phase: 'playing',
        trump: { type: 'suit', suit: TRES }, // Threes trump
        currentTrick: [{
          player: 0,
          domino: { id: 'lead', high: 4, low: 5, points: 0 } // 4-5 (fives suit)
        }],
        currentSuit: FIVES,
        currentPlayer: 1,
        players: [
          { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
          { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: playerHand, suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: TRES }) },
          { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
          { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
        ]
      });

      const validPlays = getValidPlays(state, 1);
      
      // Must play one of the fives suit dominoes
      expect(validPlays).toHaveLength(2);
      expect(validPlays.map(d => d.id)).toContain('follow-1');
      expect(validPlays.map(d => d.id)).toContain('follow-2');
      expect(validPlays.map(d => d.id)).not.toContain('trump-1');
      expect(validPlays.map(d => d.id)).not.toContain('other-1');
    });
  });
});