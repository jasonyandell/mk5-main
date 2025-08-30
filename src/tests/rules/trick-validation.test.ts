import { describe, it, expect } from 'vitest';
import { isValidPlay, getTrickWinner, getTrickPoints, getValidPlays } from '../../game/core/rules';
import { createInitialState } from '../../game/core/state';
import { analyzeSuits } from '../../game/core/suit-analysis';
import { getDominoSuit } from '../../game/core/dominoes';
import type { TrumpSelection, Play, Domino, GameState, LedSuitOrNone } from '../../game/types';
import { TRUMP_SELECTIONS } from '../../game/constants';

// Helper function to create dominoes for testing
function createDomino(high: number, low: number): Domino {
  return {
    id: `${high}-${low}`,
    high,
    low
  };
}

// Helper to create test game state
function createTestState(hands: Domino[][], trump: TrumpSelection, currentTrick: Play[] = []): GameState {
  const state = createInitialState();
  state.phase = 'playing';
  state.trump = trump;
  state.currentTrick = currentTrick;
  
  // Set currentSuit based on lead domino if trick in progress
  if (currentTrick.length > 0) {
    const firstPlay = currentTrick[0];
    if (firstPlay) {
      const leadDomino = firstPlay.domino;
      state.currentSuit = getDominoSuit(leadDomino, trump) as LedSuitOrNone;
    }
  }
  
  // Set up player hands and analyze suits
  hands.forEach((hand, i) => {
    if (i < state.players.length) {
      const player = state.players[i];
      if (player) {
        player.hand = hand;
        player.suitAnalysis = analyzeSuits(hand, trump);
      }
    }
  });
  
  return state;
}

describe('Trick Validation', () => {
  const trump: TrumpSelection = TRUMP_SELECTIONS.BLANKS!;
  
  describe('isValidPlay', () => {
    it('should allow any domino for opening lead', () => {
      const hand = [createDomino(0, 0), createDomino(1, 2), createDomino(3, 4)];
      const state = createTestState([hand, [], [], []], trump);
      
      hand.forEach(domino => {
        expect(isValidPlay(state, domino, 0)).toBe(true);
      });
    });

    it('should require following suit when possible', () => {
      const hand = [
        createDomino(0, 1), // has blank
        createDomino(2, 3),  // no blank
        createDomino(4, 5)   // no blank
      ];
      const currentTrick: Play[] = [
        { player: 1, domino: createDomino(0, 2) } // lead with blank
      ];
      const state = createTestState([hand, [], [], []], trump, currentTrick);
      
      expect(isValidPlay(state, hand[0]!, 0)).toBe(true);  // must follow
      expect(isValidPlay(state, hand[1]!, 0)).toBe(false); // can't play off-suit
      expect(isValidPlay(state, hand[2]!, 0)).toBe(false); // can't play off-suit
    });

    it('should allow any domino when cannot follow suit', () => {
      const hand = [
        createDomino(1, 2), // no blank
        createDomino(3, 4), // no blank
        createDomino(5, 6)  // no blank
      ];
      const currentTrick: Play[] = [
        { player: 1, domino: createDomino(0, 0) } // lead with double blank
      ];
      const state = createTestState([hand, [], [], []], trump, currentTrick);
      
      hand.forEach(domino => {
        expect(isValidPlay(state, domino, 0)).toBe(true);
      });
    });

    it('should handle trump suit correctly', () => {
      const trumpSuit = TRUMP_SELECTIONS.ONES;
      if (!trumpSuit) throw new Error('TRUMP_SELECTIONS.ONES is undefined');
      const hand = [
        createDomino(1, 1), // trump
        createDomino(2, 3), // not trump
        createDomino(0, 4)  // not trump
      ];
      const currentTrick: Play[] = [
        { player: 1, domino: createDomino(1, 2) } // lead with trump suit
      ];
      const state = createTestState([hand, [], [], []], trumpSuit, currentTrick);
      state.trump = trumpSuit;
      
      const trumpDomino = hand[0];
      const nonTrumpDomino1 = hand[1];
      const nonTrumpDomino2 = hand[2];
      if (!trumpDomino || !nonTrumpDomino1 || !nonTrumpDomino2) throw new Error('Hand dominoes are undefined');
      
      expect(isValidPlay(state, trumpDomino, 0)).toBe(true);  // can follow trump
      expect(isValidPlay(state, nonTrumpDomino1, 0)).toBe(false); // must follow trump
      expect(isValidPlay(state, nonTrumpDomino2, 0)).toBe(false); // must follow trump
    });
  });

  describe('getValidPlays', () => {
    it('should return all dominoes for opening lead', () => {
      const hand = [createDomino(0, 0), createDomino(1, 2), createDomino(3, 4)];
      const state = createTestState([hand, [], [], []], trump);
      
      const validPlays = getValidPlays(state, 0);
      expect(validPlays).toHaveLength(3);
      expect(validPlays).toEqual(expect.arrayContaining(hand));
    });

    it('should return only suit-following dominoes when must follow suit', () => {
      const hand = [
        createDomino(0, 1), // has blank
        createDomino(0, 5), // has blank
        createDomino(2, 3),  // no blank
      ];
      const currentTrick: Play[] = [
        { player: 1, domino: createDomino(0, 2) } // lead with blank
      ];
      const state = createTestState([hand, [], [], []], trump, currentTrick);
      
      const validPlays = getValidPlays(state, 0);
      expect(validPlays).toHaveLength(2);
      const firstSuitDomino = hand[0];
      const secondSuitDomino = hand[1];
      if (!firstSuitDomino || !secondSuitDomino) throw new Error('Expected hand dominoes are undefined');
      expect(validPlays).toEqual(expect.arrayContaining([firstSuitDomino, secondSuitDomino]));
    });
  });

  describe('getTrickWinner', () => {
    it('should identify winner of basic trick', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(1, 2) }, // leads with 2s
        { player: 1, domino: createDomino(1, 4) }, // 4s - can't follow suit
        { player: 2, domino: createDomino(1, 6) }, // 6s - can't follow suit  
        { player: 3, domino: createDomino(2, 3) }  // 3s - CAN follow suit with higher value
      ];
      
      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = getTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(3); // Player 3 wins by following suit with higher value
    });

    it('should handle trump plays correctly', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(1, 2) },
        { player: 1, domino: createDomino(0, 4) }, // trump (blank)
        { player: 2, domino: createDomino(1, 6) },
        { player: 3, domino: createDomino(2, 3) }
      ];
      
      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = getTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(1); // Player 1 with trump
    });

    it('should handle multiple trump plays', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(0, 1) }, // trump
        { player: 1, domino: createDomino(0, 4) }, // trump
        { player: 2, domino: createDomino(0, 6) }, // trump (highest)
        { player: 3, domino: createDomino(2, 3) }
      ];
      
      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = getTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(2); // Player 2 with highest trump
    });

    it('should handle double blank (highest trump)', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(0, 1) }, // trump
        { player: 1, domino: createDomino(0, 0) }, // double blank (highest trump)
        { player: 2, domino: createDomino(0, 6) }, // trump
        { player: 3, domino: createDomino(2, 3) }
      ];
      
      const firstPlay = trick[0];
      if (!firstPlay) throw new Error('First play in trick is undefined');
      const leadSuit = getDominoSuit(firstPlay.domino, trump);
      const winner = getTrickWinner(trick, trump, leadSuit);
      expect(winner).toBe(1); // Player 1 with double blank
    });
  });

  describe('getTrickPoints', () => {
    it('should calculate points correctly for basic dominoes', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(1, 4) }, // 5 points
        { player: 1, domino: createDomino(2, 3) }, // 5 points
        { player: 2, domino: createDomino(0, 0) }, // 0 points
        { player: 3, domino: createDomino(6, 6) }  // 0 points
      ];
      
      expect(getTrickPoints(trick)).toBe(10);
    });

    it('should handle special point dominoes', () => {
      const trick: Play[] = [
        { player: 0, domino: createDomino(5, 0) }, // 5 points
        { player: 1, domino: createDomino(4, 1) }, // 5 points  
        { player: 2, domino: createDomino(3, 2) }, // 5 points
        { player: 3, domino: createDomino(6, 4) }  // 10 points
      ];
      
      expect(getTrickPoints(trick)).toBe(25);
    });

    it('should handle empty trick', () => {
      expect(getTrickPoints([])).toBe(0);
    });
  });
});