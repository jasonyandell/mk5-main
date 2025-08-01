import { describe, it, expect } from 'vitest';
import { isValidPlay, getValidPlays } from '../../game/core/rules';
import { createInitialState } from '../../game/core/state';
import { analyzeSuits } from '../../game/core/suit-analysis';
import { GameTestHelper } from '../helpers/gameTestHelper';
import type { Domino, GameState } from '../../game/types';

describe('Hand Validation Rules', () => {
  function createTestState(options: {
    trump: number,
    currentTrick: { player: number; domino: Domino }[],
    playerHand: Domino[],
    currentPlayer?: number
  }): GameState {
    const state = createInitialState();
    state.phase = 'playing';
    state.trump = options.trump;
    state.currentTrick = options.currentTrick;
    state.currentPlayer = options.currentPlayer || 1;
    
    // Set currentSuit based on the first domino in currentTrick
    if (options.currentTrick.length > 0) {
      const leadDomino = options.currentTrick[0].domino;
      if (options.trump === 7) { // doubles are trump
        state.currentSuit = leadDomino.high === leadDomino.low ? 7 : Math.max(leadDomino.high, leadDomino.low);
      } else if (leadDomino.high === options.trump || leadDomino.low === options.trump) {
        state.currentSuit = options.trump; // trump was led
      } else {
        state.currentSuit = Math.max(leadDomino.high, leadDomino.low); // higher end for non-trump
      }
    } else {
      state.currentSuit = null;
    }
    
    const player = state.players[state.currentPlayer];
    player.hand = options.playerHand;
    player.suitAnalysis = analyzeSuits(options.playerHand, state.trump);
    
    return state;
  }

  describe('Must Follow Suit When Able', () => {
    it('should require following suit when player has matching suit', () => {
      const playerHand: Domino[] = [
        { id: 8, low: 1, high: 4 },  // [4|1] - must play this
        { id: 5, low: 2, high: 3 }   // [2|3] - cannot play this
      ];

      const state = createTestState({
        trump: 5, // fives are trump
        currentTrick: [
          { domino: { id: 12, low: 3, high: 4 }, player: 0 } // Led with fours (high end)
        ],
        playerHand
      });

      const validPlays = getValidPlays(state, state.currentPlayer);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(8); // Must play [4|1]
      
      // Verify invalid play is rejected
      const invalidPlay = { id: 5, low: 2, high: 3 };
      expect(isValidPlay(state, invalidPlay, state.currentPlayer)).toBe(false);
    });

    it('should allow trump when unable to follow suit', () => {
      const playerHand: Domino[] = [
        { id: 10, low: 0, high: 5 },  // [5|0] - trump
        { id: 5, low: 2, high: 3 }    // [2|3] - not trump, not fours
      ];

      const state = createTestState({
        trump: 5, // fives are trump
        currentTrick: [
          { domino: { id: 12, low: 3, high: 4 }, player: 0 } // Led with fours
        ],
        playerHand
      });

      const validPlays = getValidPlays(state, state.currentPlayer);
      expect(validPlays).toHaveLength(2); // Can play either trump or any domino
      
      // Verify trump is valid
      const trumpPlay = { id: 10, low: 0, high: 5 };
      expect(isValidPlay(state, trumpPlay, state.currentPlayer)).toBe(true);
    });

    it('should allow any domino when unable to follow suit or trump', () => {
      const playerHand: Domino[] = [
        { id: 1, low: 0, high: 1 },   // [1|0]
        { id: 5, low: 2, high: 3 }    // [2|3]
      ];

      const state = createTestState({
        trump: 6, // sixes are trump
        currentTrick: [
          { domino: { id: 12, low: 3, high: 4 }, player: 0 } // Led with fours
        ],
        playerHand
      });

      const validPlays = getValidPlays(state, state.currentPlayer);
      expect(validPlays).toHaveLength(2); // Can play any domino
      
      validPlays.forEach(domino => {
        expect(isValidPlay(state, domino, state.currentPlayer)).toBe(true);
      });
    });
  });

  describe('Doubles Natural Suit Rule', () => {
    it('should treat doubles as highest in their natural suit', () => {
      const playerHand: Domino[] = [
        { id: 6, low: 3, high: 3 },   // [3|3] - highest three, must play
        { id: 1, low: 0, high: 1 }    // [1|0] - cannot play
      ];

      const state = createTestState({
        trump: 3, // threes are trump  
        currentTrick: [
          { domino: { id: 5, low: 2, high: 3 }, player: 0 } // Led with threes
        ],
        playerHand
      });

      const validPlays = getValidPlays(state, state.currentPlayer);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(6); // Must play [3|3]
    });

    it('should recognize doubles trump when trump is declared', () => {
      const playerHand: Domino[] = [
        { id: 0, low: 0, high: 0 },   // [0|0] - double (trump)
        { id: 6, low: 3, high: 3 },   // [3|3] - double (trump)
        { id: 5, low: 2, high: 3 }    // [2|3] - not trump
      ];

      const state = createTestState({
        trump: 7, // doubles are trump
        currentTrick: [
          { domino: { id: 10, low: 4, high: 4 }, player: 0 } // Led with double (trump)
        ],
        playerHand
      });

      const validPlays = getValidPlays(state, state.currentPlayer);
      expect(validPlays).toHaveLength(2); // Must play a double
      expect(validPlays.every(d => d.high === d.low)).toBe(true);
    });
  });

  describe('Higher End Determines Suit Led', () => {
    it('should identify led suit from higher end of non-trump domino', () => {
      const playerHand: Domino[] = [
        { id: 10, low: 0, high: 5 },  // [5|0] - must play this for fives
        { id: 12, low: 3, high: 4 }   // [4|3] - cannot play
      ];

      const state = createTestState({
        trump: 6, // sixes are trump
        currentTrick: [
          { domino: { id: 9, low: 1, high: 5 }, player: 0 } // Led [5|1], suit is fives (non-trump)
        ],
        playerHand
      });

      const validPlays = getValidPlays(state, state.currentPlayer);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(10); // Must play [5|0]
    });

    it('should handle trump domino led (any suit can follow)', () => {
      const playerHand: Domino[] = [
        { id: 8, low: 1, high: 4 },   // [4|1] - has trump
        { id: 5, low: 2, high: 3 }    // [2|3] - no trump
      ];

      const state = createTestState({
        trump: 1, // ones are trump
        currentTrick: [
          { domino: { id: 7, low: 1, high: 3 }, player: 0 } // Led with trump [3|1]
        ],
        playerHand
      });

      const validPlays = getValidPlays(state, state.currentPlayer);
      expect(validPlays).toHaveLength(1); // Must follow trump
      expect(validPlays[0].id).toBe(8); // [4|1] is the only trump
    });
  });

  describe('No-Trump (Follow-Me) Rules', () => {
    it('should enforce suit following in no-trump games', () => {
      const playerHand: Domino[] = [
        { id: 12, low: 3, high: 4 },  // [4|3] - must play this
        { id: 5, low: 2, high: 3 }    // [2|3] - cannot play
      ];

      const state = createTestState({
        trump: 8, // no-trump
        currentTrick: [
          { domino: { id: 8, low: 1, high: 4 }, player: 0 } // Led [4|1], suit is fours
        ],
        playerHand
      });

      const validPlays = getValidPlays(state, state.currentPlayer);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(12); // Must play [4|3]
    });
  });
});