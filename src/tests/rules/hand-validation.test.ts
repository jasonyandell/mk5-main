import { describe, it, expect } from 'vitest';
import { isValidPlay, getValidPlays } from '../../game/core/rules';
import { GameTestHelper } from '../helpers/gameTestHelper';
import type { Domino } from '../../game/types';

describe('Hand Validation Rules', () => {
  describe('Must Follow Suit When Able', () => {
    it('should require following suit when player has matching suit', () => {
      const state = GameTestHelper.createPlayingState({
        trump: { suit: 'fives', followsSuit: true },
        currentTrick: [
          { domino: { id: 12, low: 3, high: 4 }, player: 0 } // Led with fours (high end)
        ],
        currentPlayer: 1
      });

      // Player has [4|1] and [2|3] - must play [4|1] to follow fours
      const playerHand: Domino[] = [
        { id: 8, low: 1, high: 4 },  // [4|1] - must play this
        { id: 5, low: 2, high: 3 }   // [2|3] - cannot play this
      ];

      const validPlays = getValidPlays(state, playerHand);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(8); // Must play [4|1]
      
      // Verify invalid play is rejected
      const invalidPlay = { id: 5, low: 2, high: 3 };
      expect(isValidPlay(state, invalidPlay, playerHand)).toBe(false);
    });

    it('should allow trump when unable to follow suit', () => {
      const state = GameTestHelper.createPlayingState({
        trump: { suit: 'fives', followsSuit: true },
        currentTrick: [
          { domino: { id: 12, low: 3, high: 4 }, player: 0 } // Led with fours
        ],
        currentPlayer: 1
      });

      // Player has no fours but has trump
      const playerHand: Domino[] = [
        { id: 10, low: 0, high: 5 },  // [5|0] - trump
        { id: 5, low: 2, high: 3 }    // [2|3] - not trump, not fours
      ];

      const validPlays = getValidPlays(state, playerHand);
      expect(validPlays).toHaveLength(2); // Can play either trump or any domino
      
      // Verify trump is valid
      const trumpPlay = { id: 10, low: 0, high: 5 };
      expect(isValidPlay(state, trumpPlay, playerHand)).toBe(true);
    });

    it('should allow any domino when unable to follow suit or trump', () => {
      const state = GameTestHelper.createPlayingState({
        trump: { suit: 'sixes', followsSuit: true },
        currentTrick: [
          { domino: { id: 12, low: 3, high: 4 }, player: 0 } // Led with fours
        ],
        currentPlayer: 1
      });

      // Player has no fours and no sixes
      const playerHand: Domino[] = [
        { id: 1, low: 0, high: 1 },   // [1|0]
        { id: 5, low: 2, high: 3 }    // [2|3]
      ];

      const validPlays = getValidPlays(state, playerHand);
      expect(validPlays).toHaveLength(2); // Can play any domino
      
      validPlays.forEach(domino => {
        expect(isValidPlay(state, domino, playerHand)).toBe(true);
      });
    });
  });

  describe('Doubles Natural Suit Rule', () => {
    it('should treat doubles as highest in their natural suit', () => {
      const state = GameTestHelper.createPlayingState({
        trump: { suit: 'threes', followsSuit: true },
        currentTrick: [
          { domino: { id: 5, low: 2, high: 3 }, player: 0 } // Led with threes
        ],
        currentPlayer: 1
      });

      // Player has [3|3] - should be required to play as it's a three
      const playerHand: Domino[] = [
        { id: 6, low: 3, high: 3 },   // [3|3] - highest three, must play
        { id: 1, low: 0, high: 1 }    // [1|0] - cannot play
      ];

      const validPlays = getValidPlays(state, playerHand);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(6); // Must play [3|3]
    });

    it('should recognize doubles trump when trump is declared', () => {
      const state = GameTestHelper.createPlayingState({
        trump: { suit: 'doubles', followsSuit: false },
        currentTrick: [
          { domino: { id: 1, low: 0, high: 1 }, player: 0 } // Led with ones
        ],
        currentPlayer: 1
      });

      // When doubles are trump, player must follow led suit unless playing trump
      const playerHand: Domino[] = [
        { id: 6, low: 3, high: 3 },   // [3|3] - trump (double)
        { id: 8, low: 1, high: 4 },   // [4|1] - has led suit (ones)
        { id: 5, low: 2, high: 3 }    // [2|3] - no led suit
      ];

      const validPlays = getValidPlays(state, playerHand);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(8); // Must follow ones with [4|1]
    });
  });

  describe('Higher End Determines Suit Led', () => {
    it('should identify led suit from higher end of non-trump domino', () => {
      const state = GameTestHelper.createPlayingState({
        trump: { suit: 'fives', followsSuit: true },
        currentTrick: [
          { domino: { id: 18, low: 2, high: 6 }, player: 0 } // [6|2] led - sixes is suit
        ],
        currentPlayer: 1
      });

      // Player must follow sixes if able
      const playerHand: Domino[] = [
        { id: 27, low: 6, high: 6 },  // [6|6] - highest six, must play
        { id: 5, low: 2, high: 3 }    // [2|3] - not sixes
      ];

      const validPlays = getValidPlays(state, playerHand);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(27); // Must play [6|6]
    });

    it('should handle trump domino led (any suit can follow)', () => {
      const state = GameTestHelper.createPlayingState({
        trump: { suit: 'fives', followsSuit: true },
        currentTrick: [
          { domino: { id: 10, low: 0, high: 5 }, player: 0 } // [5|0] led (trump)
        ],
        currentPlayer: 1
      });

      // When trump is led, others must follow trump if able
      const playerHand: Domino[] = [
        { id: 15, low: 5, high: 5 },  // [5|5] - trump
        { id: 5, low: 2, high: 3 }    // [2|3] - not trump
      ];

      const validPlays = getValidPlays(state, playerHand);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(15); // Must play trump [5|5]
    });
  });

  describe('No-Trump (Follow-Me) Rules', () => {
    it('should enforce suit following in no-trump games', () => {
      const state = GameTestHelper.createPlayingState({
        trump: { suit: 'no-trump', followsSuit: false },
        currentTrick: [
          { domino: { id: 12, low: 3, high: 4 }, player: 0 } // Led with fours
        ],
        currentPlayer: 1
      });

      // In no-trump, still must follow suit if able
      const playerHand: Domino[] = [
        { id: 8, low: 1, high: 4 },   // [4|1] - has fours, must play
        { id: 5, low: 2, high: 3 }    // [2|3] - cannot play
      ];

      const validPlays = getValidPlays(state, playerHand);
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe(8); // Must follow fours
    });
  });
});