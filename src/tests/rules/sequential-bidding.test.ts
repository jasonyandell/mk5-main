import { describe, it, expect } from 'vitest';
import type { GameState, Bid } from '../../game/types';
import { createInitialState, getNextStates, getPlayerLeftOfDealer } from '../../game';
import { createTestContext } from '../helpers/executionContext';

describe('Sequential Bidding', () => {
  const ctx = createTestContext();
  // Helper to create a state with specific bids already made
  function createStateWithBids(bids: Bid[]): GameState {
    const state = createInitialState({ shuffleSeed: 12345 });
    state.phase = 'bidding';
    state.dealer = 3; // Set dealer so player 0 bids first
    state.currentPlayer = getPlayerLeftOfDealer(state.dealer); // Player 0
    state.bids = [];
    
    // Apply each bid using the game engine
    for (const bid of bids) {
      // Find the matching transition
      const transitions = getNextStates(state, ctx);
      const transition = transitions.find(t => {
        if (bid.type === 'pass') return t.id === 'pass';
        if (bid.type === 'points') return t.id === `bid-${bid.value}`;
        if (bid.type === 'marks') return t.id === `bid-${bid.value}-marks`; // Always use plural
        return false;
      });
      
      if (transition) {
        Object.assign(state, transition.newState);
      }
    }
    
    return state;
  }

  // Helper to check if a specific bid is valid using the game engine
  function checkBidValidity(state: GameState, bidType: 'points' | 'marks', bidValue: number): boolean {
    const transitions = getNextStates(state, ctx);
    const bidId = bidType === 'points' 
      ? `bid-${bidValue}`
      : `bid-${bidValue}-marks`; // Always use plural "marks"
    
    return transitions.some(t => t.id === bidId);
  }

  describe('Given at least one bid has been made', () => {
    describe('When a player bids', () => {
      it('Then their bid must exceed the previous bid', () => {
        const state = createStateWithBids([
          { type: 'points', value: 30, player: 0 }
        ]);

        // Valid bids that exceed 30
        expect(checkBidValidity(state, 'points', 31)).toBe(true);
        expect(checkBidValidity(state, 'points', 35)).toBe(true);
        expect(checkBidValidity(state, 'points', 41)).toBe(true);

        // Invalid bids that don't exceed 30
        expect(checkBidValidity(state, 'points', 30)).toBe(false);
        expect(checkBidValidity(state, 'points', 29)).toBe(false);
      });

      it('And after reaching 42 (1 mark), subsequent bids must be in mark increments', () => {
        // State with 41 point bid - need to have 4 players bid to stay in bidding phase
        const state41 = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'points', value: 41, player: 1 },
          { type: 'pass', player: 2 }
          // Player 3 is the current player
        ]);

        // Can bid 1 mark (42 points) or higher marks
        expect(checkBidValidity(state41, 'marks', 1)).toBe(true);
        expect(checkBidValidity(state41, 'marks', 2)).toBe(true);

        // State with 1 mark bid (42 points)
        const state1Mark = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 1, player: 1 }
        ]);

        // Cannot bid point values after reaching 1 mark
        expect(checkBidValidity(state1Mark, 'points', 43)).toBe(false);
        expect(checkBidValidity(state1Mark, 'points', 50)).toBe(false);

        // Must bid in mark increments
        expect(checkBidValidity(state1Mark, 'marks', 2)).toBe(true);
      });

      it('And any player may bid up to 2 marks when 2 marks has not already been bid', () => {
        // No 2 marks bid yet
        const state = createStateWithBids([
          { type: 'points', value: 30, player: 0 }
        ]);

        // Can jump to 2 marks
        expect(checkBidValidity(state, 'marks', 2)).toBe(true);

        // With 1 mark bid, can still bid 2 marks
        const state1Mark = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 1, player: 1 }
        ]);

        expect(checkBidValidity(state1Mark, 'marks', 2)).toBe(true);
      });

      it('And subsequent bids after 2 marks may only be one additional mark', () => {
        // State with 2 marks bid
        const state2Marks = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 2, player: 1 }
        ]);

        // Can only bid 3 marks (one additional)
        expect(checkBidValidity(state2Marks, 'marks', 3)).toBe(true);
        expect(checkBidValidity(state2Marks, 'marks', 4)).toBe(false);
        expect(checkBidValidity(state2Marks, 'marks', 5)).toBe(false);

        // State with 3 marks bid
        const state3Marks = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 2, player: 1 },
          { type: 'marks', value: 3, player: 2 }
        ]);

        // Can only bid 4 marks (one additional)
        expect(checkBidValidity(state3Marks, 'marks', 4)).toBe(true);
        expect(checkBidValidity(state3Marks, 'marks', 5)).toBe(false);
      });
    });
  });

  describe('Sequential bidding progression examples', () => {
    it('should allow normal point bidding progression', () => {
      let state = createInitialState({ shuffleSeed: 12345 });
      state.phase = 'bidding';
      state.dealer = 3;
      state.currentPlayer = 0;
      
      // Player 0 opens with 30
      expect(checkBidValidity(state, 'points', 30)).toBe(true);
      state = createStateWithBids([{ type: 'points', value: 30, player: 0 }]);

      // Player 1 bids 32
      expect(checkBidValidity(state, 'points', 32)).toBe(true);
      state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'points', value: 32, player: 1 }
      ]);

      // Player 2 bids 35
      expect(checkBidValidity(state, 'points', 35)).toBe(true);
    });

    it('should enforce mark increment rules after reaching 1 mark', () => {
      // Start with bids up to 41 - need to keep in bidding phase
      let state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'points', value: 35, player: 1 },
        { type: 'points', value: 41, player: 2 }
        // Player 3 is the current player
      ]);

      // Next player can bid 1 mark or jump to 2 marks
      expect(checkBidValidity(state, 'marks', 1)).toBe(true);
      expect(checkBidValidity(state, 'marks', 2)).toBe(true);

      // After 1 mark is bid, only marks allowed - test during the bidding
      state = createStateWithBids([
        { type: 'pass', player: 0 },
        { type: 'marks', value: 1, player: 1 },
        { type: 'pass', player: 2 }
        // Player 3 is the current player
      ]);

      expect(checkBidValidity(state, 'points', 43)).toBe(false); // No point bids allowed
      expect(checkBidValidity(state, 'marks', 2)).toBe(true); // Can bid 2 marks
    });

    it('should enforce one-mark increment rule after 2 marks', () => {
      // Someone bid 2 marks directly
      let state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 2, player: 1 }
      ]);

      // Next valid bid is only 3 marks
      expect(checkBidValidity(state, 'marks', 3)).toBe(true);
      expect(checkBidValidity(state, 'marks', 4)).toBe(false);

      // Continue the progression
      state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 2, player: 1 },
        { type: 'marks', value: 3, player: 2 }
      ]);

      // Next valid bid is only 4 marks
      expect(checkBidValidity(state, 'marks', 4)).toBe(true);
      expect(checkBidValidity(state, 'marks', 5)).toBe(false);
    });
  });
});