import { describe, it, expect } from 'vitest';
import type { GameState, Bid } from '../../game/types';
import { createInitialState } from '../../game/core/state';

describe('Sequential Bidding', () => {
  // Helper to create a state with specific bids already made
  function createStateWithBids(bids: Bid[]): GameState {
    const state = createInitialState();
    state.bids = bids;
    if (bids.length > 0) {
      state.currentBid = bids[bids.length - 1];
    }
    state.currentPlayer = bids.length % 4; // Next player in clockwise order
    return state;
  }

  // Helper to validate if a bid is valid given the current state
  function isValidBid(state: GameState, bidType: 'points' | 'marks', bidValue: number): boolean {
    const currentBid = state.currentBid;
    
    // No current bid - check opening bid constraints
    if (!currentBid) {
      if (bidType === 'points') {
        return bidValue >= 30 && bidValue <= 41;
      } else if (bidType === 'marks') {
        return bidValue <= 2; // Max opening bid is 2 marks (except plunge)
      }
      return false;
    }

    // Calculate current bid value in points
    const currentBidPoints = currentBid.type === 'marks' 
      ? (currentBid.value || 0) * 42 
      : (currentBid.value || 0);
    
    const newBidPoints = bidType === 'marks' 
      ? bidValue * 42 
      : bidValue;

    // New bid must exceed previous bid
    if (newBidPoints <= currentBidPoints) {
      return false;
    }

    // After reaching 42 (1 mark), subsequent bids must be in mark increments
    if (currentBidPoints >= 42 && bidType === 'points') {
      return false;
    }

    // 2 marks rule: any player may bid up to 2 marks when 2 marks has not already been bid
    const hasTwoMarksBid = state.bids.some(bid => bid.type === 'marks' && bid.value === 2);
    
    if (bidType === 'marks') {
      if (!hasTwoMarksBid && bidValue <= 2) {
        return true;
      }
      // Subsequent bids after 2 marks may only be one additional mark
      if (hasTwoMarksBid && currentBid.type === 'marks') {
        return bidValue === (currentBid.value || 0) + 1;
      }
    }

    return true;
  }

  describe('Given at least one bid has been made', () => {
    describe('When a player bids', () => {
      it('Then their bid must exceed the previous bid', () => {
        const state = createStateWithBids([
          { type: 'points', value: 30, player: 0 }
        ]);

        // Valid bids that exceed 30
        expect(isValidBid(state, 'points', 31)).toBe(true);
        expect(isValidBid(state, 'points', 35)).toBe(true);
        expect(isValidBid(state, 'points', 41)).toBe(true);

        // Invalid bids that don't exceed 30
        expect(isValidBid(state, 'points', 30)).toBe(false);
        expect(isValidBid(state, 'points', 29)).toBe(false);
      });

      it('And after reaching 42 (1 mark), subsequent bids must be in mark increments', () => {
        // State with 41 point bid
        const state41 = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'points', value: 41, player: 1 }
        ]);

        // Can bid 1 mark (42 points) or higher marks
        expect(isValidBid(state41, 'marks', 1)).toBe(true);
        expect(isValidBid(state41, 'marks', 2)).toBe(true);

        // State with 1 mark bid (42 points)
        const state1Mark = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 1, player: 1 }
        ]);

        // Cannot bid point values after reaching 1 mark
        expect(isValidBid(state1Mark, 'points', 43)).toBe(false);
        expect(isValidBid(state1Mark, 'points', 50)).toBe(false);

        // Must bid in mark increments
        expect(isValidBid(state1Mark, 'marks', 2)).toBe(true);
      });

      it('And any player may bid up to 2 marks when 2 marks has not already been bid', () => {
        // No 2 marks bid yet
        const state = createStateWithBids([
          { type: 'points', value: 30, player: 0 }
        ]);

        // Can jump to 2 marks
        expect(isValidBid(state, 'marks', 2)).toBe(true);

        // With 1 mark bid, can still bid 2 marks
        const state1Mark = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 1, player: 1 }
        ]);

        expect(isValidBid(state1Mark, 'marks', 2)).toBe(true);
      });

      it('And subsequent bids after 2 marks may only be one additional mark', () => {
        // State with 2 marks bid
        const state2Marks = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 2, player: 1 }
        ]);

        // Can only bid 3 marks (one additional)
        expect(isValidBid(state2Marks, 'marks', 3)).toBe(true);
        expect(isValidBid(state2Marks, 'marks', 4)).toBe(false);
        expect(isValidBid(state2Marks, 'marks', 5)).toBe(false);

        // State with 3 marks bid
        const state3Marks = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 2, player: 1 },
          { type: 'marks', value: 3, player: 2 }
        ]);

        // Can only bid 4 marks (one additional)
        expect(isValidBid(state3Marks, 'marks', 4)).toBe(true);
        expect(isValidBid(state3Marks, 'marks', 5)).toBe(false);
      });
    });
  });

  describe('Sequential bidding progression examples', () => {
    it('should allow normal point bidding progression', () => {
      let state = createInitialState();
      
      // Player 0 opens with 30
      expect(isValidBid(state, 'points', 30)).toBe(true);
      state = createStateWithBids([{ type: 'points', value: 30, player: 0 }]);

      // Player 1 bids 32
      expect(isValidBid(state, 'points', 32)).toBe(true);
      state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'points', value: 32, player: 1 }
      ]);

      // Player 2 bids 35
      expect(isValidBid(state, 'points', 35)).toBe(true);
    });

    it('should enforce mark increment rules after reaching 1 mark', () => {
      // Start with bids up to 41
      let state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'points', value: 35, player: 1 },
        { type: 'points', value: 41, player: 2 }
      ]);

      // Next player can bid 1 mark or jump to 2 marks
      expect(isValidBid(state, 'marks', 1)).toBe(true);
      expect(isValidBid(state, 'marks', 2)).toBe(true);

      // After 1 mark is bid, only marks allowed
      state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'points', value: 35, player: 1 },
        { type: 'points', value: 41, player: 2 },
        { type: 'marks', value: 1, player: 3 }
      ]);

      expect(isValidBid(state, 'points', 43)).toBe(false); // No point bids allowed
      expect(isValidBid(state, 'marks', 2)).toBe(true); // Can bid 2 marks
    });

    it('should enforce one-mark increment rule after 2 marks', () => {
      // Someone bid 2 marks directly
      let state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 2, player: 1 }
      ]);

      // Next valid bid is only 3 marks
      expect(isValidBid(state, 'marks', 3)).toBe(true);
      expect(isValidBid(state, 'marks', 4)).toBe(false);

      // Continue the progression
      state = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 2, player: 1 },
        { type: 'marks', value: 3, player: 2 }
      ]);

      // Next valid bid is only 4 marks
      expect(isValidBid(state, 'marks', 4)).toBe(true);
      expect(isValidBid(state, 'marks', 5)).toBe(false);
    });
  });
});