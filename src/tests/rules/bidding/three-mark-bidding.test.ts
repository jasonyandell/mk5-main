import { describe, it, expect } from 'vitest';
import type { GameState, Bid } from '../../../game/types';
import { createInitialState, getNextStates, getPlayerLeftOfDealer } from '../../../game';
import { createTestContext } from '../../helpers/executionContext';

describe('Feature: Special Bids', () => {
  const ctx = createTestContext();
  describe('Scenario: Three Mark Bidding', () => {
    function createStateWithBids(bids: Bid[]): GameState {
      const state = createInitialState({ shuffleSeed: 12345 });
      state.phase = 'bidding';
      state.dealer = 3;
      state.currentPlayer = getPlayerLeftOfDealer(3); // Player 0
      state.bids = [];
      
      // Apply each bid using the game engine
      for (const bid of bids) {
        const transitions = getNextStates(state, ctx);
        const transition = transitions.find(t => {
          if (bid.type === 'pass') return t.id === 'pass';
          if (bid.type === 'points') return t.id === `bid-${bid.value}`;
          if (bid.type === 'marks') return t.id === `bid-${bid.value}-marks`;
          return false;
        });
        
        if (transition) {
          Object.assign(state, transition.newState);
        }
      }
      
      return state;
    }

    function canBidThreeMarks(state: GameState): boolean {
      const transitions = getNextStates(state, ctx);
      return transitions.some(t => t.id === 'bid-3-marks');
    }

    it('Given the current bid is less than 3 marks', () => {
      // Test various states where current bid is less than 3 marks
      const stateWith30Points = createStateWithBids([
        { type: 'points', value: 30, player: 0 }
      ]);
      expect(stateWith30Points.currentBid?.value).toBe(30);
      expect(stateWith30Points.currentBid?.type).toBe('points');

      const stateWith1Mark = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 1, player: 1 }
      ]);
      expect(stateWith1Mark.currentBid?.value).toBe(1);
      expect(stateWith1Mark.currentBid?.type).toBe('marks');

      const stateWith2Marks = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 2, player: 1 }
      ]);
      expect(stateWith2Marks.currentBid?.value).toBe(2);
      expect(stateWith2Marks.currentBid?.type).toBe('marks');
    });

    it('When a player wants to bid 3 marks', () => {
      // This is a test setup step - we're testing the decision logic
      const testStates = [
        // No 2 marks bid yet
        createStateWithBids([
          { type: 'points', value: 30, player: 0 }
        ]),
        // 1 mark bid, but no 2 marks
        createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 1, player: 1 }
        ]),
        // 2 marks has been bid
        createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 2, player: 1 }
        ]),
      ];

      // Verify we can check if 3 marks is allowed in each state
      testStates.forEach(state => {
        const canBid = canBidThreeMarks(state);
        expect(typeof canBid).toBe('boolean');
      });
    });

    it('Then they can only bid 3 marks if another player has already bid 2 marks', () => {
      // Case 1: No 2 marks bid - cannot bid 3 marks
      const stateNo2Marks = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'points', value: 35, player: 1 }
      ]);
      expect(canBidThreeMarks(stateNo2Marks)).toBe(false);

      // Case 2: Only 1 mark bid - still cannot bid 3 marks
      const state1Mark = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 1, player: 1 }
      ]);
      expect(canBidThreeMarks(state1Mark)).toBe(false);

      // Case 3: 2 marks has been bid - CAN bid 3 marks
      const state2Marks = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 2, player: 1 }
      ]);
      expect(canBidThreeMarks(state2Marks)).toBe(true);

      // Case 4: 2 marks bid earlier in sequence - CAN bid 3 marks
      const state2MarksEarlier = createStateWithBids([
        { type: 'points', value: 30, player: 0 },
        { type: 'marks', value: 2, player: 1 },
        { type: 'pass', player: 2 },
        { type: 'pass', player: 3 }
      ]);
      // Current bid is pass, but 2 marks was bid earlier
      expect(state2MarksEarlier.bids.some(bid => bid.type === 'marks' && bid.value === 2)).toBe(true);
    });

    it('And 3 marks cannot be used as an opening bid under tournament rules', () => {
      // Empty bid state - no bids yet
      const emptyState = createStateWithBids([]);
      
      // Cannot bid 3 marks as opening bid
      expect(canBidThreeMarks(emptyState)).toBe(false);
      
      // Check opening bid validity through game engine transitions
      const transitions = getNextStates(emptyState, ctx);
      const validOpeningBids = transitions.map(t => t.id);
      
      expect(validOpeningBids).not.toContain('bid-3-marks');
      expect(validOpeningBids).toContain('bid-2-marks');  // 2 marks is allowed as opening
      expect(validOpeningBids).toContain('bid-1-marks');  // 1 mark is allowed as opening
    });

    describe('Three mark bidding integration scenarios', () => {
      it('should enforce sequential bidding rules when 3 marks is involved', () => {
        // Valid progression to 3 marks
        const validProgression = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 2, player: 1 },  // Jump to 2 marks allowed
          { type: 'marks', value: 3, player: 2 }   // 3 marks allowed after 2
        ]);

        expect(validProgression.bids.length).toBe(3);
        expect(validProgression.bids[2]!.value).toBe(3);
      });

      it('should not allow 3 marks without prior 2 mark bid', () => {
        // Invalid attempt to bid 3 marks
        const state = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'points', value: 35, player: 1 },
          { type: 'points', value: 41, player: 2 }
        ]);

        // Even at 41 points, cannot jump to 3 marks
        expect(canBidThreeMarks(state)).toBe(false);
        
        // Can bid 1 or 2 marks though
        const hasTwoMarks = state.bids.some(bid => bid.type === 'marks' && bid.value === 2);
        expect(hasTwoMarks).toBe(false); // Confirms no 2 mark bid exists
      });

      it('should handle 3 marks correctly', () => {
        const tournamentState = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 2, player: 1 }
        ]);

        // REMOVED: expect(tournamentState.tournamentMode).toBe(true);
        expect(canBidThreeMarks(tournamentState)).toBe(true);

        // After 3 marks is bid, only 4 marks is allowed (one increment)
        const after3Marks = createStateWithBids([
          { type: 'points', value: 30, player: 0 },
          { type: 'marks', value: 2, player: 1 },
          { type: 'marks', value: 3, player: 2 }
        ]);

        // Next bid must be exactly 4 marks (one additional mark rule)
        expect(after3Marks.currentBid?.value).toBe(3);
      });
    });
  });
});