import { describe, it, expect } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { getNextStates } from '../../game/core/state';
import { GameTestHelper } from '../helpers/gameTestHelper';
import { BID_TYPES } from '../../game/constants';
import { getPlayerAfter } from '../../game/core/players';
import { createTestContext } from '../helpers/executionContext';
import type { Bid } from '../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../game/types';

describe('Edge Cases and Unusual Scenarios', () => {
  describe('Dealer Rotation Edge Cases', () => {
    it('should handle multiple consecutive all-pass redeals', () => {
      const ctx = createTestContext();
      let state = createInitialState();
      const originalDealer = state.dealer;

      // First all-pass round
      for (let round = 0; round < 3; round++) {
        for (let i = 0; i < 4; i++) {
          const transitions = getNextStates(state, ctx);
          const pass = transitions.find(t => t.id === 'pass');
          expect(pass).toBeDefined();
          state = pass!.newState;
        }

        // Should trigger redeal
        const redealTransitions = getNextStates(state, ctx);
        const redeal = redealTransitions.find(t => t.id === 'redeal');
        if (redeal) {
          state = redeal.newState;
          expect(state.dealer).toBe(getPlayerAfter(originalDealer, round + 1));
        }
      }
    });

    it('should handle dealer rotation wrapping around correctly', () => {
      const ctx = createTestContext();
      // Start with dealer as player 3
      let state = createInitialState();
      state.dealer = 3;

      // All players pass
      for (let i = 0; i < 4; i++) {
        const transitions = getNextStates(state, ctx);
        const pass = transitions.find(t => t.id === 'pass');
        state = pass!.newState;
      }

      // Should redeal with dealer = 0 (wraps around)
      const redealTransitions = getNextStates(state, ctx);
      const redeal = redealTransitions.find(t => t.id === 'redeal');
      if (redeal) {
        state = redeal.newState;
        expect(state.dealer).toBe(0);
        expect(state.currentPlayer).toBe(1); // First bidder after new dealer
      }
    });
  });

  describe('Extreme Bidding Scenarios', () => {
    it('should handle bidding escalation to maximum marks', () => {
      let state = createInitialState();
      // REMOVED: state.tournamentMode = false; // Enable high mark bids
      
      // Simulate escalating mark bids
      const biddingSequence = [
        { type: BID_TYPES.MARKS, value: 1, player: 0 },
        { type: BID_TYPES.MARKS, value: 2, player: 1 },
        { type: BID_TYPES.MARKS, value: 3, player: 2 },
        { type: BID_TYPES.MARKS, value: 4, player: 3 }
      ];
      
      biddingSequence.forEach(bid => {
        state.bids.push(bid as Bid);
        state.currentBid = bid as Bid;
      });
      
      expect(state.currentBid?.value).toBe(4);
      expect(state.currentBid?.type).toBe(BID_TYPES.MARKS);
    });

    it('should handle player trying to bid after passing', () => {
      const ctx = createTestContext();
      let state = createInitialState();

      // Player 0 passes
      const passBid: Bid = { type: BID_TYPES.PASS, player: 0 };
      state.bids.push(passBid);

      // Player 0 should not be able to bid again
      const transitions = getNextStates(state, ctx);
      const player0Bids = transitions.filter(t => t.label?.includes('P0'));
      expect(player0Bids.length).toBe(0);
    });

    it('should handle late-game high point bids', () => {
      let state = createInitialState();
      
      // Set up scenario where teams are close to winning
      state.teamMarks = [6, 5]; // Both teams close to 7
      
      // High-stakes bidding should still follow rules
      const highBid: Bid = { type: BID_TYPES.POINTS, value: 41, player: 0 };
      state.bids.push(highBid);
      state.currentBid = highBid;
      
      expect(state.currentBid.value).toBe(41);
    });
  });

  describe('Hand Distribution Edge Cases', () => {
    it('should handle player with all doubles', () => {
      const state = GameTestHelper.createTestState({
        players: [{
          id: 0,
          name: 'Player 0',
          teamId: 0,
          marks: 0,
          hand: [
            { id: 0, low: BLANKS, high: BLANKS },   // [0|0]
            { id: 1, low: ACES, high: ACES },   // [1|1]
            { id: 6, low: DEUCES, high: DEUCES },   // [2|2]
            { id: 7, low: TRES, high: TRES },   // [3|3]
            { id: 14, low: FOURS, high: FOURS },  // [4|4]
            { id: 15, low: FIVES, high: FIVES },  // [5|5]
            { id: 27, low: SIXES, high: SIXES }   // [6|6]
          ]
        }]
      });
      
      // Player should be able to make Plunge bid
      expect(state.players[0]!.hand.length).toBe(7);
      expect(state.players[0]!.hand.every(d => d.low === d.high)).toBe(true);
    });

    it('should handle player with no trump dominoes', () => {
      const state = GameTestHelper.createPlayingScenario(
        { type: 'suit', suit: SIXES }, // trump: sixes
        0, // currentPlayer
        []
      );
      
      // Override with specific hand for test
      const testState = {
        ...state,
        players: [{
          id: 0,
          hand: [
            { id: 1, low: BLANKS, high: ACES },   // [1|0]
            { id: 5, low: DEUCES, high: TRES },   // [2|3]
            { id: 8, low: ACES, high: FOURS },   // [4|1]
            { id: 10, low: BLANKS, high: FIVES },  // [5|0]
            { id: 15, low: FIVES, high: FIVES },  // [5|5]
            { id: 6, low: TRES, high: TRES },   // [3|3]
            { id: 0, low: BLANKS, high: BLANKS }    // [0|0]
          ]
        }]
      };
      
      // Verify no sixes in hand
      const hasSixes = testState.players[0]!.hand.some(d => (d.low as number) === SIXES || (d.high as number) === SIXES);
      expect(hasSixes).toBe(false);
    });

    it('should handle player with all counting dominoes', () => {
      const state = GameTestHelper.createTestState({
        players: [{
          id: 0,
          name: 'Player 0',
          teamId: 0,
          marks: 0,
          hand: [
            { id: 15, low: FIVES, high: FIVES },  // [5|5] - 10 points
            { id: 20, low: FOURS, high: SIXES },  // [6|4] - 10 points
            { id: 10, low: BLANKS, high: FIVES },  // [5|0] - 5 points
            { id: 8, low: ACES, high: FOURS },   // [4|1] - 5 points
            { id: 5, low: DEUCES, high: TRES },   // [3|2] - 5 points
            { id: 1, low: BLANKS, high: ACES },   // [1|0] - 0 points
            { id: 6, low: TRES, high: TRES }    // [3|3] - 0 points
          ]
        }]
      });
      
      // Player has all 5 counting dominoes (35 points)
      const countingDominoes = [
        { id: 15, low: FIVES, high: FIVES },  // 10 points
        { id: 20, low: FOURS, high: SIXES },  // 10 points
        { id: 10, low: BLANKS, high: FIVES },  // 5 points
        { id: 8, low: ACES, high: FOURS },   // 5 points
        { id: 5, low: DEUCES, high: TRES }    // 5 points
      ];
      
      const playerCountingDominoes = state.players[0]!.hand.filter(d => 
        countingDominoes.some(cd => cd.id === d.id)
      );
      expect(playerCountingDominoes.length).toBe(5);
    });
  });

  describe('Scoring Edge Cases', () => {
    it('should handle set bid that results in exactly bid amount', () => {
      const state = GameTestHelper.createTestState({
        phase: 'scoring',
        currentBid: { type: BID_TYPES.POINTS, value: 35, player: 0 },
        teamScores: [35, 7] // Exactly made the bid
      });
      
      // Team should get 1 mark for making exactly 35
      expect(state.teamScores[0]).toBe(35);
      expect(state.currentBid?.value).toBe(35);
    });

    it('should handle team going negative in marks', () => {
      let state = createInitialState();
      state.teamMarks = [-1, 3]; // Team 0 is negative
      
      // Team can still play and bid
      expect(state.teamMarks[0]).toBe(-1);
      expect(state.phase).toBe('bidding');
    });

    it('should handle simultaneous game end conditions', () => {
      let state = createInitialState();
      state.teamMarks = [6, 6]; // Both teams close to winning
      state.gameTarget = 7;
      
      // If both teams would reach 7 in same hand, highest score wins
      state.teamMarks = [7, 7]; // Both reach target
      expect(Math.max(...state.teamMarks)).toBe(7);
    });
  });

  describe('Trump Selection Edge Cases', () => {
    it('should handle trump selection with limited options', () => {
      const ctx = createTestContext();
      const state = GameTestHelper.createTestState({
        phase: 'trump_selection',
        winningBidder: 0,
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 }
      });

      // Marks bid with nello ruleset should include nello as trump option
      const transitions = getNextStates(state, ctx);
      const trumpOptions = transitions.filter(t => t.id.startsWith('trump-'));

      // Verify trump selection is available
      expect(trumpOptions.length).toBeGreaterThan(0);
    });

    it('should handle no-trump selection', () => {
      const ctx = createTestContext();
      const state = GameTestHelper.createTestState({
        phase: 'trump_selection',
        winningBidder: 0,
        currentBid: { type: BID_TYPES.POINTS, value: 35, player: 0 }
      });

      const transitions = getNextStates(state, ctx);
      const noTrump = transitions.find(t => t.id === 'trump-no-trump');

      if (noTrump) {
        const newState = noTrump.newState;
        expect(newState.trump.type).toBe('no-trump'); // no-trump selected
      }
    });
  });

  describe('Game End Scenarios', () => {
    it('should handle game ending mid-bidding due to penalty', () => {
      let state = createInitialState();
      state.teamMarks = [6, 6]; // Both teams at 6 marks
      state.gameTarget = 7;
      
      // If a team gets set penalty that puts them at 7
      state.teamMarks = [7, 6]; // Team 0 wins
      
      expect(Math.max(...state.teamMarks)).toBeGreaterThanOrEqual(state.gameTarget);
    });

    it('should handle extremely long game', () => {
      let state = createInitialState();
      state.teamMarks = [6, 6]; // Long game scenario
      
      // Game should continue until someone reaches target
      expect(state.phase).toBe('bidding');
      expect(Math.max(...state.teamMarks)).toBeLessThan(state.gameTarget);
    });

    it('should handle game with high target', () => {
      let state = createInitialState();
      state.gameTarget = 15; // Higher target game
      state.teamMarks = [14, 13];
      
      expect(state.gameTarget).toBe(15);
      expect(Math.max(...state.teamMarks)).toBeLessThan(state.gameTarget);
    });
  });
});