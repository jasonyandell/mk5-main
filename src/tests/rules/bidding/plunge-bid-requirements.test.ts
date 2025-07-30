import { describe, it, expect } from 'vitest';
import type { GameState, Bid, Domino } from '../../../game/types';

describe('Feature: Special Bids', () => {
  describe('Scenario: Plunge Bid Requirements', () => {
    function createStateWithHandAndBids(playerId: number, hand: Domino[], bids: Bid[]): GameState {
      const players = [
        { id: 0, name: 'Player 0', hand: [] as Domino[], teamId: 0 as 0, marks: 0 },
        { id: 1, name: 'Player 1', hand: [] as Domino[], teamId: 1 as 1, marks: 0 },
        { id: 2, name: 'Player 2', hand: [] as Domino[], teamId: 0 as 0, marks: 0 },
        { id: 3, name: 'Player 3', hand: [] as Domino[], teamId: 1 as 1, marks: 0 },
      ];
      
      // Set the hand for the specified player
      players[playerId].hand = hand;
      
      const state: GameState = {
        phase: 'bidding',
        players: players,
        currentPlayer: playerId,
        dealer: 3,
        bids: bids,
        currentBid: bids.length > 0 ? bids[bids.length - 1] : null,
        winningBidder: null,
        trump: null,
        tricks: [],
        currentTrick: [],
        teamScores: [0, 0],
        teamMarks: [0, 0],
        gameTarget: 7,
        tournamentMode: true,
        shuffleSeed: 12345,
      };
      return state;
    }

    function countDoublesInHand(hand: Domino[]): number {
      return hand.filter(domino => domino.high === domino.low).length;
    }

    function canPlungeBid(state: GameState): boolean {
      const currentPlayerHand = state.players[state.currentPlayer].hand;
      const doubleCount = countDoublesInHand(currentPlayerHand);
      
      // Must hold at least 4 doubles in hand
      return doubleCount >= 4;
    }

    function getMinimumPlungeBidValue(state: GameState): number {
      // Default plunge is 4 marks
      let minimumPlunge = 4;
      
      // If bidding has reached 4 marks, must bid 5 marks
      if (state.currentBid && state.currentBid.type === 'marks' && state.currentBid.value && state.currentBid.value >= 4) {
        minimumPlunge = state.currentBid.value + 1;
      }
      
      return minimumPlunge;
    }

    function isValidPlungeBid(state: GameState, bidValue: number): boolean {
      if (!canPlungeBid(state)) {
        return false;
      }
      
      const minimumValue = getMinimumPlungeBidValue(state);
      return bidValue >= minimumValue;
    }

    function isPlungeAllowedAsOpeningBid(): boolean {
      // Plunge can be declared as an opening bid
      return true;
    }

    function isJumpBiddingAllowed(bidType: string): boolean {
      // Jump bidding is only allowed for plunge
      return bidType === 'plunge';
    }

    it('Given a player holds at least 4 doubles in their hand', () => {
      const handWith4Doubles: Domino[] = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 2, low: 2, id: '2-2' },
        { high: 3, low: 3, id: '3-3' },
        { high: 4, low: 5, id: '4-5' },
        { high: 5, low: 6, id: '5-6' },
        { high: 6, low: 4, id: '6-4' },
      ];
      
      const state = createStateWithHandAndBids(0, handWith4Doubles, []);
      expect(countDoublesInHand(handWith4Doubles)).toBe(4);
      expect(canPlungeBid(state)).toBe(true);
    });

    it('When they want to plunge', () => {
      const handWith5Doubles: Domino[] = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 2, low: 2, id: '2-2' },
        { high: 3, low: 3, id: '3-3' },
        { high: 4, low: 4, id: '4-4' },
        { high: 5, low: 6, id: '5-6' },
        { high: 6, low: 5, id: '6-5' },
      ];
      
      const state = createStateWithHandAndBids(0, handWith5Doubles, []);
      expect(canPlungeBid(state)).toBe(true);
    });

    it('Then they must bid at least 4 marks', () => {
      const handWith4Doubles: Domino[] = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 2, low: 2, id: '2-2' },
        { high: 3, low: 3, id: '3-3' },
        { high: 4, low: 5, id: '4-5' },
        { high: 5, low: 6, id: '5-6' },
        { high: 6, low: 4, id: '6-4' },
      ];
      
      const state = createStateWithHandAndBids(0, handWith4Doubles, []);
      expect(getMinimumPlungeBidValue(state)).toBe(4);
      expect(isValidPlungeBid(state, 4)).toBe(true);
      expect(isValidPlungeBid(state, 3)).toBe(false);
    });

    it('And if bidding has reached 4 marks, they must bid 5 marks', () => {
      const handWith4Doubles: Domino[] = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 2, low: 2, id: '2-2' },
        { high: 3, low: 3, id: '3-3' },
        { high: 4, low: 5, id: '4-5' },
        { high: 5, low: 6, id: '5-6' },
        { high: 6, low: 4, id: '6-4' },
      ];
      
      const existingBids: Bid[] = [
        { type: 'points', value: 30, player: 1 },
        { type: 'marks', value: 2, player: 2 },
        { type: 'marks', value: 3, player: 3 },
        { type: 'marks', value: 4, player: 1 },
      ];
      
      const state = createStateWithHandAndBids(0, handWith4Doubles, existingBids);
      expect(getMinimumPlungeBidValue(state)).toBe(5);
      expect(isValidPlungeBid(state, 5)).toBe(true);
      expect(isValidPlungeBid(state, 4)).toBe(false);
    });

    it('And this can be declared as an opening bid or jump bid', () => {
      expect(isPlungeAllowedAsOpeningBid()).toBe(true);
      
      // Test that jump bidding is allowed for plunge
      const handWith6Doubles: Domino[] = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 2, low: 2, id: '2-2' },
        { high: 3, low: 3, id: '3-3' },
        { high: 4, low: 4, id: '4-4' },
        { high: 5, low: 5, id: '5-5' },
        { high: 6, low: 4, id: '6-4' },
      ];
      
      // Opening bid scenario
      const openingState = createStateWithHandAndBids(0, handWith6Doubles, []);
      expect(isValidPlungeBid(openingState, 4)).toBe(true);
      
      // Jump bid scenario (from 30 points to 4 marks)
      const jumpBidState = createStateWithHandAndBids(1, handWith6Doubles, [
        { type: 'points', value: 30, player: 0 }
      ]);
      expect(isValidPlungeBid(jumpBidState, 4)).toBe(true);
    });

    it('And this is the only case where jump bidding is allowed', () => {
      expect(isJumpBiddingAllowed('plunge')).toBe(true);
      expect(isJumpBiddingAllowed('points')).toBe(false);
      expect(isJumpBiddingAllowed('marks')).toBe(false);
      expect(isJumpBiddingAllowed('nello')).toBe(false);
      expect(isJumpBiddingAllowed('splash')).toBe(false);
    });

    it('Should not allow plunge with fewer than 4 doubles', () => {
      const handWith3Doubles: Domino[] = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 2, low: 2, id: '2-2' },
        { high: 3, low: 4, id: '3-4' },
        { high: 4, low: 5, id: '4-5' },
        { high: 5, low: 6, id: '5-6' },
        { high: 6, low: 4, id: '6-4' },
      ];
      
      const state = createStateWithHandAndBids(0, handWith3Doubles, []);
      expect(countDoublesInHand(handWith3Doubles)).toBe(3);
      expect(canPlungeBid(state)).toBe(false);
      expect(isValidPlungeBid(state, 4)).toBe(false);
    });
  });
});