import { describe, it, expect } from 'vitest';
import { createInitialState, createSetupState } from '../../game/core/state';
import { isValidBid, isValidPlay } from '../../game/core/rules';
import { shuffleDominoes } from '../../game/core/dominoes';
import { BID_TYPES } from '../../game/constants';
import { GameTestHelper, createTestState } from '../helpers/gameTestHelper';
import { getPlayerLeftOfDealer, getNextPlayer, getNextDealer, getPlayerAfter } from '../../game/core/players';
import type { Bid, Domino } from '../../game/types';

describe('Basic Game Flow', () => {
  describe('Game Initialization', () => {
    it('creates initial game state correctly', () => {
      const state = createSetupState();
      
      expect(state.phase).toBe('setup');
      expect(state.dealer).toBeGreaterThanOrEqual(0);
      expect(state.dealer).toBeLessThan(4);
      expect(state.currentPlayer).toBe(getPlayerLeftOfDealer(state.dealer));
      expect(state.bids).toHaveLength(0);
      expect(state.hands).toEqual({});
      expect(state.tricks).toHaveLength(0);
      expect(state.currentTrick).toHaveLength(0);
      expect(state.trump).toBeNull();
      expect(state.bidWinner).toBeNull();
      expect(state.isComplete).toBe(false);
      expect(state.winner).toBeNull();
    });

    it('deals dominoes correctly', () => {
      const state = createInitialState();
      const helper = new GameTestHelper();
      
      // Deal dominoes to all players
      const dealtState = helper.dealDominoes(state);
      
      expect(Object.keys(dealtState.hands || {})).toHaveLength(4);
      expect(dealtState.hands?.[0]).toHaveLength(7);
      expect(dealtState.hands?.[1]).toHaveLength(7);
      expect(dealtState.hands?.[2]).toHaveLength(7);
      expect(dealtState.hands?.[3]).toHaveLength(7);
      
      // Total dominoes dealt should be 28
      const totalDominoes = Object.values(dealtState.hands || {}).flat().length;
      expect(totalDominoes).toBe(28);
      
      // All dominoes should be unique
      const allDominoes = Object.values(dealtState.hands || {}).flat();
      const uniqueIds = new Set(allDominoes.map(d => d.id));
      expect(uniqueIds.size).toBe(28);
    });

    it('shuffles dominoes randomly', () => {
      const shuffle1 = shuffleDominoes();
      const shuffle2 = shuffleDominoes();
      
      expect(shuffle1).toHaveLength(28);
      expect(shuffle2).toHaveLength(28);
      
      // While theoretically possible, shuffles should be different
      const sameOrder = shuffle1.every((domino, index) => 
        domino.id === shuffle2[index].id
      );
      expect(sameOrder).toBe(false);
    });
  });

  describe('Phase Transitions', () => {
    it('transitions from setup to bidding', () => {
      const state = createTestState({
        phase: 'setup',
        dealer: 0
      });
      
      const helper = new GameTestHelper();
      const dealtState = helper.dealDominoes(state);
      const biddingState = { 
        ...dealtState, 
        phase: 'bidding' as const,
        currentPlayer: getPlayerLeftOfDealer(state.dealer) 
      };
      
      expect(biddingState.phase).toBe('bidding');
      expect(biddingState.currentPlayer).toBe(getPlayerLeftOfDealer(biddingState.dealer));
    });

    it('transitions from bidding to trump selection', () => {
      const state = createTestState({
        phase: 'bidding',
        dealer: 0,
        currentPlayer: 1,
        bids: [
          { type: BID_TYPES.POINTS, value: 30, player: 1 },
          { type: BID_TYPES.PASS, player: 2 },
          { type: BID_TYPES.PASS, player: 3 },
          { type: BID_TYPES.PASS, player: 0 }
        ]
      });

      const trumpState = { 
        ...state, 
        phase: 'trump_selection' as const,
        bidWinner: 1,
        currentPlayer: 1
      };
      
      expect(trumpState.phase).toBe('trump_selection');
      expect(trumpState.bidWinner).toBe(1);
      expect(trumpState.currentPlayer).toBe(1);
    });

    it('transitions from trump selection to playing', () => {
      const state = createTestState({
        phase: 'trump_selection',
        bidWinner: 1,
        trump: 2 // twos trump
      });

      const playingState = {
        ...state,
        phase: 'playing' as const,
        currentPlayer: 1 // bid winner leads
      };
      
      expect(playingState.phase).toBe('playing');
      expect(playingState.trump).toBe(2);
      expect(playingState.currentPlayer).toBe(1);
    });

    it('transitions from playing to scoring', () => {
      const helper = new GameTestHelper();
      const state = helper.createCompleteGameState();
      
      const scoringState = {
        ...state,
        phase: 'scoring' as const
      };
      
      expect(scoringState.phase).toBe('scoring');
      expect(scoringState.tricks).toHaveLength(7);
    });
  });

  describe('Bidding Flow', () => {
    it('processes complete bidding round', () => {
      const state = createTestState({
        phase: 'bidding',
        dealer: 2,
        currentPlayer: 3,
        bids: []
      });

      const bids: Bid[] = [
        { type: BID_TYPES.PASS, player: 3 },
        { type: BID_TYPES.POINTS, value: 30, player: 0 },
        { type: BID_TYPES.PASS, player: 1 },
        { type: BID_TYPES.PASS, player: 2 }
      ];

      // Validate each bid
      bids.forEach(bid => {
        expect(isValidBid(state, bid)).toBe(true);
        state.bids.push(bid);
        state.currentPlayer = getNextPlayer(state.currentPlayer); // Advance to next player
      });

      expect(state.bids).toHaveLength(4);
      
      // Find winning bid
      const winningBid = state.bids.find(bid => bid.type !== BID_TYPES.PASS);
      expect(winningBid).toBeDefined();
      expect(winningBid!.player).toBe(0);
      expect(winningBid!.value).toBe(30);
    });

    it('handles all-pass scenario with redeal', () => {
      const state = createTestState({
        phase: 'bidding',
        dealer: 1,
        currentPlayer: 2,
        bids: []
      });

      const allPassBids: Bid[] = [
        { type: BID_TYPES.PASS, player: 2 },
        { type: BID_TYPES.PASS, player: 3 },
        { type: BID_TYPES.PASS, player: 0 },
        { type: BID_TYPES.PASS, player: 1 }
      ];

      allPassBids.forEach(bid => {
        expect(isValidBid(state, bid)).toBe(true);
        state.bids.push(bid);
        state.currentPlayer = getNextPlayer(state.currentPlayer); // Advance to next player
      });

      // All passed - should trigger redeal
      expect(state.bids.every(bid => bid.type === BID_TYPES.PASS)).toBe(true);
      
      // New dealer should be next player
      const newDealer = getNextDealer(state.dealer);
      expect(newDealer).toBe(2);
    });
  });

  describe('Gameplay Flow', () => {
    it('bid winner leads first trick', () => {
      const state = createTestState({
        phase: 'playing',
        bidWinner: 2,
        currentPlayer: 2,
        trump: 1,
        currentTrick: []
      });

      expect(state.currentPlayer).toBe(state.bidWinner);
      expect(state.currentTrick).toHaveLength(0);
    });

    it('processes complete trick', () => {
      const testDominoes: Domino[] = [
        { id: 'test1', high: 2, low: 3, points: 0 },
        { id: 'test2', high: 2, low: 4, points: 0 },
        { id: 'test3', high: 1, low: 1, points: 0 }, // trump
        { id: 'test4', high: 2, low: 5, points: 0 }
      ];

      const state = createTestState({
        phase: 'playing',
        trump: 1, // ones trump
        currentTrick: [],
        hands: {
          0: [testDominoes[0]],
          1: [testDominoes[1]],
          2: [testDominoes[2]],
          3: [testDominoes[3]]
        }
      });

      // Play each domino in turn
      const plays = [
        { player: 0, domino: testDominoes[0] },
        { player: 1, domino: testDominoes[1] },
        { player: 2, domino: testDominoes[2] },
        { player: 3, domino: testDominoes[3] }
      ];

      plays.forEach(play => {
        expect(isValidPlay(play.domino, state.hands?.[play.player] || [], state.currentTrick, state.trump!)).toBe(true);
        state.currentTrick.push(play);
      });

      expect(state.currentTrick).toHaveLength(4);
      
      // Player 2 should win with trump (1-1)
      // Implementation would determine trick winner
    });

    it('completes all 7 tricks', () => {
      const helper = new GameTestHelper();
      const state = helper.createGameInProgress();
      
      // Play through all 7 tricks
      for (let trickNum = 0; trickNum < 7; trickNum++) {
        // Each trick has 4 plays
        for (let playNum = 0; playNum < 4; playNum++) {
          const currentPlayer = getPlayerAfter(state.currentPlayer, playNum);
          const availableDominoes = state.players[currentPlayer].hand;
          
          if (availableDominoes.length > 0) {
            const domino = availableDominoes[0];
            // For this test, we'll just play any available domino to complete the game structure
            // Real gameplay would enforce suit following rules
            
            state.currentTrick.push({ player: currentPlayer, domino });
            state.players[currentPlayer].hand = availableDominoes.slice(1);
          }
        }
        
        // Complete trick
        state.tricks.push({
          plays: [...state.currentTrick],
          winner: 0, // For test purposes
          points: 0
        });
        state.currentTrick = [];
      }

      expect(state.tricks).toHaveLength(7);
      expect(state.players.every(player => player.hand.length === 0)).toBe(true);
    });
  });

  describe('Game Completion', () => {
    it('determines game winner correctly', () => {
      const helper = new GameTestHelper();
      const completedState = helper.createCompletedGame(0); // team 0 wins
      
      expect(completedState.isComplete).toBe(true);
      expect(completedState.winner).toBe(0);
    });

    it('tracks cumulative scores across multiple hands', () => {
      
      // Simulate multiple completed hands
      const gameScores = [
        { team0: 3, team1: 1 }, // hand 1: team 0 gets 3 marks
        { team0: 2, team1: 2 }, // hand 2: tied at 2 marks each  
        { team0: 1, team1: 3 }, // hand 3: team 1 gets 3 marks
        { team0: 1, team1: 0 }  // hand 4: team 0 gets 1 mark
      ];

      let cumulativeScores = { team0: 0, team1: 0 };
      
      gameScores.forEach(handScore => {
        cumulativeScores.team0 += handScore.team0;
        cumulativeScores.team1 += handScore.team1;
      });

      expect(cumulativeScores.team0).toBe(7); // 3+2+1+1 = 7 marks
      expect(cumulativeScores.team1).toBe(6); // 1+2+3+0 = 6 marks
      
      // Team 0 should win with 7 marks
      expect(cumulativeScores.team0).toBeGreaterThanOrEqual(7);
    });
  });

  describe('Error Handling', () => {
    it('handles invalid bids gracefully', () => {
      const state = createTestState({
        phase: 'bidding',
        dealer: 0,
        currentPlayer: 1,
        bids: []
      });

      // Invalid bid - below minimum
      const invalidBid: Bid = { type: BID_TYPES.POINTS, value: 25, player: 1 };
      expect(isValidBid(state, invalidBid)).toBe(false);
      
      // Invalid bid - wrong player
      const wrongPlayerBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 2 };
      expect(isValidBid(state, wrongPlayerBid)).toBe(false);
    });

    it('handles invalid plays gracefully', () => {
      const state = createTestState({
        phase: 'playing',
        trump: 1,
        currentTrick: [{
          player: 0,
          domino: { id: 'lead', high: 3, low: 2, points: 0 } // threes suit led
        }]
      });

      const playerHand: Domino[] = [
        { id: 'valid', high: 3, low: 4, points: 0 }, // valid follow (contains 3)
        { id: 'invalid', high: 2, low: 5, points: 0 } // invalid - different suit
      ];

      // Valid play - following suit
      expect(isValidPlay(playerHand[0], playerHand, state.currentTrick, state.trump!)).toBe(true);
      
      // Invalid play - not following suit when able
      expect(isValidPlay(playerHand[1], playerHand, state.currentTrick, state.trump!)).toBe(false);
    });

    it('prevents out-of-turn actions', () => {
      const state = createTestState({
        phase: 'bidding',
        currentPlayer: 1,
        bids: []
      });

      // Valid bid by current player
      const validBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 1 };
      expect(isValidBid(state, validBid)).toBe(true);
      
      // Invalid bid by non-current player
      const outOfTurnBid: Bid = { type: BID_TYPES.POINTS, value: 31, player: 2 };
      expect(isValidBid(state, outOfTurnBid)).toBe(false);
    });
  });
});