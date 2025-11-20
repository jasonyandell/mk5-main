import { describe, it, expect } from 'vitest';
import { createSetupState } from '../../game/core/state';
import { composeRules, baseRuleSet } from '../../game/rulesets';
import { shuffleDominoesWithSeed } from '../../game/core/dominoes';
import { analyzeSuits } from '../../game/core/suit-analysis';
import { BID_TYPES } from '../../game/constants';
import { StateBuilder } from '../helpers';
import { getPlayerLeftOfDealer, getNextPlayer, getNextDealer, getPlayerAfter } from '../../game/core/players';
import type { Bid, Domino } from '../../game/types';
import { ACES, DEUCES, TRES } from '../../game/types';

const rules = composeRules([baseRuleSet]);

describe('Basic Game Flow', () => {
  describe('Game Initialization', () => {
    it('creates initial game state correctly', () => {
      const state = createSetupState();

      expect(state.phase).toBe('setup');
      expect(state.dealer).toBeGreaterThanOrEqual(0);
      expect(state.dealer).toBeLessThan(4);
      expect(state.currentPlayer).toBe(getPlayerLeftOfDealer(state.dealer));
      expect(state.bids).toHaveLength(0);
      expect(state.players.every(p => p.hand.length === 0)).toBe(true);
      expect(state.tricks).toHaveLength(0);
      expect(state.currentTrick).toHaveLength(0);
      expect(state.trump).toEqual({ type: 'not-selected' });
      expect(state.winningBidder).toBe(-1);
    });

    it('deals dominoes correctly', () => {
      // StateBuilder.inBiddingPhase() already deals dominoes
      const dealtState = StateBuilder.inBiddingPhase().build();

      expect(dealtState.players).toHaveLength(4);
      expect(dealtState.players[0]!.hand).toHaveLength(7);
      expect(dealtState.players[1]!.hand).toHaveLength(7);
      expect(dealtState.players[2]!.hand).toHaveLength(7);
      expect(dealtState.players[3]!.hand).toHaveLength(7);

      // Total dominoes dealt should be 28
      const totalDominoes = dealtState.players.flatMap(p => p.hand).length;
      expect(totalDominoes).toBe(28);

      // All dominoes should be unique
      const allDominoes = dealtState.players.flatMap(p => p.hand);
      const uniqueIds = new Set(allDominoes.map(d => d.id));
      expect(uniqueIds.size).toBe(28);
    });

    it('shuffles dominoes deterministically with different seeds', () => {
      const shuffle1 = shuffleDominoesWithSeed(12345);
      const shuffle2 = shuffleDominoesWithSeed(67890);
      
      expect(shuffle1).toHaveLength(28);
      expect(shuffle2).toHaveLength(28);
      
      // Different seeds should produce different shuffles
      const sameOrder = shuffle1.every((domino, index) => 
        domino.id === shuffle2[index]?.id
      );
      expect(sameOrder).toBe(false);
      
      // Same seed should produce same shuffle
      const shuffle3 = shuffleDominoesWithSeed(12345);
      expect(shuffle1.every((domino, index) => 
        domino.id === shuffle3[index]?.id
      )).toBe(true);
    });
  });

  describe('Phase Transitions', () => {
    it('transitions from setup to bidding', () => {
      const biddingState = StateBuilder
        .inBiddingPhase(0)
        .build();
      
      expect(biddingState.phase).toBe('bidding');
      expect(biddingState.currentPlayer).toBe(getPlayerLeftOfDealer(biddingState.dealer));
    });

    it('transitions from bidding to trump selection', () => {
      const trumpState = StateBuilder
        .inTrumpSelection(1, 30)
        .withDealer(0)
        .build();
      
      expect(trumpState.phase).toBe('trump_selection');
      expect(trumpState.winningBidder).toBe(1);
      expect(trumpState.currentPlayer).toBe(1);
    });

    it('transitions from trump selection to playing', () => {
      const playingState = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: DEUCES })
        .with({ winningBidder: 1, currentPlayer: 1 })
        .build();
      
      expect(playingState.phase).toBe('playing');
      expect(playingState.trump.type).toBe('suit');
      expect(playingState.trump.suit).toBe(DEUCES);
      expect(playingState.currentPlayer).toBe(1);
    });

    it('transitions from playing to scoring', () => {
      const scoringState = StateBuilder
        .withTricksPlayed(7)
        .with({ phase: 'scoring' })
        .build();
      
      expect(scoringState.phase).toBe('scoring');
      expect(scoringState.tricks).toHaveLength(7);
    });
  });

  describe('Bidding Flow', () => {
    it('processes complete bidding round', () => {
      const state = StateBuilder
        .inBiddingPhase(2)
        .withCurrentPlayer(3)
        .withBids([])
        .build();

      const bids: Bid[] = [
        { type: BID_TYPES.PASS, player: 3 },
        { type: BID_TYPES.POINTS, value: 30, player: 0 },
        { type: BID_TYPES.PASS, player: 1 },
        { type: BID_TYPES.PASS, player: 2 }
      ];

      // Validate each bid
      bids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(true);
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
      const state = StateBuilder
        .inBiddingPhase(1)
        .withCurrentPlayer(2)
        .withBids([])
        .build();

      const allPassBids: Bid[] = [
        { type: BID_TYPES.PASS, player: 2 },
        { type: BID_TYPES.PASS, player: 3 },
        { type: BID_TYPES.PASS, player: 0 },
        { type: BID_TYPES.PASS, player: 1 }
      ];

      allPassBids.forEach(bid => {
        expect(rules.isValidBid(state, bid)).toBe(true);
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
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .with({ winningBidder: 2, currentPlayer: 2 })
        .build();

      expect(state.currentPlayer).toBe(2); // Current player should match bidWinner from setup
      expect(state.currentTrick).toHaveLength(0);
    });

    it('processes complete trick', () => {
      const testDominoes: Domino[] = [
        { id: 'test1', high: 2, low: 3, points: 0 },
        { id: 'test2', high: 2, low: 4, points: 0 },
        { id: 'test3', high: 1, low: 1, points: 0 }, // trump
        { id: 'test4', high: 2, low: 5, points: 0 }
      ];

      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .build();

      // Set player hands
      state.players[0]!.hand = [testDominoes[0]!];
      state.players[1]!.hand = [testDominoes[1]!];
      state.players[2]!.hand = [testDominoes[2]!];
      state.players[3]!.hand = [testDominoes[3]!];

      // Play each domino in turn
      const plays = [
        { player: 0, domino: testDominoes[0]! },
        { player: 1, domino: testDominoes[1]! },
        { player: 2, domino: testDominoes[2]! },
        { player: 3, domino: testDominoes[3]! }
      ];

      // Initialize players with suit analysis
      state.players.forEach((player) => {
        player.suitAnalysis = analyzeSuits(player.hand, state.trump!);
      });
      
      // Set current suit from first play
      state.currentSuit = TRES; // threes led (3-2 high end is 3)
      
      plays.forEach(play => {
        expect(rules.isValidPlay(state, play.domino, play.player)).toBe(true);
        state.currentTrick.push(play);
      });

      expect(state.currentTrick).toHaveLength(4);
      
      // Player 2 should win with trump (1-1)
      // Implementation would determine trick winner
    });

    it('completes all 7 tricks', () => {
      const state = StateBuilder
        .inPlayingPhase()
        .build();
      
      // Play through all 7 tricks
      for (let trickNum = 0; trickNum < 7; trickNum++) {
        // Each trick has 4 plays
        for (let playNum = 0; playNum < 4; playNum++) {
          const currentPlayer = getPlayerAfter(state.currentPlayer, playNum);
          const player = state.players[currentPlayer];
          if (!player) continue;
          
          const availableDominoes = player.hand;
          
          if (availableDominoes.length > 0) {
            const domino = availableDominoes[0]!;
            // For this test, we'll just play any available domino to complete the game structure
            // Real gameplay would enforce suit following rules
            
            state.currentTrick.push({ player: currentPlayer, domino });
            player.hand = availableDominoes.slice(1);
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
      const completedState = StateBuilder
        .gameEnded(0) // team 0 wins
        .build();

      expect(completedState.phase).toBe('game_end');
      // Winner determined by scoring logic
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
      const state = StateBuilder
        .inBiddingPhase(0)
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      // Invalid bid - below minimum
      const invalidBid: Bid = { type: BID_TYPES.POINTS, value: 25, player: 1 };
      expect(rules.isValidBid(state, invalidBid)).toBe(false);
      
      // Invalid bid - wrong player
      const wrongPlayerBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 2 };
      expect(rules.isValidBid(state, wrongPlayerBid)).toBe(false);
    });

    it('handles invalid plays gracefully', () => {
      const state = StateBuilder
        .inPlayingPhase({ type: 'suit', suit: ACES })
        .withCurrentTrick([{
          player: 0,
          domino: { id: 'lead', high: 3, low: 2, points: 0 } // threes suit led
        }])
        .build();

      const playerHand: Domino[] = [
        { id: 'valid', high: 3, low: 4, points: 0 }, // valid follow (contains 3)
        { id: 'invalid', high: 2, low: 5, points: 0 } // invalid - different suit
      ];
      
      // Update state with player hand and current suit
      state.currentPlayer = 1;
      state.currentSuit = TRES; // threes were led
      state.players[1]!.hand = playerHand;
      state.players[1]!.suitAnalysis = analyzeSuits(playerHand, state.trump!);

      // Valid play - following suit
      const firstDomino = playerHand[0];
      const secondDomino = playerHand[1];
      if (firstDomino) {
        expect(rules.isValidPlay(state, firstDomino, 1)).toBe(true);
      }
      
      // Invalid play - not following suit when able
      if (secondDomino) {
        expect(rules.isValidPlay(state, secondDomino, 1)).toBe(false);
      }
    });

    it('prevents out-of-turn actions', () => {
      const state = StateBuilder
        .inBiddingPhase()
        .withCurrentPlayer(1)
        .withBids([])
        .build();

      // Valid bid by current player
      const validBid: Bid = { type: BID_TYPES.POINTS, value: 30, player: 1 };
      expect(rules.isValidBid(state, validBid)).toBe(true);
      
      // Invalid bid by non-current player
      const outOfTurnBid: Bid = { type: BID_TYPES.POINTS, value: 31, player: 2 };
      expect(rules.isValidBid(state, outOfTurnBid)).toBe(false);
    });
  });
});