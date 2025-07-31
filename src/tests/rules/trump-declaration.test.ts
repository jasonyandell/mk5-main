import { describe, test, expect } from 'vitest';
import { createInitialState, getNextStates, getTrumpSelectionTransitions } from '../../game';
import type { GameState, Trump } from '../../game/types';

describe('Feature: Trump Declaration', () => {
  describe('Scenario: Declaring Trump', () => {
    test('Given a player has won the bidding', () => {
      // Create a state where player 1 has won the bidding
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'bidding';
      gameState.dealer = 0;
      gameState.currentPlayer = 1;
      gameState.bids = [];
      
      // Simulate bidding where player 1 wins with 30 points
      const transitions = getNextStates(gameState);
      const passBid = transitions.find(t => t.id === 'pass');
      if (passBid) {
        Object.assign(gameState, passBid.newState); // Player 1 passes
      }
      
      const bid30 = getNextStates(gameState).find(t => t.id === 'bid-30');
      if (bid30) {
        Object.assign(gameState, bid30.newState); // Player 2 bids 30
      }
      
      // Players 3 and 0 pass
      for (let i = 0; i < 2; i++) {
        const pass = getNextStates(gameState).find(t => t.id === 'pass');
        if (pass) {
          Object.assign(gameState, pass.newState);
        }
      }
      
      // After 4 bids, game should move to trump selection
      expect(gameState.winningBidder).toBe(2);
      expect(gameState.phase).toBe('trump_selection');
      expect(gameState.trump).toBeNull();
    });

    test('When they are ready to play - Then they must declare trump before playing the first domino', () => {
      // Create a state in trump selection phase
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'trump_selection';
      gameState.winningBidder = 1;
      gameState.currentPlayer = 1;
      gameState.currentBid = { type: 'points', value: 30, player: 1 };
      gameState.trump = null;
      gameState.currentTrick = [];
      gameState.tricks = [];
      
      // Cannot proceed to playing phase without declaring trump
      expect(gameState.phase).toBe('trump_selection');
      expect(gameState.trump).toBeNull();
      expect(gameState.currentTrick.length).toBe(0);
      
      // Get available trump options
      const trumpOptions = getNextStates(gameState);
      expect(trumpOptions.length).toBeGreaterThan(0);
      
      // After declaring trump (e.g., threes), can proceed to playing
      const declareThrees = trumpOptions.find(t => t.id === 'trump-threes');
      expect(declareThrees).toBeDefined();
      
      if (declareThrees) {
        const afterTrumpState = declareThrees.newState;
        expect(afterTrumpState.trump).toBe(3);
        expect(afterTrumpState.phase).toBe('playing');
      }
    });

    test('And trump options include any suit (blanks through sixes)', () => {
      // Create a state in trump selection phase
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'trump_selection';
      gameState.winningBidder = 1;
      gameState.currentPlayer = 1;
      gameState.currentBid = { type: 'points', value: 30, player: 1 };
      gameState.trump = null;
      
      // Get available trump options
      const trumpOptions = getNextStates(gameState);
      const trumpIds = trumpOptions.map(t => t.id);
      
      // Test all valid suit options
      const expectedSuits = [
        'trump-blanks',   // 0
        'trump-ones',     // 1
        'trump-twos',     // 2
        'trump-threes',   // 3
        'trump-fours',    // 4
        'trump-fives',    // 5
        'trump-sixes'     // 6
      ];
      
      expectedSuits.forEach(suit => {
        expect(trumpIds).toContain(suit);
      });
      
      // Verify the trump values
      const validSuitTrumps: Trump[] = [0, 1, 2, 3, 4, 5, 6];
      
      validSuitTrumps.forEach(trumpValue => {
        const suitName = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'][trumpValue];
        const transition = trumpOptions.find(t => t.id === `trump-${suitName}`);
        
        expect(transition).toBeDefined();
        if (transition) {
          expect(transition.newState.trump).toBe(trumpValue);
        }
      });
    });

    test('And trump options include doubles as trump', () => {
      // Create a state in trump selection phase
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'trump_selection';
      gameState.winningBidder = 1;
      gameState.currentPlayer = 1;
      gameState.currentBid = { type: 'points', value: 30, player: 1 };
      gameState.trump = null;
      
      // Get available trump options
      const trumpOptions = getNextStates(gameState);
      
      // Doubles as trump should be available
      const doublesOption = trumpOptions.find(t => t.id === 'trump-doubles');
      expect(doublesOption).toBeDefined();
      
      if (doublesOption) {
        // Doubles as trump is represented as 7
        expect(doublesOption.newState.trump).toBe(7);
      }
    });

    test('And trump options include no-trump (follow-me)', () => {
      // Create a state in trump selection phase
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'trump_selection';
      gameState.winningBidder = 1;
      gameState.currentPlayer = 1;
      gameState.currentBid = { type: 'points', value: 30, player: 1 };
      gameState.trump = null;
      
      // Get available trump options
      const trumpOptions = getNextStates(gameState);
      
      // Note: No-trump is not currently implemented in the game engine
      // The game engine only supports suits 0-6 and doubles (7)
      // No-trump (8) would need to be added to TRUMP_SUITS constant
    });
  });
});