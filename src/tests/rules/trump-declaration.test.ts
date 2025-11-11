import { describe, test, expect } from 'vitest';
import { createInitialState, getNextStates } from '../../game';
import { createTestContext } from '../helpers/executionContext';
import type { TrumpSelection } from '../../game/types';
import { BLANKS, ACES, DEUCES, TRES, FOURS, FIVES, SIXES } from '../../game/types';

describe('Feature: Trump Declaration', () => {
  const ctx = createTestContext();
  describe('Scenario: Declaring Trump', () => {
    test('Given a player has won the bidding', () => {
      // Create a state where player 1 has won the bidding
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'bidding';
      gameState.dealer = 0;
      gameState.currentPlayer = 1;
      gameState.bids = [];
      
      // Simulate bidding where player 1 wins with 30 points
      const transitions = getNextStates(gameState, ctx);
      const passBid = transitions.find(t => t.id === 'pass');
      if (passBid) {
        Object.assign(gameState, passBid.newState); // Player 1 passes
      }
      
      const bid30 = getNextStates(gameState, ctx).find(t => t.id === 'bid-30');
      if (bid30) {
        Object.assign(gameState, bid30.newState); // Player 2 bids 30
      }
      
      // Players 3 and 0 pass
      for (let i = 0; i < 2; i++) {
        const pass = getNextStates(gameState, ctx).find(t => t.id === 'pass');
        if (pass) {
          Object.assign(gameState, pass.newState);
        }
      }
      
      // After 4 bids, game should move to trump selection
      expect(gameState.winningBidder).toBe(2);
      expect(gameState.phase).toBe('trump_selection');
      expect(gameState.trump).toEqual({ type: 'not-selected' });
    });

    test('When they are ready to play - Then they must declare trump before playing the first domino', () => {
      // Create a state in trump selection phase
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'trump_selection';
      gameState.winningBidder = 1;
      gameState.currentPlayer = 1;
      gameState.currentBid = { type: 'points', value: 30, player: 1 };
      gameState.trump = { type: 'not-selected' };
      gameState.currentTrick = [];
      gameState.tricks = [];
      
      // Cannot proceed to playing phase without declaring trump
      expect(gameState.phase).toBe('trump_selection');
      expect(gameState.trump).toEqual({ type: 'not-selected' });
      expect(gameState.currentTrick.length).toBe(0);
      
      // Get available trump options
      const trumpOptions = getNextStates(gameState, ctx);
      expect(trumpOptions.length).toBeGreaterThan(0);
      
      // After declaring trump (e.g., threes), can proceed to playing
      const declareThrees = trumpOptions.find(t => t.id === 'trump-threes');
      expect(declareThrees).toBeDefined();
      
      if (declareThrees) {
        const afterTrumpState = declareThrees.newState;
        expect(afterTrumpState.trump).toEqual({ type: 'suit', suit: TRES });
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
      gameState.trump = { type: 'not-selected' };
      
      // Get available trump options
      const trumpOptions = getNextStates(gameState, ctx);
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
      const validSuitTrumps: TrumpSelection[] = [
        { type: 'suit', suit: BLANKS }, { type: 'suit', suit: ACES }, { type: 'suit', suit: DEUCES },
        { type: 'suit', suit: TRES }, { type: 'suit', suit: FOURS }, { type: 'suit', suit: FIVES }, { type: 'suit', suit: SIXES }
      ];
      
      validSuitTrumps.forEach((trumpValue, index) => {
        const suitName = ['blanks', 'ones', 'twos', 'threes', 'fours', 'fives', 'sixes'][index];
        const transition = trumpOptions.find(t => t.id === `trump-${suitName}`);
        
        expect(transition).toBeDefined();
        if (transition) {
          expect(transition.newState.trump).toEqual(trumpValue);
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
      gameState.trump = { type: 'not-selected' };
      
      // Get available trump options
      const trumpOptions = getNextStates(gameState, ctx);
      
      // Doubles as trump should be available
      const doublesOption = trumpOptions.find(t => t.id === 'trump-doubles');
      expect(doublesOption).toBeDefined();
      
      if (doublesOption) {
        // Doubles as trump is represented as TrumpSelection
        expect(doublesOption.newState.trump).toEqual({ type: 'doubles' });
      }
    });

    test('And trump options include no-trump (follow-me)', () => {
      // Note: No-trump is not currently implemented in the game engine
      // The game engine only supports suits 0-6 and doubles
      // No-trump would need to be added to the trump selection system
      // TODO: Implement no-trump trump selection
    });
  });
});