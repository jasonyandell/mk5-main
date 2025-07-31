import { describe, it, expect } from 'vitest';
import { createInitialState, getNextStates, getPlayerLeftOfDealer } from '../../game';
import type { GameState } from '../../game/types';

describe('Feature: Standard Bidding', () => {
  describe('Scenario: Bidding Order', () => {
    it('Given a new hand has been dealt', () => {
      // Test setup - create initial state with specific dealer
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'bidding';
      gameState.dealer = 2; // Player 2 is the dealer/shaker
      gameState.currentPlayer = getPlayerLeftOfDealer(2); // Should be player 3
      gameState.bids = [];

      expect(gameState.phase).toBe('bidding');
      expect(gameState.bids).toEqual([]);
      expect(gameState.dealer).toBe(2);
      expect(gameState.currentPlayer).toBe(3);
    });

    it('When bidding begins, Then the player to the left of the shaker bids first', () => {
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'bidding';
      gameState.dealer = 2; // Player 2 is the shaker
      gameState.currentPlayer = getPlayerLeftOfDealer(gameState.dealer);

      // Verify player to left of shaker (dealer) bids first
      const playerToLeftOfShaker = getPlayerLeftOfDealer(gameState.dealer);
      expect(gameState.currentPlayer).toBe(playerToLeftOfShaker);
      expect(gameState.currentPlayer).toBe(3);
    });

    it('And bidding proceeds clockwise', () => {
      // Test bidding order progression using actual game engine
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'bidding';
      gameState.dealer = 2;
      gameState.currentPlayer = getPlayerLeftOfDealer(gameState.dealer); // Player 3
      gameState.bids = [];

      const expectedOrder = [3, 0, 1, 2]; // Starting from player 3 (left of dealer 2)
      const actualOrder: number[] = [];

      // Simulate bidding by having each player pass
      for (let i = 0; i < 4; i++) {
        actualOrder.push(gameState.currentPlayer);
        
        // Get available actions and choose pass
        const transitions = getNextStates(gameState);
        const passTransition = transitions.find(t => t.id === 'pass');
        expect(passTransition).toBeDefined();
        
        if (passTransition && i < 3) {
          // Apply the pass transition
          gameState.bids = passTransition.newState.bids;
          gameState.currentPlayer = passTransition.newState.currentPlayer;
        }
      }

      expect(actualOrder).toEqual(expectedOrder);
    });

    it('And each player gets exactly one opportunity to bid or pass', () => {
      // Test that each player bids exactly once using the game engine
      const gameState = createInitialState({ shuffleSeed: 12345 });
      gameState.phase = 'bidding';
      gameState.dealer = 2;
      gameState.currentPlayer = getPlayerLeftOfDealer(gameState.dealer); // Player 3
      gameState.bids = [];

      // Track who has bid
      const playerBidCount = new Map<number, number>();

      // Simulate bidding: Player 3 passes, Player 0 bids 30, Player 1 passes, Player 2 bids 31
      const actions = ['pass', 'bid-30', 'pass', 'bid-31'];
      
      for (let i = 0; i < 4; i++) {
        const currentPlayer = gameState.currentPlayer;
        playerBidCount.set(currentPlayer, (playerBidCount.get(currentPlayer) || 0) + 1);
        
        const transitions = getNextStates(gameState);
        const chosenTransition = transitions.find(t => t.id === actions[i]);
        
        expect(chosenTransition).toBeDefined();
        
        if (chosenTransition) {
          // Apply the transition
          Object.assign(gameState, chosenTransition.newState);
        }
      }

      // Verify each player bid exactly once
      expect(playerBidCount.size).toBe(4);
      playerBidCount.forEach((count, player) => {
        expect(count).toBe(1);
      });
      
      // Verify we recorded the expected bids
      expect(gameState.bids).toHaveLength(4);
      expect(gameState.bids[0]).toMatchObject({ type: 'pass', player: 3 });
      expect(gameState.bids[1]).toMatchObject({ type: 'points', value: 30, player: 0 });
      expect(gameState.bids[2]).toMatchObject({ type: 'pass', player: 1 });
      expect(gameState.bids[3]).toMatchObject({ type: 'points', value: 31, player: 2 });
    });
  });
});