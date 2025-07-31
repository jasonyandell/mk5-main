import { describe, it, expect } from 'vitest';
import type { Bid, GameState } from '../../../game/types';
import { isValidBid, createInitialState, GAME_CONSTANTS, createDominoes, dealDominoes, shuffleDominoes } from '../../../game';

describe('Feature: Standard Bidding - Valid Point Bids', () => {
  describe('Scenario: Valid Point Bids', () => {
    it('should allow valid point bids from 30 to 41', () => {
      // Given it is a player's turn to bid
      const gameState = createInitialState();
      gameState.phase = 'bidding';
      gameState.currentPlayer = 0;
      
      // Deal dominoes to players
      const dominoes = createDominoes();
      const hands = dealDominoes(shuffleDominoes(dominoes, 12345));
      gameState.players.forEach((player, i) => {
        player.hand = hands[i];
      });

      // When they make a point bid
      const validPointBids = [30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41];
      
      // Then valid bids are 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, or 41 points
      validPointBids.forEach(bidValue => {
        const bid: Bid = {
          type: 'points',
          value: bidValue,
          player: 0
        };
        
        // Verify bid is valid - use legacy signature: isValidBid(bid, currentBid, state)
        const isValid = isValidBid(bid, null, gameState);
        expect(isValid).toBe(true);
        expect(bid.value).toBeGreaterThanOrEqual(GAME_CONSTANTS.MIN_BID);
        expect(bid.value).toBeLessThanOrEqual(GAME_CONSTANTS.MAX_BID);
      });
    });

    it('should reject point bids outside valid range', () => {
      // Given it is a player's turn to bid
      const gameState = createInitialState();
      gameState.phase = 'bidding';
      gameState.currentPlayer = 0;
      
      // Deal dominoes to players
      const dominoes = createDominoes();
      const hands = dealDominoes(shuffleDominoes(dominoes, 12345));
      gameState.players.forEach((player, i) => {
        player.hand = hands[i];
      });
      
      // Test invalid bids are rejected
      const invalidPointBids = [29, 42, 43, 50];
      
      invalidPointBids.forEach(bidValue => {
        const bid: Bid = {
          type: 'points',
          value: bidValue,
          player: 0
        };
        
        const isValid = isValidBid(bid, null, gameState);
        expect(isValid).toBe(false);
      });
    });
  });
});