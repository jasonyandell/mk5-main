import { describe, it, expect } from 'vitest';
import type { Bid } from '../../../game/types';
import { createInitialState, GAME_CONSTANTS, dealDominoesWithSeed } from '../../../game';
import { composeRules, baseRuleSet } from '../../../game/rulesets';

const rules = composeRules([baseRuleSet]);

describe('Feature: Standard Bidding - Valid Point Bids', () => {
  describe('Scenario: Valid Point Bids', () => {
    it('should allow valid point bids from 30 to 41', () => {
      // Given it is a player's turn to bid
      const gameState = createInitialState();
      gameState.phase = 'bidding';
      gameState.currentPlayer = 0;
      
      // Deal dominoes to players
      const hands = dealDominoesWithSeed(12345);
      gameState.players.forEach((player, i) => {
        const hand = hands[i];
        if (hand) {
          player.hand = hand;
        }
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
        
        // Verify bid is valid
        const isValid = rules.isValidBid(gameState, bid);
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
      const hands = dealDominoesWithSeed(12345);
      gameState.players.forEach((player, i) => {
        const hand = hands[i];
        if (hand) {
          player.hand = hand;
        }
      });
      
      // Test invalid bids are rejected
      const invalidPointBids = [29, 42, 43, 50];
      
      invalidPointBids.forEach(bidValue => {
        const bid: Bid = {
          type: 'points',
          value: bidValue,
          player: 0
        };
        
        const isValid = rules.isValidBid(gameState, bid);
        expect(isValid).toBe(false);
      });
    });
  });
});