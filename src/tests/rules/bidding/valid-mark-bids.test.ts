import { describe, it, expect } from 'vitest';
import type { Bid } from '../../../game/types';
import { createInitialState, GAME_CONSTANTS, dealDominoesWithSeed } from '../../../game';
import { composeRules, baseLayer } from '../../../game/layers';

const rules = composeRules([baseLayer]);

describe('Feature: Standard Bidding - Valid Mark Bids', () => {
  describe('Scenario: Valid Mark Bids', () => {
    it('should accept 1 mark bid (42 points)', () => {
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
      
      const bid: Bid = {
        type: 'marks',
        value: 1,
        player: 0
      };
      
      const isValid = rules.isValidBid(gameState, bid);
      expect(isValid).toBe(true);
      
      // 1 mark equals 42 points
      expect(bid.value! * GAME_CONSTANTS.TOTAL_POINTS).toBe(42);
    });

    it('should accept 2 marks bid (84 points)', () => {
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
      
      const bid: Bid = {
        type: 'marks',
        value: 2,
        player: 0
      };
      
      const isValid = rules.isValidBid(gameState, bid);
      expect(isValid).toBe(true);
      
      // 2 marks equals 84 points
      expect(bid.value! * GAME_CONSTANTS.TOTAL_POINTS).toBe(84);
    });

    it('should calculate higher marks as multiples of 42 points', () => {
      const gameState = createInitialState();
      gameState.phase = 'bidding';
      gameState.currentPlayer = 0;
      
      // Test various mark values
      const testCases = [
        { marks: 3, expectedPoints: 126 },
        { marks: 4, expectedPoints: 168 },
        { marks: 5, expectedPoints: 210 },
        { marks: 6, expectedPoints: 252 },
        { marks: 7, expectedPoints: 294 }
      ];

      testCases.forEach(({ marks, expectedPoints }) => {
        const actualPoints = marks * GAME_CONSTANTS.TOTAL_POINTS;
        expect(actualPoints).toBe(expectedPoints);
      });
    });
  });
});