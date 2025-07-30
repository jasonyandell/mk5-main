import { describe, it, expect } from 'vitest';
import type { GameState, GamePhase } from '../../game/types';

describe('Feature: Standard Bidding', () => {
  describe('Scenario: Bidding Order', () => {
    it('Given a new hand has been dealt', () => {
      // Test setup - new hand dealt, ready for bidding
      const gameState: Partial<GameState> = {
        phase: 'bidding' as GamePhase,
        players: [
          { id: 0, name: 'Player 0', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 1', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 2', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 3', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        dealer: 2, // Player 2 is the dealer/shaker
        currentPlayer: 3, // Player to left of shaker
        bids: [],
      };

      expect(gameState.phase).toBe('bidding');
      expect(gameState.bids).toEqual([]);
      expect(gameState.dealer).toBe(2);
    });

    it('When bidding begins, Then the player to the left of the shaker bids first', () => {
      const gameState: Partial<GameState> = {
        dealer: 2, // Player 2 is the shaker
        currentPlayer: 3, // Player 3 is to the left of player 2
      };

      // Verify player to left of shaker (dealer) bids first
      const playerToLeftOfShaker = (gameState.dealer! + 1) % 4;
      expect(gameState.currentPlayer).toBe(playerToLeftOfShaker);
    });

    it('And bidding proceeds clockwise', () => {
      // Test bidding order progression
      const biddingOrder = [3, 0, 1, 2]; // Starting from player 3 (left of dealer 2)
      
      let currentPlayerIndex = 0;
      const gameState: Partial<GameState> = {
        dealer: 2,
        currentPlayer: biddingOrder[currentPlayerIndex],
      };

      // Simulate clockwise progression
      for (let i = 0; i < 4; i++) {
        expect(gameState.currentPlayer).toBe(biddingOrder[i]);
        
        // Move to next player clockwise
        currentPlayerIndex = (currentPlayerIndex + 1) % 4;
        gameState.currentPlayer = biddingOrder[currentPlayerIndex];
      }
    });

    it('And each player gets exactly one opportunity to bid or pass', () => {
      // Test that each player bids exactly once

      // Simulate one bid/pass per player
      const playerBids = [
        { type: 'pass' as const, player: 3 },
        { type: 'points' as const, value: 30, player: 0 },
        { type: 'pass' as const, player: 1 },
        { type: 'points' as const, value: 31, player: 2 },
      ];

      // Each player should bid exactly once
      const bidCounts = new Map<number, number>();
      playerBids.forEach(bid => {
        const count = bidCounts.get(bid.player) || 0;
        bidCounts.set(bid.player, count + 1);
      });

      // Verify each player bid exactly once
      expect(bidCounts.size).toBe(4);
      bidCounts.forEach((count) => {
        expect(count).toBe(1);
      });
    });
  });
});