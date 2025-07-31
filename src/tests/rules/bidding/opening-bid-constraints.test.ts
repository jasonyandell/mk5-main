import { describe, it, expect } from 'vitest';
import type { Bid } from '../../../game/types';
import { isValidOpeningBid, isValidBid, createInitialState, GAME_CONSTANTS, createDominoes, dealDominoes, shuffleDominoes } from '../../../game';

describe('Feature: Standard Bidding - Opening Bid Constraints', () => {
  describe('Scenario: Opening Bid Constraints', () => {
    it('should enforce opening bid constraints', () => {
      // Given no bids have been made
      const gameState = createInitialState();
      gameState.phase = 'bidding';
      gameState.currentPlayer = 0;
      gameState.bids = [];
      gameState.currentBid = null;
      
      // Deal dominoes to players
      const dominoes = createDominoes();
      const hands = dealDominoes(shuffleDominoes(dominoes, 12345));
      gameState.players.forEach((player, i) => {
        player.hand = hands[i];
      });

      // When a player makes the opening bid
      // Then the minimum bid is 30 points
      const pointsBids = [
        { bid: { type: 'points' as const, value: 29, player: 0 }, valid: false },
        { bid: { type: 'points' as const, value: 30, player: 0 }, valid: true },
        { bid: { type: 'points' as const, value: 41, player: 0 }, valid: true },
        { bid: { type: 'points' as const, value: 42, player: 0 }, valid: false }
      ];
      
      pointsBids.forEach(({ bid, valid }) => {
        expect(isValidOpeningBid(bid, gameState.players[0].hand, gameState.tournamentMode)).toBe(valid);
      });

      // And the maximum opening bid is 2 marks (84 points)
      const markBids = [
        { bid: { type: 'marks' as const, value: 1, player: 0 }, valid: true },
        { bid: { type: 'marks' as const, value: 2, player: 0 }, valid: true },
        { bid: { type: 'marks' as const, value: 3, player: 0 }, valid: false },
        { bid: { type: 'marks' as const, value: 4, player: 0 }, valid: false }
      ];
      
      markBids.forEach(({ bid, valid }) => {
        expect(isValidOpeningBid(bid, gameState.players[0].hand, gameState.tournamentMode)).toBe(valid);
      });
      
      // Verify 2 marks equals 84 points
      expect(2 * GAME_CONSTANTS.TOTAL_POINTS).toBe(84);

      // And the exception is a plunge bid which may open at 4 marks (not in tournament mode)
      // In tournament mode, plunge is not allowed
      if (gameState.tournamentMode) {
        const plungeBid: Bid = { type: 'plunge' as const, value: 4, player: 0 };
        expect(isValidOpeningBid(plungeBid, gameState.players[0].hand, gameState.tournamentMode)).toBe(false);
      } else {
        // Test invalid plunge first (with current hand - not enough doubles)
        const invalidPlunge: Bid = { type: 'plunge' as const, value: 3, player: 0 };
        expect(isValidOpeningBid(invalidPlunge, gameState.players[0].hand, false)).toBe(false);
        
        // Now give player 4 doubles for valid plunge tests
        const handWith4Doubles = [
          { high: 0, low: 0, id: '0-0' },
          { high: 1, low: 1, id: '1-1' },
          { high: 2, low: 2, id: '2-2' },
          { high: 3, low: 3, id: '3-3' },
          { high: 4, low: 0, id: '4-0' },
          { high: 5, low: 0, id: '5-0' },
          { high: 6, low: 0, id: '6-0' }
        ];
        
        const validPlunge4: Bid = { type: 'plunge' as const, value: 4, player: 0 };
        const validPlunge5: Bid = { type: 'plunge' as const, value: 5, player: 0 };
        
        expect(isValidOpeningBid(validPlunge4, handWith4Doubles, false)).toBe(true);
        expect(isValidOpeningBid(validPlunge5, handWith4Doubles, false)).toBe(true);
      }

      // Pass is always valid (but handled separately, not through isValidOpeningBid)
      const passBid: Bid = { type: 'pass', player: 0 };
      // Pass bids are handled by isValidBid, not isValidOpeningBid
      expect(isValidBid(passBid, null, gameState)).toBe(true);
    });

    it('should create valid opening bids', () => {
      const gameState = createInitialState();
      gameState.phase = 'bidding';
      gameState.currentPlayer = 0;
      
      // Give player doubles for plunge bid
      gameState.players[0].hand = [
        { high: 0, low: 0, id: '0-0' },
        { high: 1, low: 1, id: '1-1' },
        { high: 2, low: 2, id: '2-2' },
        { high: 3, low: 3, id: '3-3' },
        { high: 4, low: 0, id: '4-0' },
        { high: 5, low: 0, id: '5-0' },
        { high: 6, low: 0, id: '6-0' }
      ];

      // Test creating actual bid objects (excluding pass which is handled separately)
      const validOpeningBids: Bid[] = [
        { type: 'points', value: 30, player: 0 },
        { type: 'points', value: 35, player: 0 },
        { type: 'points', value: 41, player: 0 },
        { type: 'marks', value: 1, player: 0 },
        { type: 'marks', value: 2, player: 0 }
      ];

      validOpeningBids.forEach(bid => {
        expect(isValidOpeningBid(bid, gameState.players[0].hand, gameState.tournamentMode)).toBe(true);
      });
      
      // Pass is handled by isValidBid
      const passBid: Bid = { type: 'pass', player: 0 };
      expect(isValidBid(passBid, null, gameState)).toBe(true);
      
      // Plunge is only valid in non-tournament mode with 4+ doubles
      if (!gameState.tournamentMode) {
        const plungeBid: Bid = { type: 'plunge', value: 4, player: 0 };
        expect(isValidOpeningBid(plungeBid, gameState.players[0].hand, false)).toBe(true);
      }
    });

    it('should reject invalid opening bids', () => {
      const gameState = createInitialState();
      gameState.phase = 'bidding';
      gameState.currentPlayer = 0;
      
      // Deal dominoes to players
      const dominoes = createDominoes();
      const hands = dealDominoes(shuffleDominoes(dominoes, 12345));
      gameState.players.forEach((player, i) => {
        player.hand = hands[i];
      });
      
      // Test creating invalid bid objects that should be rejected
      const invalidOpeningBids: Bid[] = [
        { type: 'points', value: 29, player: 0 }, // below minimum
        { type: 'points', value: 42, player: 0 }, // above maximum
        { type: 'marks', value: 3, player: 0 }, // 3 marks cannot be opening bid
        { type: 'marks', value: 4, player: 0 }, // regular marks cannot open at 4
        { type: 'plunge', value: 3, player: 0 } // plunge must be at least 4 marks
      ];

      invalidOpeningBids.forEach(bid => {
        expect(isValidOpeningBid(bid, gameState.players[0].hand, gameState.tournamentMode)).toBe(false);
      });
    });
  });
});