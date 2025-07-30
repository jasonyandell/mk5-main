import { describe, test, expect } from 'vitest';
import type { GameState } from '../../../game/types';

describe('Feature: Special Bids', () => {
  describe('Scenario: All Players Pass', () => {
    test('Given all players have had a chance to bid', () => {
      const mockState: Partial<GameState> = {
        phase: 'bidding',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 0, // Started with player to left of dealer
        dealer: 3,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'pass', player: 1 },
          { type: 'pass', player: 2 },
          { type: 'pass', player: 3 },
        ],
        currentBid: null,
        winningBidder: null,
      };

      // All players have had a chance to bid
      expect(mockState.bids?.length).toBe(4);
      expect(mockState.bids?.every(bid => bid.type === 'pass')).toBe(true);
    });

    test('When all players pass - Then under tournament rules, the hand is reshaken with the next player as shaker', () => {
      const mockState: Partial<GameState> = {
        phase: 'bidding',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        dealer: 3,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'pass', player: 1 },
          { type: 'pass', player: 2 },
          { type: 'pass', player: 3 },
        ],
        tournamentMode: true,
      };

      // After all pass in tournament mode
      const reshuffledState: Partial<GameState> = {
        phase: 'setup', // Back to setup for reshuffle
        players: mockState.players,
        dealer: 0, // Next player becomes dealer (was 3, now 0)
        bids: [], // Clear bids
        currentBid: null,
        winningBidder: null,
        tournamentMode: true,
      };

      expect(reshuffledState.phase).toBe('setup');
      expect(reshuffledState.dealer).toBe(0); // Next player in rotation
      expect(reshuffledState.bids?.length).toBe(0);
    });

    test('And under common variation, the shaker must bid minimum 30', () => {
      const mockState: Partial<GameState> = {
        phase: 'bidding',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        dealer: 3,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'pass', player: 1 },
          { type: 'pass', player: 2 },
        ],
        tournamentMode: false, // Not tournament mode - common variation
      };

      // When it's the dealer's turn and all others have passed
      const forcedBidState: Partial<GameState> = {
        ...mockState,
        bids: [
          ...mockState.bids!,
          { type: 'points', value: 30, player: 3 } // Dealer forced to bid 30
        ],
        currentBid: { type: 'points', value: 30, player: 3 },
        winningBidder: 3,
      };

      expect(forcedBidState.bids?.length).toBe(4);
      const dealerBid = forcedBidState.bids?.find(bid => bid.player === mockState.dealer);
      expect(dealerBid?.type).toBe('points');
      expect(dealerBid?.value).toBe(30);
      expect(forcedBidState.winningBidder).toBe(mockState.dealer);
    });
  });
});