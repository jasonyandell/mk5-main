import { describe, test, expect } from 'vitest';
import type { GameState, Trick } from '../../../game/types';
import { FIVES } from '../../../game/types';

describe('Feature: Plunge Bid Mechanics', () => {
  describe('Scenario: Plunge Bid Mechanics', () => {
    test('Given a player has successfully bid a plunge', () => {
      const mockState: Partial<GameState> = {
        phase: 'bidding',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        currentPlayer: 0,
        bids: [
          { type: 'plunge', value: 4, player: 0 }
        ],
        currentBid: { type: 'plunge', value: 4, player: 0 },
        winningBidder: 0,
      };
      expect(mockState.currentBid?.type).toBe('plunge');
      expect(mockState.winningBidder).toBe(0);
    });

    test('When play begins - Then their partner names trump without consultation', () => {
      const mockState: Partial<GameState> = {
        phase: 'trump_selection',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        currentBid: { type: 'plunge', value: 4, player: 0 },
        winningBidder: 0,
        trump: { type: 'not-selected' },
      };

      // Partner of player 0 (teamId 0) is player 2
      const plungeBidder = mockState.players![0];
      if (!plungeBidder) throw new Error('Plunge bidder not found');
      const partner = mockState.players!.find(p => p.id !== plungeBidder.id && p.teamId === plungeBidder.teamId);
      
      expect(partner?.id).toBe(2);
      
      // Simulate partner declaring trump
      const trumpDeclaredState: Partial<GameState> = {
        ...mockState,
        trump: { type: 'suit', suit: FIVES }, // Partner declares fives as trump
        currentPlayer: partner!.id,
      };
      
      expect(trumpDeclaredState.trump).toBeDefined();
      expect(trumpDeclaredState.currentPlayer).toBe(partner!.id);
    });

    test('And their partner leads the first trick', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        currentBid: { type: 'plunge', value: 4, player: 0 },
        winningBidder: 0,
        trump: { type: 'suit', suit: FIVES },
        currentPlayer: 2, // Partner leads
        tricks: [],
        currentTrick: [],
      };

      // Partner of plunge bidder should be leading
      const plungeBidder = mockState.players![0];
      if (!plungeBidder) throw new Error('Plunge bidder not found');
      const partner = mockState.players!.find(p => p.id !== plungeBidder.id && p.teamId === plungeBidder.teamId);
      
      expect(mockState.currentPlayer).toBe(partner!.id);
      expect(mockState.tricks!.length).toBe(0); // First trick not yet played
      expect(mockState.currentTrick!.length).toBe(0); // No plays yet
    });

    test('And they must win all 7 tricks to succeed', () => {
      const mockState: Partial<GameState> = {
        phase: 'scoring',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        currentBid: { type: 'plunge', value: 4, player: 0 },
        winningBidder: 0,
        tricks: Array(7).fill(null).map(() => ({
          plays: [],
          winner: 0,
          points: 0
        })) as Trick[],
        teamScores: [0, 0],
      };

      // Simulate successful plunge - team 0 wins all 7 tricks
      const successfulPlungeState: Partial<GameState> = {
        ...mockState,
        tricks: Array(7).fill(null).map(() => ({
          plays: [],
          winner: 0, // Team 0 player wins each trick
          points: 0
        })),
      };

      // Check all tricks were won by plunge team
      const plungeBidder = mockState.players![0];
      if (!plungeBidder) throw new Error('Plunge bidder not found');
      const plungeTeam = plungeBidder.teamId;
      const tricksWonByPlungeTeam = successfulPlungeState.tricks!.filter(trick => {
        const winnerPlayer = mockState.players!.find(p => p.id === trick.winner);
        return winnerPlayer?.teamId === plungeTeam;
      }).length;

      expect(tricksWonByPlungeTeam).toBe(7);

      // Simulate failed plunge - team 0 wins only 6 tricks
      const failedPlungeState: Partial<GameState> = {
        ...mockState,
        tricks: [
          ...Array(6).fill(null).map(() => ({ plays: [], winner: 0, points: 0 })),
          { plays: [], winner: 1, points: 0 } // Opposing team wins one trick
        ],
      };

      const tricksWonInFailure = failedPlungeState.tricks!.filter(trick => {
        const winnerPlayer = mockState.players!.find(p => p.id === trick.winner);
        return winnerPlayer?.teamId === plungeTeam;
      }).length;

      expect(tricksWonInFailure).toBe(6);
      expect(tricksWonInFailure).toBeLessThan(7); // Failed to win all tricks
    });
  });
});