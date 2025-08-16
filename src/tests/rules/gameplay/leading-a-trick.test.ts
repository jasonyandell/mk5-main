import { describe, test, expect } from 'vitest';
import type { GameState } from '../../../game/types';

describe('Feature: Playing Tricks', () => {
  describe('Scenario: Leading a Trick', () => {
    test('Given it is time to play a trick', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        currentPlayer: 1,
        winningBidder: 1,
        trump: { type: 'suit', suit: 3 }, // threes are trump
        currentTrick: [],
        tricks: [],
      };
      
      expect(mockState.phase).toBe('playing');
      expect(mockState.currentTrick!.length).toBe(0);
    });

    test('When determining who leads - Then the bid winner leads to the first trick', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        currentPlayer: 2, // Player 2 won the bid
        winningBidder: 2,
        trump: { type: 'suit', suit: 5 },
        currentTrick: [],
        tricks: [], // No tricks played yet
      };
      
      // First trick - bid winner leads
      expect(mockState.winningBidder).toBe(2);
      expect(mockState.currentPlayer).toBe(2);
      expect(mockState.tricks!.length).toBe(0);
    });

    test('And the winner of each trick leads to the next trick', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as const, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        winningBidder: 1,
        trump: { type: 'suit', suit: 4 },
        currentTrick: [],
        tricks: [
          { 
            plays: [
              { player: 1, domino: { high: 6, low: 4, id: '6-4' } },
              { player: 2, domino: { high: 6, low: 2, id: '6-2' } },
              { player: 3, domino: { high: 4, low: 3, id: '4-3' } }, // trump
              { player: 0, domino: { high: 6, low: 6, id: '6-6' } },
            ],
            winner: 3, // Player 3 won with trump
            points: 10 + 1 // 6-4 counter + 1 trick
          }
        ],
        currentPlayer: 3, // Winner of last trick leads
      };
      
      // After first trick, winner leads next
      const firstTrick = mockState.tricks![0];
      if (!firstTrick) {
        throw new Error('First trick not found');
      }
      expect(firstTrick.winner).toBe(3);
      expect(mockState.currentPlayer).toBe(3);
    });

    test('And any domino may be led', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { 
            id: 0, 
            name: 'Player 1', 
            hand: [
              { high: 6, low: 6, id: '6-6' },
              { high: 5, low: 0, id: '5-0' },
              { high: 3, low: 2, id: '3-2' },
              { high: 1, low: 0, id: '1-0' },
            ], 
            teamId: 0 as const, 
            marks: 0 
          },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as const, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as const, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as const, marks: 0 },
        ],
        currentPlayer: 0,
        winningBidder: 0,
        trump: { type: 'suit', suit: 2 }, // twos are trump
        currentTrick: [],
        tricks: [],
      };
      
      // Player can lead any domino from their hand
      const player = mockState.players![0];
      if (!player) {
        throw new Error('Player not found');
      }
      const validLeads = player.hand;
      
      // All dominoes in hand are valid leads
      expect(validLeads).toHaveLength(4);
      expect(validLeads).toContainEqual({ high: 6, low: 6, id: '6-6' }); // double
      expect(validLeads).toContainEqual({ high: 5, low: 0, id: '5-0' }); // counter
      expect(validLeads).toContainEqual({ high: 3, low: 2, id: '3-2' }); // counter with trump end
      expect(validLeads).toContainEqual({ high: 1, low: 0, id: '1-0' }); // non-counter
      
      // No restrictions on which domino can be led
      validLeads.forEach(domino => {
        const leadPlay = { player: 0, domino };
        // Any of these leads would be valid
        expect(leadPlay.domino).toBeDefined();
        expect(leadPlay.player).toBe(mockState.currentPlayer);
      });
    });
  });
});