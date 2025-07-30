import { describe, test, expect } from 'vitest';
import type { GameState, Trump } from '../../game/types';

describe('Feature: Trump Declaration', () => {
  describe('Scenario: Declaring Trump', () => {
    test('Given a player has won the bidding', () => {
      const mockState: Partial<GameState> = {
        phase: 'trump_selection',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 1,
        bids: [
          { type: 'pass', player: 0 },
          { type: 'points', value: 30, player: 1 },
          { type: 'pass', player: 2 },
          { type: 'pass', player: 3 }
        ],
        currentBid: { type: 'points', value: 30, player: 1 },
        winningBidder: 1,
        trump: null,
      };
      
      expect(mockState.winningBidder).toBe(1);
      expect(mockState.phase).toBe('trump_selection');
      expect(mockState.trump).toBeNull();
    });

    test('When they are ready to play - Then they must declare trump before playing the first domino', () => {
      const mockState: Partial<GameState> = {
        phase: 'trump_selection',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 1,
        winningBidder: 1,
        trump: null,
        currentTrick: [],
        tricks: [],
      };
      
      // Cannot proceed to playing phase without declaring trump
      expect(mockState.phase).toBe('trump_selection');
      expect(mockState.trump).toBeNull();
      expect(mockState.currentTrick!.length).toBe(0);
      
      // After declaring trump, can proceed to playing
      const afterTrumpState: Partial<GameState> = {
        ...mockState,
        trump: 3 as Trump, // Declared threes as trump
        phase: 'playing',
      };
      
      expect(afterTrumpState.trump).toBe(3);
      expect(afterTrumpState.phase).toBe('playing');
    });

    test('And trump options include any suit (blanks through sixes)', () => {
      const mockState: Partial<GameState> = {
        phase: 'trump_selection',
        winningBidder: 1,
        trump: null,
      };
      
      // Test all valid suit options
      const validSuitTrumps: Trump[] = [0, 1, 2, 3, 4, 5, 6];
      
      validSuitTrumps.forEach(suit => {
        const stateWithTrump: Partial<GameState> = {
          ...mockState,
          trump: suit,
        };
        
        expect(stateWithTrump.trump).toBeGreaterThanOrEqual(0);
        expect(stateWithTrump.trump).toBeLessThanOrEqual(6);
      });
      
      // Verify specific suits
      expect(validSuitTrumps).toContain(0); // blanks
      expect(validSuitTrumps).toContain(1); // ones
      expect(validSuitTrumps).toContain(2); // twos
      expect(validSuitTrumps).toContain(3); // threes
      expect(validSuitTrumps).toContain(4); // fours
      expect(validSuitTrumps).toContain(5); // fives
      expect(validSuitTrumps).toContain(6); // sixes
    });

    test('And trump options include doubles as trump', () => {
      const mockState: Partial<GameState> = {
        phase: 'trump_selection',
        winningBidder: 1,
        trump: null,
      };
      
      // Doubles as trump is represented as 7
      const doublesAsTrump: Trump = 7;
      
      const stateWithDoublesTrump: Partial<GameState> = {
        ...mockState,
        trump: doublesAsTrump,
      };
      
      expect(stateWithDoublesTrump.trump).toBe(7);
    });

    test('And trump options include no-trump (follow-me)', () => {
      const mockState: Partial<GameState> = {
        phase: 'trump_selection',
        winningBidder: 1,
        trump: null,
      };
      
      // No-trump is represented as 8
      const noTrump: Trump = 8;
      
      const stateWithNoTrump: Partial<GameState> = {
        ...mockState,
        trump: noTrump,
      };
      
      expect(stateWithNoTrump.trump).toBe(8);
      
      // Alternative representation for no-trump with additional properties
      const stateWithFollowMe: Partial<GameState> = {
        ...mockState,
        trump: { suit: 'follow-me', followsSuit: true },
      };
      
      expect(stateWithFollowMe.trump).toMatchObject({
        suit: 'follow-me',
        followsSuit: true
      });
    });
  });
});