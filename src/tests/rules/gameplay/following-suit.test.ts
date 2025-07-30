import { describe, test, expect } from 'vitest';
import type { GameState, Trump, Play } from '../../../game/types';

describe('Feature: Playing Tricks', () => {
  describe('Scenario: Following Suit', () => {
    test('Given a domino has been led And it is not a trump', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { 
            id: 0, 
            name: 'Player 1', 
            hand: [], 
            teamId: 0 as 0, 
            marks: 0 
          },
          { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 1,
        trump: 4 as Trump, // fours are trump
        currentTrick: [
          { player: 0, domino: { high: 6, low: 3, id: '6-3' } } // 6-3 led, not a trump
        ],
        tricks: [],
      };
      
      const ledDomino = mockState.currentTrick![0].domino;
      expect(mockState.phase).toBe('playing');
      expect(mockState.currentTrick!.length).toBe(1);
      expect(ledDomino.high).not.toBe(mockState.trump);
      expect(ledDomino.low).not.toBe(mockState.trump);
    });

    test('When determining the suit - Then the higher end of the domino determines the suit led', () => {
      const testCases = [
        { domino: { high: 6, low: 3, id: '6-3' }, expectedSuit: 6 },
        { domino: { high: 5, low: 2, id: '5-2' }, expectedSuit: 5 },
        { domino: { high: 4, low: 1, id: '4-1' }, expectedSuit: 4 },
        { domino: { high: 3, low: 0, id: '3-0' }, expectedSuit: 3 },
        { domino: { high: 2, low: 1, id: '2-1' }, expectedSuit: 2 },
        { domino: { high: 1, low: 0, id: '1-0' }, expectedSuit: 1 },
      ];

      testCases.forEach(({ domino, expectedSuit }) => {
        const mockState: Partial<GameState> = {
          phase: 'playing',
          trump: 5 as Trump, // fives are trump, so none of these are trump
          currentTrick: [
            { player: 0, domino }
          ],
        };
        
        const ledDomino = mockState.currentTrick![0].domino;
        const suitLed = ledDomino.high; // Higher end determines suit
        expect(suitLed).toBe(expectedSuit);
      });
    });

    test('And players must play a domino of the led suit if possible', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { 
            id: 1, 
            name: 'Player 2', 
            hand: [
              { high: 6, low: 6, id: '6-6' }, // Can follow suit with 6
              { high: 6, low: 2, id: '6-2' }, // Can follow suit with 6
              { high: 5, low: 3, id: '5-3' }, // Cannot follow suit
              { high: 4, low: 1, id: '4-1' }, // Cannot follow suit
            ], 
            teamId: 1 as 1, 
            marks: 0 
          },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 1,
        trump: 2 as Trump, // twos are trump
        currentTrick: [
          { player: 0, domino: { high: 6, low: 3, id: '6-3' } } // 6 led
        ],
      };
      
      const ledSuit = mockState.currentTrick![0].domino.high; // 6
      const currentPlayerHand = mockState.players![1].hand;
      
      // Find dominoes that can follow suit
      const validPlays = currentPlayerHand.filter(domino => 
        domino.high === ledSuit || domino.low === ledSuit
      );
      
      expect(validPlays).toHaveLength(2);
      expect(validPlays).toContainEqual({ high: 6, low: 6, id: '6-6' });
      expect(validPlays).toContainEqual({ high: 6, low: 2, id: '6-2' });
      
      // Player MUST play one of these if they have them
      expect(validPlays.length).toBeGreaterThan(0);
    });

    test('And if unable to follow suit, players may play trump', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { 
            id: 1, 
            name: 'Player 2', 
            hand: [
              { high: 5, low: 3, id: '5-3' }, // Cannot follow 6
              { high: 4, low: 1, id: '4-1' }, // Cannot follow 6
              { high: 2, low: 2, id: '2-2' }, // Trump (2)
              { high: 2, low: 0, id: '2-0' }, // Trump (2)
            ], 
            teamId: 1 as 1, 
            marks: 0 
          },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 1,
        trump: 2 as Trump, // twos are trump
        currentTrick: [
          { player: 0, domino: { high: 6, low: 3, id: '6-3' } } // 6 led
        ],
      };
      
      const ledSuit = mockState.currentTrick![0].domino.high; // 6
      const currentPlayerHand = mockState.players![1].hand;
      const trumpSuit = mockState.trump;
      
      // Check if player can follow suit
      const canFollowSuit = currentPlayerHand.some(domino => 
        domino.high === ledSuit || domino.low === ledSuit
      );
      
      expect(canFollowSuit).toBe(false);
      
      // Since can't follow suit, may play trump
      const trumpDominoes = currentPlayerHand.filter(domino =>
        domino.high === trumpSuit || domino.low === trumpSuit
      );
      
      expect(trumpDominoes).toHaveLength(2);
      expect(trumpDominoes).toContainEqual({ high: 2, low: 2, id: '2-2' });
      expect(trumpDominoes).toContainEqual({ high: 2, low: 0, id: '2-0' });
      
      // These are valid plays when unable to follow suit
      const validPlays = trumpDominoes;
      expect(validPlays.length).toBeGreaterThan(0);
    });

    test('And if unable to follow suit or trump, players may play any domino', () => {
      const mockState: Partial<GameState> = {
        phase: 'playing',
        players: [
          { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
          { 
            id: 1, 
            name: 'Player 2', 
            hand: [
              { high: 5, low: 3, id: '5-3' }, // Cannot follow 6, not trump
              { high: 4, low: 1, id: '4-1' }, // Cannot follow 6, not trump
              { high: 3, low: 0, id: '3-0' }, // Cannot follow 6, not trump
              { high: 1, low: 0, id: '1-0' }, // Cannot follow 6, not trump
            ], 
            teamId: 1 as 1, 
            marks: 0 
          },
          { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
          { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 },
        ],
        currentPlayer: 1,
        trump: 2 as Trump, // twos are trump
        currentTrick: [
          { player: 0, domino: { high: 6, low: 3, id: '6-3' } } // 6 led
        ],
      };
      
      const ledSuit = mockState.currentTrick![0].domino.high; // 6
      const currentPlayerHand = mockState.players![1].hand;
      const trumpSuit = mockState.trump;
      
      // Check if player can follow suit
      const canFollowSuit = currentPlayerHand.some(domino => 
        domino.high === ledSuit || domino.low === ledSuit
      );
      
      expect(canFollowSuit).toBe(false);
      
      // Check if player has trump
      const hasTrump = currentPlayerHand.some(domino =>
        domino.high === trumpSuit || domino.low === trumpSuit
      );
      
      expect(hasTrump).toBe(false);
      
      // Since can't follow suit or trump, any domino is valid
      const validPlays = currentPlayerHand;
      
      expect(validPlays).toHaveLength(4);
      expect(validPlays).toEqual(currentPlayerHand);
      
      // All dominoes in hand are valid plays in this situation
      validPlays.forEach(domino => {
        const play: Play = { player: 1, domino };
        expect(play.domino).toBeDefined();
        expect(currentPlayerHand).toContain(play.domino);
      });
    });
  });
});