import { describe, test, expect } from 'vitest';
import { 
  createInitialState, 
  getValidPlays, 
  canFollowSuit, 
  isValidPlay,
  type Domino
} from '../../../game';
import { analyzeSuits } from '../../../game/core/suit-analysis';

describe('Feature: Playing Tricks', () => {
  describe('Scenario: Following Suit', () => {
    test('Given a domino has been led And it is not a trump', () => {
      const gameState = createInitialState();
      gameState.phase = 'playing';
      gameState.trump = 4; // fours are trump
      gameState.currentTrick = [
        { player: 0, domino: { high: 6, low: 3, id: '6-3' } } // 6-3 led, not a trump
      ];
      
      const ledDomino = gameState.currentTrick[0].domino;
      expect(gameState.phase).toBe('playing');
      expect(gameState.currentTrick.length).toBe(1);
      expect(ledDomino.high).not.toBe(gameState.trump);
      expect(ledDomino.low).not.toBe(gameState.trump);
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
        const gameState = createInitialState();
        gameState.phase = 'playing';
        gameState.trump = 5; // fives are trump, so none of these are trump
        gameState.currentTrick = [
          { player: 0, domino }
        ];
        
        const ledDomino = gameState.currentTrick[0].domino;
        const suitLed = ledDomino.high; // Higher end determines suit
        expect(suitLed).toBe(expectedSuit);
      });
    });

    test('And players must play a domino of the led suit if possible', () => {
      const gameState = createInitialState();
      gameState.phase = 'playing';
      gameState.trump = 2; // twos are trump
      gameState.currentTrick = [
        { player: 0, domino: { high: 6, low: 3, id: '6-3' } } // 6 led
      ];
      gameState.currentSuit = 6; // Sixes were led
      
      const playerHand: Domino[] = [
        { high: 6, low: 6, id: '6-6' }, // Can follow suit with 6
        { high: 6, low: 4, id: '6-4' }, // Can follow suit with 6
        { high: 5, low: 3, id: '5-3' }, // Cannot follow suit
        { high: 4, low: 1, id: '4-1' }, // Cannot follow suit
      ];
      
      gameState.players[1].hand = playerHand;
      gameState.players[1].suitAnalysis = analyzeSuits(playerHand, gameState.trump);
      
      const validPlays = getValidPlays(gameState, 1);
      
      // Should be able to follow suit with dominoes containing 6
      const canFollowSuitResult = canFollowSuit(gameState.players[1], 6);
      expect(canFollowSuitResult).toBe(true);
      
      // Valid plays should include only dominoes that can follow suit (contain 6)
      expect(validPlays).toHaveLength(2);
      expect(validPlays).toContainEqual({ high: 6, low: 6, id: '6-6' });
      expect(validPlays).toContainEqual({ high: 6, low: 4, id: '6-4' });
      
      // Verify each valid play is actually valid
      validPlays.forEach(domino => {
        expect(isValidPlay(gameState, domino, 1)).toBe(true);
      });
    });

    test('And if unable to follow suit, players may play trump', () => {
      const gameState = createInitialState();
      gameState.phase = 'playing';
      gameState.trump = 2; // twos are trump
      gameState.currentTrick = [
        { player: 0, domino: { high: 6, low: 3, id: '6-3' } } // 6 led
      ];
      gameState.currentSuit = 6; // Sixes were led
      
      const playerHand: Domino[] = [
        { high: 5, low: 3, id: '5-3' }, // Cannot follow 6
        { high: 4, low: 1, id: '4-1' }, // Cannot follow 6
        { high: 2, low: 2, id: '2-2' }, // Trump (2)
        { high: 2, low: 0, id: '2-0' }, // Trump (2)
      ];
      
      gameState.players[1].hand = playerHand;
      gameState.players[1].suitAnalysis = analyzeSuits(playerHand, gameState.trump);
      
      // Check if player can follow suit
      const canFollowSuitResult = canFollowSuit(gameState.players[1], 6);
      expect(canFollowSuitResult).toBe(false);
      
      // Since can't follow suit, all dominoes are valid (including trump)
      const validPlays = getValidPlays(gameState, 1);
      expect(validPlays).toHaveLength(4); // All dominoes are valid
      
      // Verify trump dominoes are valid plays
      const trumpDominoes = [
        { high: 2, low: 2, id: '2-2' },
        { high: 2, low: 0, id: '2-0' }
      ];
      
      trumpDominoes.forEach(domino => {
        expect(validPlays).toContainEqual(domino);
        expect(isValidPlay(gameState, domino, 1)).toBe(true);
      });
    });

    test('And if unable to follow suit or trump, players may play any domino', () => {
      const gameState = createInitialState();
      gameState.phase = 'playing';
      gameState.trump = 2; // twos are trump
      gameState.currentTrick = [
        { player: 0, domino: { high: 6, low: 3, id: '6-3' } } // 6 led
      ];
      gameState.currentSuit = 6; // Sixes were led
      
      const playerHand: Domino[] = [
        { high: 5, low: 3, id: '5-3' }, // Cannot follow 6, not trump
        { high: 4, low: 1, id: '4-1' }, // Cannot follow 6, not trump
        { high: 3, low: 0, id: '3-0' }, // Cannot follow 6, not trump
        { high: 1, low: 0, id: '1-0' }, // Cannot follow 6, not trump
      ];
      
      gameState.players[1].hand = playerHand;
      gameState.players[1].suitAnalysis = analyzeSuits(playerHand, gameState.trump);
      
      // Check if player can follow suit
      const canFollowSuitResult = canFollowSuit(gameState.players[1], 6);
      expect(canFollowSuitResult).toBe(false);
      
      // Check if player has trump (they don't)
      const hasTrump = playerHand.some(domino =>
        domino.high === gameState.trump || domino.low === gameState.trump
      );
      expect(hasTrump).toBe(false);
      
      // Since can't follow suit or trump, any domino is valid
      const validPlays = getValidPlays(gameState, 1);
      expect(validPlays).toHaveLength(4);
      expect(validPlays).toEqual(playerHand);
      
      // Verify all dominoes are valid plays in this situation
      playerHand.forEach(domino => {
        expect(isValidPlay(gameState, domino, 1)).toBe(true);
      });
    });

    test('And doubles follow their natural suit unless doubles are trump', () => {
      const gameState = createInitialState();
      gameState.phase = 'playing';
      gameState.trump = 3; // threes are trump
      gameState.currentTrick = [
        { player: 0, domino: { high: 5, low: 5, id: '5-5' } } // 5-5 led (natural suit 5)
      ];
      gameState.currentSuit = 5; // Fives were led
      
      const playerHand: Domino[] = [
        { high: 5, low: 4, id: '5-4' }, // Can follow suit (contains 5)
        { high: 6, low: 6, id: '6-6' }, // Cannot follow suit (double 6, not 5)
        { high: 3, low: 3, id: '3-3' }, // Trump double
        { high: 4, low: 1, id: '4-1' }, // Cannot follow suit
      ];
      
      gameState.players[1].hand = playerHand;
      gameState.players[1].suitAnalysis = analyzeSuits(playerHand, gameState.trump);
      
      // Player can follow suit with 5-4
      const canFollowSuitResult = canFollowSuit(gameState.players[1], 5);
      expect(canFollowSuitResult).toBe(true);
      
      // Only dominoes that can follow suit should be valid
      const validPlays = getValidPlays(gameState, 1);
      expect(validPlays).toHaveLength(1);
      expect(validPlays).toContainEqual({ high: 5, low: 4, id: '5-4' });
      
      // Verify the 5-4 is a valid play
      expect(isValidPlay(gameState, { high: 5, low: 4, id: '5-4' }, 1)).toBe(true);
      
      // Verify other dominoes are not valid (must follow suit)
      expect(isValidPlay(gameState, { high: 6, low: 6, id: '6-6' }, 1)).toBe(false);
    });

    test('And trump dominoes do not have to follow suit (they trump)', () => {
      const gameState = createInitialState();
      gameState.phase = 'playing';
      gameState.trump = 3; // threes are trump
      gameState.currentTrick = [
        { player: 0, domino: { high: 6, low: 5, id: '6-5' } } // 6 led
      ];
      
      const playerHand: Domino[] = [
        { high: 3, low: 2, id: '3-2' }, // Trump domino (contains 3)
        { high: 4, low: 1, id: '4-1' }, // Cannot follow suit, not trump
      ];
      
      gameState.players[1].hand = playerHand;
      gameState.players[1].suitAnalysis = analyzeSuits(playerHand, gameState.trump);
      
      // Player cannot follow suit with 6
      const canFollowSuitResult = canFollowSuit(gameState.players[1], 6);
      expect(canFollowSuitResult).toBe(false);
      
      // Since can't follow suit, all dominoes are valid
      const validPlays = getValidPlays(gameState, 1);
      expect(validPlays).toHaveLength(2);
      
      // Trump domino is valid even though it doesn't contain 6
      expect(isValidPlay(gameState, { high: 3, low: 2, id: '3-2' }, 1)).toBe(true);
      expect(isValidPlay(gameState, { high: 4, low: 1, id: '4-1' }, 1)).toBe(true);
    });
  });
});