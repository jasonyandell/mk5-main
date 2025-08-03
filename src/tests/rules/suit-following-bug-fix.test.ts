import { describe, it, expect } from 'vitest';
import { createTestState } from '../helpers/gameTestHelper';
import { isValidPlay } from '../../game/core/rules';
import { analyzeSuits } from '../../game/core/suit-analysis';
import type { Domino } from '../../game/types';

describe('Suit Following Bug Fix', () => {
  it('should not allow playing 1-1 when player has 5-1 and must follow suit (5s)', () => {
    // This is the original bug report scenario
    const playerHand: Domino[] = [
      { id: '1-2', high: 2, low: 1 },
      { id: '5-1', high: 5, low: 1 }, // Must play this to follow 5s
      { id: '1-1', high: 1, low: 1 }, // Cannot play this
      { id: '3-2', high: 3, low: 2 },
      { id: '4-3', high: 4, low: 3 },
      { id: '6-2', high: 6, low: 2 },
      { id: '4-2', high: 4, low: 2 }
    ];

    const state = createTestState({
      phase: 'playing',
      trump: { type: 'suit', suit: 1 }, // Ones are trump
      currentTrick: [{
        player: 0,
        domino: { id: '5-4', high: 5, low: 4 } // 5-4 leads (fives suit)
      }],
      currentSuit: 5, // Fives were led
      currentPlayer: 2,
      players: [
        { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
        { id: 1, name: 'Player 1', teamId: 1, marks: 0, hand: [] },
        { 
          id: 2, 
          name: 'Player 2', 
          teamId: 0, 
          marks: 0, 
          hand: playerHand,
          suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: 1 }) // Trump = 1
        },
        { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
      ]
    });

    // Player 2 should NOT be able to play 1-1 (trump)
    const domino_1_1 = playerHand.find(d => d.id === '1-1')!;
    expect(isValidPlay(state, domino_1_1, 2)).toBe(false);

    // Player 2 MUST be able to play 5-1 (has the led suit)
    const domino_5_1 = playerHand.find(d => d.id === '5-1')!;
    expect(isValidPlay(state, domino_5_1, 2)).toBe(true);

    // Other dominoes without 5s should not be playable
    const domino_3_2 = playerHand.find(d => d.id === '3-2')!;
    expect(isValidPlay(state, domino_3_2, 2)).toBe(false);
  });

  it('should allow playing trump dominoes that also contain the led suit', () => {
    // Test that 5-1 can be played to follow 5s even though it contains trump (1)
    const playerHand: Domino[] = [
      { id: '5-1', high: 5, low: 1 }, // Has both 5 (led suit) and 1 (trump)
      { id: '5-3', high: 5, low: 3 }, // Has 5 (led suit)
      { id: '1-2', high: 2, low: 1 }, // Has trump but not led suit
      { id: '3-4', high: 4, low: 3 }  // Has neither
    ];

    const state = createTestState({
      phase: 'playing',
      trump: { type: 'suit', suit: 1 }, // Ones are trump
      currentTrick: [{
        player: 0,
        domino: { id: '5-6', high: 6, low: 5 } // 5-6 leads (sixes suit)
      }],
      currentSuit: 6, // Sixes were led
      currentPlayer: 1,
      players: [
        { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
        { 
          id: 1, 
          name: 'Player 1', 
          teamId: 1, 
          marks: 0, 
          hand: playerHand,
          suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: 1 }) // Trump = 1
        },
        { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
        { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
      ]
    });

    // Cannot follow 6s, so all plays should be valid
    expect(isValidPlay(state, playerHand[0], 1)).toBe(true); // 5-1
    expect(isValidPlay(state, playerHand[1], 1)).toBe(true); // 5-3
    expect(isValidPlay(state, playerHand[2], 1)).toBe(true); // 1-2
    expect(isValidPlay(state, playerHand[3], 1)).toBe(true); // 3-4
  });
});