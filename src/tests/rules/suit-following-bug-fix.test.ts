import { describe, it, expect } from 'vitest';
import { createTestState } from '../helpers/gameTestHelper';
import { composeRules } from '../../game/layers/compose';
import { baseLayer } from '../../game/layers';
import { analyzeSuits } from '../../game/core/suit-analysis';
import type { Domino } from '../../game/types';
import { ACES, FIVES, SIXES } from '../../game/types';

const rules = composeRules([baseLayer]);

describe('Suit Following Bug Fix', () => {
  it('should allow any domino when only trump dominoes contain the led suit', () => {
    // When 1s are trump and 5s are led, 5-1 is trump and cannot follow 5s
    const playerHand: Domino[] = [
      { id: '1-2', high: 2, low: 1 }, // Trump, cannot follow 5s
      { id: '5-1', high: 5, low: 1 }, // Trump (contains 1), cannot follow 5s
      { id: '1-1', high: 1, low: 1 }, // Trump, cannot follow 5s
      { id: '3-2', high: 3, low: 2 }, // Not trump, cannot follow 5s
      { id: '4-3', high: 4, low: 3 }, // Not trump, cannot follow 5s
      { id: '6-2', high: 6, low: 2 }, // Not trump, cannot follow 5s
      { id: '4-2', high: 4, low: 2 }  // Not trump, cannot follow 5s
    ];

    const state = createTestState({
      phase: 'playing',
      trump: { type: 'suit', suit: ACES }, // Ones are trump
      currentTrick: [{
        player: 0,
        domino: { id: '5-4', high: 5, low: 4 } // 5-4 leads (fives suit)
      }],
      currentSuit: FIVES, // Fives were led
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
          suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: ACES }) // Trump = 1
        },
        { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
      ]
    });

    // Since 5-1 is trump, it cannot follow 5s
    // Player has NO dominoes that can follow 5s (non-trump 5s)
    // Therefore ALL dominoes should be playable
    const domino_1_1 = playerHand.find(d => d.id === '1-1')!;
    expect(rules.isValidPlay(state, domino_1_1, 2)).toBe(true);

    const domino_5_1 = playerHand.find(d => d.id === '5-1')!;
    expect(rules.isValidPlay(state, domino_5_1, 2)).toBe(true);

    const domino_3_2 = playerHand.find(d => d.id === '3-2')!;
    expect(rules.isValidPlay(state, domino_3_2, 2)).toBe(true);
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
      trump: { type: 'suit', suit: ACES }, // Ones are trump
      currentTrick: [{
        player: 0,
        domino: { id: '5-6', high: 6, low: 5 } // 5-6 leads (sixes suit)
      }],
      currentSuit: SIXES, // Sixes were led
      currentPlayer: 1,
      players: [
        { id: 0, name: 'Player 0', teamId: 0, marks: 0, hand: [] },
        { 
          id: 1, 
          name: 'Player 1', 
          teamId: 1, 
          marks: 0, 
          hand: playerHand,
          suitAnalysis: analyzeSuits(playerHand, { type: 'suit', suit: ACES }) // Trump = 1
        },
        { id: 2, name: 'Player 2', teamId: 0, marks: 0, hand: [] },
        { id: 3, name: 'Player 3', teamId: 1, marks: 0, hand: [] }
      ]
    });

    // Cannot follow 6s, so all plays should be valid
    const domino0 = playerHand[0];
    const domino1 = playerHand[1]; 
    const domino2 = playerHand[2];
    const domino3 = playerHand[3];
    
    if (!domino0 || !domino1 || !domino2 || !domino3) {
      throw new Error('Player hand dominoes cannot be undefined');
    }
    
    expect(rules.isValidPlay(state, domino0, 1)).toBe(true); // 5-1
    expect(rules.isValidPlay(state, domino1, 1)).toBe(true); // 5-3
    expect(rules.isValidPlay(state, domino2, 1)).toBe(true); // 1-2
    expect(rules.isValidPlay(state, domino3, 1)).toBe(true); // 3-4
  });
});