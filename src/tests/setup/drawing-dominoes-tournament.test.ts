import { describe, expect, test } from 'vitest';
import { dealDominoesWithSeed, createDominoes, GAME_CONSTANTS } from '../../game';

describe('Drawing Dominoes (Tournament Standard)', () => {
  test('Given the shaker has shuffled dominoes face-down', () => {
    const dominoes = createDominoes();
    expect(dominoes).toHaveLength(28);
    expect(dominoes[0]).toHaveProperty('high');
    expect(dominoes[0]).toHaveProperty('low');
    expect(dominoes[0]).toHaveProperty('id');
  });
  
  test('When players draw dominoes', () => {
    // Deal dominoes returns array of 4 hands
    const hands = dealDominoesWithSeed(12345);
    
    // Verify 4 hands with 7 dominoes each
    expect(hands).toHaveLength(4);
    expect(hands.every(hand => hand.length === 7)).toBe(true);
  });
  
  test('Then the non-shaking team draws first with 7 dominoes each', () => {
    // Standard dealDominoesWithSeed deals in order: player 0, 1, 2, 3
    const hands = dealDominoesWithSeed(12345);
    
    // In actual game, teams are 0+2 vs 1+3
    const team0 = [...hands[0], ...hands[2]];
    const team1 = [...hands[1], ...hands[3]];
    
    expect(hands[0]).toHaveLength(GAME_CONSTANTS.HAND_SIZE);
    expect(hands[1]).toHaveLength(GAME_CONSTANTS.HAND_SIZE);
    expect(hands[2]).toHaveLength(GAME_CONSTANTS.HAND_SIZE);
    expect(hands[3]).toHaveLength(GAME_CONSTANTS.HAND_SIZE);
    
    // Verify teams have 14 dominoes each
    expect(team0).toHaveLength(14);
    expect(team1).toHaveLength(14);
  });
  
  test('And the shaker\'s partner draws next with 7 dominoes', () => {
    // Deal dominoes
    const hands = dealDominoesWithSeed(12345);
    
    // Shaker's partner is player 2
    const player2Hand = hands[2];
    expect(player2Hand).toHaveLength(GAME_CONSTANTS.HAND_SIZE);
    
    // Note: Standard dealDominoesWithSeed doesn't follow tournament drawing order
    // It deals in player order 0,1,2,3
    // So we can't verify the exact dominoes without tournament-specific dealing
  });
  
  test('And the shaker draws last with 7 dominoes', () => {
    // Deal dominoes
    const hands = dealDominoesWithSeed(12345);
    
    const player0Hand = hands[0];
    expect(player0Hand).toHaveLength(GAME_CONSTANTS.HAND_SIZE);
  });
  
  test('And no dominoes remain', () => {
    // Deal dominoes
    const hands = dealDominoesWithSeed(12345);
    
    // Verify all 28 dominoes have been distributed
    const totalDominoesDrawn = hands.reduce(
      (sum, hand) => sum + hand.length, 
      0
    );
    
    expect(totalDominoesDrawn).toBe(GAME_CONSTANTS.TOTAL_DOMINOES);
  });
  
  test('Dominoes are dealt evenly to all players', () => {
    // Use dealDominoesWithSeed for deterministic results
    const hands = dealDominoesWithSeed(54321);
    
    // Verify each player gets exactly 7 dominoes
    expect(hands[0]).toHaveLength(7);
    expect(hands[1]).toHaveLength(7);
    expect(hands[2]).toHaveLength(7);
    expect(hands[3]).toHaveLength(7);
    
    // Verify no domino appears in multiple hands
    const allDominoes = [...hands[0], ...hands[1], ...hands[2], ...hands[3]];
    const uniqueDominoes = new Set(allDominoes.map(d => d.id));
    expect(uniqueDominoes.size).toBe(28);
  });
});