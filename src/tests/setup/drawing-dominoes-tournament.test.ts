import { describe, expect, test } from 'vitest';
import type { Domino, Player } from '../../game/types';
import { GAME_CONSTANTS } from '../../game/constants';

// Test-only implementation for drawing dominoes
interface DrawingResult {
  nonShakingTeam: Player[];
  shakerPartner: Player;
  shaker: Player;
  remainingDominoes: Domino[];
}

function drawDominoesForPlayers(
  shuffledDominoes: Domino[],
  players: Player[],
  shakerId: number
): DrawingResult {
  // Validate inputs
  if (shuffledDominoes.length !== GAME_CONSTANTS.TOTAL_DOMINOES) {
    throw new Error('Invalid domino set: must have exactly 28 dominoes');
  }
  
  if (players.length !== GAME_CONSTANTS.PLAYERS) {
    throw new Error('Invalid player count: must have exactly 4 players');
  }
  
  // Identify players by their relationship to the shaker
  const shaker = players.find(p => p.id === shakerId);
  if (!shaker) {
    throw new Error(`Shaker with id ${shakerId} not found`);
  }
  
  const shakerTeamId = shaker.teamId;
  const shakerPartner = players.find(p => p.id !== shakerId && p.teamId === shakerTeamId);
  const nonShakingTeam = players.filter(p => p.teamId !== shakerTeamId);
  
  if (!shakerPartner || nonShakingTeam.length !== 2) {
    throw new Error('Invalid team configuration');
  }
  
  let dominoIndex = 0;
  
  // Non-shaking team draws first (7 dominoes each)
  nonShakingTeam.forEach(player => {
    player.hand = shuffledDominoes.slice(dominoIndex, dominoIndex + GAME_CONSTANTS.HAND_SIZE);
    dominoIndex += GAME_CONSTANTS.HAND_SIZE;
  });
  
  // Shaker's partner draws next (7 dominoes)
  shakerPartner.hand = shuffledDominoes.slice(dominoIndex, dominoIndex + GAME_CONSTANTS.HAND_SIZE);
  dominoIndex += GAME_CONSTANTS.HAND_SIZE;
  
  // Shaker draws last (7 dominoes)
  shaker.hand = shuffledDominoes.slice(dominoIndex, dominoIndex + GAME_CONSTANTS.HAND_SIZE);
  dominoIndex += GAME_CONSTANTS.HAND_SIZE;
  
  // Verify all dominoes have been drawn
  const remainingDominoes = shuffledDominoes.slice(dominoIndex);
  
  return {
    nonShakingTeam,
    shakerPartner,
    shaker,
    remainingDominoes
  };
}

describe('Drawing Dominoes (Tournament Standard)', () => {
  // Helper to create a mock domino
  const createDomino = (high: number, low: number): Domino => ({
    high,
    low,
    id: `${high}-${low}`
  });
  
  // Helper to create mock players
  const createPlayers = (): Player[] => [
    { id: 0, name: 'Player 0', hand: [], teamId: 0, marks: 0 },
    { id: 1, name: 'Player 1', hand: [], teamId: 1, marks: 0 },
    { id: 2, name: 'Player 2', hand: [], teamId: 0, marks: 0 },
    { id: 3, name: 'Player 3', hand: [], teamId: 1, marks: 0 }
  ];
  
  // Create a complete set of 28 dominoes
  const createDominoSet = (): Domino[] => {
    const dominoes: Domino[] = [];
    for (let i = 0; i <= 6; i++) {
      for (let j = i; j <= 6; j++) {
        dominoes.push(createDomino(j, i));
      }
    }
    return dominoes;
  };
  
  test('Given the shaker has shuffled dominoes face-down', () => {
    const dominoes = createDominoSet();
    expect(dominoes).toHaveLength(28);
    expect(dominoes[0]).toHaveProperty('high');
    expect(dominoes[0]).toHaveProperty('low');
    expect(dominoes[0]).toHaveProperty('id');
  });
  
  test('When players draw dominoes', () => {
    const shuffledDominoes = createDominoSet();
    const players = createPlayers();
    const shakerId = 0;
    
    const result = drawDominoesForPlayers(shuffledDominoes, players, shakerId);
    
    expect(result).toHaveProperty('nonShakingTeam');
    expect(result).toHaveProperty('shakerPartner');
    expect(result).toHaveProperty('shaker');
    expect(result).toHaveProperty('remainingDominoes');
  });
  
  test('Then the non-shaking team draws first with 7 dominoes each', () => {
    const shuffledDominoes = createDominoSet();
    const players = createPlayers();
    const shakerId = 0; // Player 0 is shaker (team 0)
    
    const result = drawDominoesForPlayers(shuffledDominoes, players, shakerId);
    
    // Non-shaking team is team 1 (players 1 and 3)
    expect(result.nonShakingTeam).toHaveLength(2);
    expect(result.nonShakingTeam[0].teamId).toBe(1);
    expect(result.nonShakingTeam[1].teamId).toBe(1);
    
    // Each non-shaking team player should have 7 dominoes
    result.nonShakingTeam.forEach(player => {
      expect(player.hand).toHaveLength(7);
    });
  });
  
  test('And the shaker\'s partner draws next with 7 dominoes', () => {
    const shuffledDominoes = createDominoSet();
    const players = createPlayers();
    const shakerId = 0; // Player 0 is shaker
    
    const result = drawDominoesForPlayers(shuffledDominoes, players, shakerId);
    
    // Shaker's partner is player 2 (same team as player 0)
    expect(result.shakerPartner.id).toBe(2);
    expect(result.shakerPartner.teamId).toBe(0);
    expect(result.shakerPartner.hand).toHaveLength(7);
  });
  
  test('And the shaker draws last with 7 dominoes', () => {
    const shuffledDominoes = createDominoSet();
    const players = createPlayers();
    const shakerId = 0;
    
    const result = drawDominoesForPlayers(shuffledDominoes, players, shakerId);
    
    expect(result.shaker.id).toBe(0);
    expect(result.shaker.hand).toHaveLength(7);
  });
  
  test('And no dominoes remain', () => {
    const shuffledDominoes = createDominoSet();
    const players = createPlayers();
    const shakerId = 0;
    
    const result = drawDominoesForPlayers(shuffledDominoes, players, shakerId);
    
    expect(result.remainingDominoes).toHaveLength(0);
    
    // Verify all 28 dominoes have been distributed
    const totalDominoesDrawn = 
      result.nonShakingTeam[0].hand.length +
      result.nonShakingTeam[1].hand.length +
      result.shakerPartner.hand.length +
      result.shaker.hand.length;
    
    expect(totalDominoesDrawn).toBe(28);
  });
  
  test('Drawing order is correct for different shaker positions', () => {
    const shuffledDominoes = createDominoSet();
    
    // Test with player 3 as shaker (team 1)
    const players = createPlayers();
    const shakerId = 3;
    
    const result = drawDominoesForPlayers(shuffledDominoes, players, shakerId);
    
    // Non-shaking team should be team 0 (players 0 and 2)
    expect(result.nonShakingTeam.map(p => p.id).sort()).toEqual([0, 2]);
    
    // Shaker's partner should be player 1
    expect(result.shakerPartner.id).toBe(1);
    
    // Shaker should be player 3
    expect(result.shaker.id).toBe(3);
    
    // Verify drawing order by checking which dominoes each player got
    const firstDomino = shuffledDominoes[0];
    const fourteenthDomino = shuffledDominoes[13];
    const fifteenthDomino = shuffledDominoes[14];
    const twentySecondDomino = shuffledDominoes[21];
    
    // First 14 dominoes go to non-shaking team
    const nonShakingTeamDominoes = [
      ...result.nonShakingTeam[0].hand,
      ...result.nonShakingTeam[1].hand
    ];
    expect(nonShakingTeamDominoes).toContainEqual(firstDomino);
    expect(nonShakingTeamDominoes).toContainEqual(fourteenthDomino);
    
    // Next 7 go to shaker's partner
    expect(result.shakerPartner.hand).toContainEqual(fifteenthDomino);
    
    // Last 7 go to shaker
    expect(result.shaker.hand).toContainEqual(twentySecondDomino);
  });
});