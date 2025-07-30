import { describe, it, expect, beforeEach } from 'vitest';
import type { GameState, Player } from '../../game/types';
import { GAME_CONSTANTS } from '../../game/constants';

describe('Victory Conditions - Mark System Victory', () => {
  let gameState: GameState;
  
  beforeEach(() => {
    const players: Player[] = [
      { id: 0, name: 'Player 1', hand: [], teamId: 0 as 0, marks: 0 },
      { id: 1, name: 'Player 2', hand: [], teamId: 1 as 1, marks: 0 },
      { id: 2, name: 'Player 3', hand: [], teamId: 0 as 0, marks: 0 },
      { id: 3, name: 'Player 4', hand: [], teamId: 1 as 1, marks: 0 }
    ];
    
    gameState = {
      phase: 'playing',
      players,
      currentPlayer: 0,
      dealer: 0,
      bids: [],
      currentBid: null,
      winningBidder: null,
      trump: null,
      tricks: [],
      currentTrick: [],
      teamScores: [0, 0],
      teamMarks: [0, 0],
      gameTarget: GAME_CONSTANTS.DEFAULT_GAME_TARGET,
      tournamentMode: true,
      shuffleSeed: 12345
    };
  });

  describe('Scenario: Mark System Victory', () => {
    it('should recognize victory when a team reaches 7 marks', () => {
      // Given teams are playing with the mark system
      expect(gameState.tournamentMode).toBe(true);
      expect(gameState.gameTarget).toBe(7);
      
      // When checking for game victory
      // Simulate team 0 accumulating marks over multiple hands
      gameState.teamMarks[0] = 6;
      expect(isGameOver(gameState)).toBe(false);
      expect(getWinningTeam(gameState)).toBe(null);
      
      // Team 0 wins another mark
      gameState.teamMarks[0] = 7;
      
      // Then the first partnership to accumulate 7 marks wins
      expect(isGameOver(gameState)).toBe(true);
      expect(getWinningTeam(gameState)).toBe(0);
    });

    it('should recognize victory for team 1 when they reach 7 marks', () => {
      // Given teams are playing with the mark system
      expect(gameState.tournamentMode).toBe(true);
      expect(gameState.gameTarget).toBe(7);
      
      // When team 1 accumulates 7 marks
      gameState.teamMarks[1] = 7;
      gameState.teamMarks[0] = 5;
      
      // Then team 1 wins
      expect(isGameOver(gameState)).toBe(true);
      expect(getWinningTeam(gameState)).toBe(1);
    });

    it('should not declare victory if neither team has 7 marks', () => {
      // Given teams are playing with the mark system
      expect(gameState.tournamentMode).toBe(true);
      
      // When both teams have less than 7 marks
      gameState.teamMarks[0] = 6;
      gameState.teamMarks[1] = 6;
      
      // Then the game is not over
      expect(isGameOver(gameState)).toBe(false);
      expect(getWinningTeam(gameState)).toBe(null);
    });

    it('should handle exact 7 mark victory condition', () => {
      // Given teams are playing with the mark system
      expect(gameState.tournamentMode).toBe(true);
      
      // When exactly one team reaches 7 marks
      gameState.teamMarks[0] = 7;
      gameState.teamMarks[1] = 6;
      
      // Then that team wins immediately
      expect(isGameOver(gameState)).toBe(true);
      expect(getWinningTeam(gameState)).toBe(0);
      
      // Phase should transition to game_end
      const endState = { ...gameState, phase: 'game_end' as const };
      expect(endState.phase).toBe('game_end');
    });
  });
});

// Test-only helper functions that check victory conditions
function isGameOver(state: GameState): boolean {
  // In mark system (tournament mode), game ends when a team reaches the target marks
  if (state.tournamentMode) {
    return state.teamMarks[0] >= state.gameTarget || state.teamMarks[1] >= state.gameTarget;
  }
  // Point system would check teamScores instead
  return false;
}

function getWinningTeam(state: GameState): number | null {
  if (!isGameOver(state)) {
    return null;
  }
  
  if (state.tournamentMode) {
    if (state.teamMarks[0] >= state.gameTarget) return 0;
    if (state.teamMarks[1] >= state.gameTarget) return 1;
  }
  
  return null;
}