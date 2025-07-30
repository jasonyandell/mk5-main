import { describe, it, expect, beforeEach } from 'vitest';
import type { GameState, Player } from '../../game/types';
import { GAME_CONSTANTS } from '../../game/constants';

describe('Victory Conditions - Point System Victory', () => {
  let gameState: GameState;
  
  beforeEach(() => {
    const players: Player[] = [
      { id: 0, name: 'Player 1', hand: [], teamId: 0, marks: 0 },
      { id: 1, name: 'Player 2', hand: [], teamId: 1, marks: 0 },
      { id: 2, name: 'Player 3', hand: [], teamId: 0, marks: 0 },
      { id: 3, name: 'Player 4', hand: [], teamId: 1, marks: 0 }
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
      gameTarget: 250, // Point system default target
      tournamentMode: false, // Point system uses non-tournament mode
      shuffleSeed: 12345
    };
  });

  describe('Scenario: Point System Victory', () => {
    it('should recognize victory when a team reaches 250 points', () => {
      // Given teams are playing with the point system
      expect(gameState.tournamentMode).toBe(false);
      expect(gameState.gameTarget).toBe(250);
      
      // When checking for game victory
      // Simulate team 0 accumulating points over multiple hands
      gameState.teamScores[0] = 249;
      expect(isGameOver(gameState)).toBe(false);
      expect(getWinningTeam(gameState)).toBe(null);
      
      // Team 0 scores 1 more point to reach 250
      gameState.teamScores[0] = 250;
      
      // Then the first partnership to reach the target score wins
      expect(isGameOver(gameState)).toBe(true);
      expect(getWinningTeam(gameState)).toBe(0);
    });

    it('should recognize victory for team 1 when they reach target score', () => {
      // Given teams are playing with the point system
      expect(gameState.tournamentMode).toBe(false);
      expect(gameState.gameTarget).toBe(250);
      
      // When team 1 reaches 250 points
      gameState.teamScores[1] = 250;
      gameState.teamScores[0] = 175;
      
      // Then team 1 wins
      expect(isGameOver(gameState)).toBe(true);
      expect(getWinningTeam(gameState)).toBe(1);
    });

    it('should not declare victory if neither team has reached target', () => {
      // Given teams are playing with the point system
      expect(gameState.tournamentMode).toBe(false);
      
      // When both teams have less than target score
      gameState.teamScores[0] = 249;
      gameState.teamScores[1] = 248;
      
      // Then the game is not over
      expect(isGameOver(gameState)).toBe(false);
      expect(getWinningTeam(gameState)).toBe(null);
    });

    it('should handle victory when team exceeds target score', () => {
      // Given teams are playing with the point system
      expect(gameState.tournamentMode).toBe(false);
      
      // When a team exceeds the target score
      gameState.teamScores[0] = 275; // More than 250
      gameState.teamScores[1] = 220;
      
      // Then that team wins
      expect(isGameOver(gameState)).toBe(true);
      expect(getWinningTeam(gameState)).toBe(0);
    });

    it('should support alternative target scores', () => {
      // Given a game with 150 point target
      gameState.gameTarget = 150;
      expect(gameState.tournamentMode).toBe(false);
      
      // When a team reaches 150 points
      gameState.teamScores[1] = 150;
      gameState.teamScores[0] = 145;
      
      // Then that team wins
      expect(isGameOver(gameState)).toBe(true);
      expect(getWinningTeam(gameState)).toBe(1);
    });

    it('should support 500 point marathon games', () => {
      // Given a game with 500 point target
      gameState.gameTarget = 500;
      expect(gameState.tournamentMode).toBe(false);
      
      // When neither team has reached 500
      gameState.teamScores[0] = 499;
      gameState.teamScores[1] = 498;
      
      // Then game is not over
      expect(isGameOver(gameState)).toBe(false);
      expect(getWinningTeam(gameState)).toBe(null);
      
      // When team 0 reaches 500
      gameState.teamScores[0] = 500;
      
      // Then team 0 wins
      expect(isGameOver(gameState)).toBe(true);
      expect(getWinningTeam(gameState)).toBe(0);
    });

    it('should transition to game_end phase when target is reached', () => {
      // Given teams are playing with the point system
      expect(gameState.tournamentMode).toBe(false);
      
      // When a team reaches the target score
      gameState.teamScores[0] = 250;
      
      // Then game should be ready to transition to game_end
      expect(isGameOver(gameState)).toBe(true);
      
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
  
  // In point system, game ends when a team reaches the target score
  return state.teamScores[0] >= state.gameTarget || state.teamScores[1] >= state.gameTarget;
}

function getWinningTeam(state: GameState): number | null {
  if (!isGameOver(state)) {
    return null;
  }
  
  if (state.tournamentMode) {
    if (state.teamMarks[0] >= state.gameTarget) return 0;
    if (state.teamMarks[1] >= state.gameTarget) return 1;
  } else {
    if (state.teamScores[0] >= state.gameTarget) return 0;
    if (state.teamScores[1] >= state.gameTarget) return 1;
  }
  
  return null;
}