import { describe, it, expect } from 'vitest';
import { 
  createInitialState, 
  cloneGameState, 
  validateGameState, 
  isGameComplete, 
  getWinningTeam 
} from '../../game/core/state';
import { GameTestHelper } from '../helpers/gameTestHelper';

describe('Game State Management', () => {
  describe('createInitialState', () => {
    it('should create a valid initial state', () => {
      const state = createInitialState();
      
      expect(state.phase).toBe('bidding');
      expect(state.players).toHaveLength(4);
      expect(state.currentPlayer).toBe(0);
      expect(state.dealer).toBe(3);
      expect(state.bids).toHaveLength(0);
      expect(state.tricks).toHaveLength(0);
      expect(state.currentTrick).toHaveLength(0);
      expect(state.teamScores).toEqual([0, 0]);
      expect(state.teamMarks).toEqual([0, 0]);
      expect(state.gameTarget).toBe(7);
      expect(state.tournamentMode).toBe(true);
    });
    
    it('should deal exactly 28 dominoes total', () => {
      const state = createInitialState();
      const totalDominoes = state.players.reduce((total, player) => 
        total + player.hand.length, 0
      );
      
      expect(totalDominoes).toBe(28);
    });
    
    it('should assign players to correct teams', () => {
      const state = createInitialState();
      
      expect(state.players[0].teamId).toBe(0);
      expect(state.players[1].teamId).toBe(1);
      expect(state.players[2].teamId).toBe(0);
      expect(state.players[3].teamId).toBe(1);
    });
  });
  
  describe('cloneGameState', () => {
    it('should create a deep copy', () => {
      const original = createInitialState();
      const clone = cloneGameState(original);
      
      expect(clone).toEqual(original);
      expect(clone).not.toBe(original);
      expect(clone.players).not.toBe(original.players);
      expect(clone.players[0].hand).not.toBe(original.players[0].hand);
    });
    
    it('should allow independent modifications', () => {
      const original = createInitialState();
      const clone = cloneGameState(original);
      
      clone.currentPlayer = 2;
      clone.players[0].hand.pop();
      
      expect(original.currentPlayer).toBe(0);
      expect(original.players[0].hand.length).toBe(7);
    });
  });
  
  describe('validateGameState', () => {
    it('should validate a correct initial state', () => {
      const state = createInitialState();
      const errors = validateGameState(state);
      
      expect(errors).toHaveLength(0);
    });
    
    it('should detect invalid player count', () => {
      const state = createInitialState();
      state.players = state.players.slice(0, 3);
      
      const errors = validateGameState(state);
      expect(errors).toContain('Game must have exactly 4 players');
    });
    
    it('should detect invalid current player', () => {
      const state = createInitialState();
      state.currentPlayer = 5;
      
      const errors = validateGameState(state);
      expect(errors).toContain('Current player ID out of bounds');
    });
    
    it('should detect invalid dealer', () => {
      const state = createInitialState();
      state.dealer = -1;
      
      const errors = validateGameState(state);
      expect(errors).toContain('Dealer ID out of bounds');
    });
  });
  
  describe('isGameComplete', () => {
    it('should return false for initial state', () => {
      const state = createInitialState();
      expect(isGameComplete(state)).toBe(false);
    });
    
    it('should return true when team 0 reaches target', () => {
      const state = createInitialState();
      state.teamMarks[0] = 7;
      
      expect(isGameComplete(state)).toBe(true);
    });
    
    it('should return true when team 1 reaches target', () => {
      const state = createInitialState();
      state.teamMarks[1] = 7;
      
      expect(isGameComplete(state)).toBe(true);
    });
  });
  
  describe('getWinningTeam', () => {
    it('should return null when game not complete', () => {
      const state = createInitialState();
      expect(getWinningTeam(state)).toBeNull();
    });
    
    it('should return team 0 when they reach target first', () => {
      const state = createInitialState();
      state.teamMarks[0] = 7;
      
      expect(getWinningTeam(state)).toBe(0);
    });
    
    it('should return team 1 when they reach target first', () => {
      const state = createInitialState();
      state.teamMarks[1] = 7;
      
      expect(getWinningTeam(state)).toBe(1);
    });
  });
  
  describe('GameTestHelper validation', () => {
    it('should validate correct game rules', () => {
      const state = createInitialState();
      const errors = GameTestHelper.validateGameRules(state);
      
      expect(errors).toHaveLength(0);
    });
    
    it('should detect hand size violations', () => {
      const state = createInitialState();
      state.players[0].hand.push({ high: 0, low: 0, id: 'extra' });
      
      const errors = GameTestHelper.validateGameRules(state);
      expect(errors.length).toBeGreaterThan(0);
    });
    
    it('should validate tournament rules', () => {
      const state = createInitialState();
      const errors = GameTestHelper.validateTournamentRules(state);
      
      expect(errors).toHaveLength(0);
    });
  });
});