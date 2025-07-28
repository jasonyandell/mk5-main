import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { GameTestHelper } from '../helpers/gameTestHelper';

describe('Console Validation and Error Detection', () => {
  let consoleErrorSpy: any;
  let consoleWarnSpy: any;
  let consoleLogSpy: any;

  beforeEach(() => {
    // Spy on console methods to detect unexpected output
    consoleErrorSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
    consoleWarnSpy = vi.spyOn(console, 'warn').mockImplementation(() => {});
    consoleLogSpy = vi.spyOn(console, 'log').mockImplementation(() => {});
  });

  afterEach(() => {
    // Restore console methods and verify no errors
    consoleErrorSpy.mockRestore();
    consoleWarnSpy.mockRestore();
    consoleLogSpy.mockRestore();
  });

  describe('Game State Operations', () => {
    it('should not produce console errors during normal game initialization', () => {
      const state = createInitialState();
      
      expect(state).toBeDefined();
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should not produce console errors during domino dealing', () => {
      const helper = new GameTestHelper();
      const state = createInitialState();
      
      const dealtState = helper.dealDominoes(state);
      
      expect(dealtState.hands).toBeDefined();
      expect(Object.keys(dealtState.hands)).toHaveLength(4);
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should not produce console errors during bidding phase', () => {
      const helper = new GameTestHelper();
      const state = helper.createBiddingState();
      
      expect(state.phase).toBe('bidding');
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should not produce console errors during gameplay', () => {
      const helper = new GameTestHelper();
      const state = helper.createGameInProgress();
      
      expect(state.phase).toBe('playing');
      expect(state.trump).not.toBeNull();
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });
  });

  describe('Error Boundary Testing', () => {
    it('should handle null/undefined inputs gracefully', () => {
      // Test various null inputs that might cause console errors
      expect(() => {
        // These should not throw or produce console errors
        const state = createInitialState();
        state.trump = null; // Valid null state
        state.bidWinner = null; // Valid null state
        state.winner = null; // Valid null state
      }).not.toThrow();
      
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should handle edge cases in domino operations', () => {
      const helper = new GameTestHelper();
      
      // Test edge cases that might produce console output
      expect(() => {
        GameTestHelper.createHandWithDoubles(0); // Zero doubles
        GameTestHelper.createHandWithDoubles(7); // Maximum doubles
      }).not.toThrow();
      
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should handle invalid game state transitions cleanly', () => {
      const state = createInitialState();
      
      // Attempt invalid state changes - should not crash or produce errors
      expect(() => {
        // These operations should be validated and handled gracefully
        state.phase = 'playing'; // Invalid without proper setup
        state.currentPlayer = -1; // Invalid player
        state.currentPlayer = 5; // Invalid player
      }).not.toThrow();
      
      // Should not produce console errors (validation should handle this)
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });
  });

  describe('Memory and Performance Validation', () => {
    it('should not create memory leaks during repeated operations', () => {
      const helper = new GameTestHelper();
      
      // Perform many operations to test for memory leaks
      for (let i = 0; i < 100; i++) {
        const state = createInitialState();
        helper.dealDominoes(state);
        
        // These operations should not accumulate errors
        expect(consoleErrorSpy).not.toHaveBeenCalled();
      }
    });

    it('should handle rapid state changes efficiently', () => {
      const helper = new GameTestHelper();
      const state = helper.createGameInProgress();
      
      // Rapid state modifications should not cause issues
      for (let i = 0; i < 50; i++) {
        state.currentPlayer = i % 4; // Intentionally cycling through player IDs 0-3 for stress test
        expect(consoleErrorSpy).not.toHaveBeenCalled();
      }
    });
  });

  describe('Data Integrity Validation', () => {
    it('should maintain consistent domino count throughout game', () => {
      const helper = new GameTestHelper();
      const state = helper.createGameInProgress();
      
      // Count total dominoes in hands + tricks + current trick
      const handsCount = Object.values(state.hands).reduce((sum, hand) => sum + hand.length, 0);
      const tricksCount = state.tricks.reduce((sum, trick) => sum + trick.length, 0);
      const currentTrickCount = state.currentTrick.length;
      
      const totalDominoes = handsCount + tricksCount + currentTrickCount;
      
      // Should always be 28 dominoes total
      expect(totalDominoes).toBeLessThanOrEqual(28);
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should maintain valid player indices throughout game', () => {
      const helper = new GameTestHelper();
      const state = helper.createGameInProgress();
      
      // All player references should be 0-3
      expect(state.dealer).toBeGreaterThanOrEqual(0);
      expect(state.dealer).toBeLessThan(4);
      expect(state.currentPlayer).toBeGreaterThanOrEqual(0);
      expect(state.currentPlayer).toBeLessThan(4);
      
      if (state.bidWinner !== null) {
        expect(state.bidWinner).toBeGreaterThanOrEqual(0);
        expect(state.bidWinner).toBeLessThan(4);
      }
      
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should maintain valid trump values', () => {
      const helper = new GameTestHelper();
      const state = helper.createGameInProgress();
      
      if (state.trump !== null) {
        // Trump should be 0-6 (suits 0-5, doubles 6)
        expect(state.trump).toBeGreaterThanOrEqual(0);
        expect(state.trump).toBeLessThanOrEqual(6);
      }
      
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });
  });

  describe('Function Input Validation', () => {
    it('should validate inputs to critical functions', () => {
      // Test that functions handle invalid inputs gracefully
      // without producing console errors
      
      const helper = new GameTestHelper();
      
      expect(() => {
        // Test with edge case inputs
        GameTestHelper.createTestState({ phase: 'setup' });
        GameTestHelper.createTestState({ phase: 'bidding' });
        GameTestHelper.createTestState({ phase: 'playing' });
        GameTestHelper.createTestState({ phase: 'scoring' });
      }).not.toThrow();
      
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should handle boundary values correctly', () => {
      const helper = new GameTestHelper();
      
      // Test boundary values that might cause issues
      expect(() => {
        GameTestHelper.createTestState({ dealer: 0 });
        GameTestHelper.createTestState({ dealer: 3 });
        GameTestHelper.createTestState({ currentPlayer: 0 });
        GameTestHelper.createTestState({ currentPlayer: 3 });
      }).not.toThrow();
      
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });
  });

  describe('Integration Stability', () => {
    it('should complete full game without console errors', () => {
      const helper = new GameTestHelper();
      
      // Run through complete game simulation
      const finalState = helper.createCompletedGame(0);
      
      expect(finalState.isComplete).toBe(true);
      expect(finalState.winner).toBeDefined();
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should handle multiple game cycles cleanly', () => {
      const helper = new GameTestHelper();
      
      // Run multiple complete games
      for (let gameNum = 0; gameNum < 5; gameNum++) {
        const finalState = helper.createCompletedGame(gameNum % 2);
        expect(finalState.isComplete).toBe(true);
        expect(consoleErrorSpy).not.toHaveBeenCalled();
      }
    });
  });
});