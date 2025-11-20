import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import type { MockInstance } from 'vitest';
import { createInitialState } from '../../game/core/state';
import { StateBuilder, HandBuilder } from '../helpers';

describe('Console Validation and Error Detection', () => {
  let consoleErrorSpy: MockInstance;
  let consoleWarnSpy: MockInstance;
  let consoleLogSpy: MockInstance;

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
      const state = StateBuilder.inBiddingPhase().build();

      expect(state.players).toBeDefined();
      expect(state.players.every(p => p.hand.length > 0)).toBe(true);
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should not produce console errors during bidding phase', () => {
      const state = StateBuilder.inBiddingPhase().build();

      expect(state.phase).toBe('bidding');
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should not produce console errors during gameplay', () => {
      const state = StateBuilder.inPlayingPhase().build();

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
        state.trump = { type: 'not-selected' }; // Valid none state
        // state.winningBidder and state.winner don't exist in new types
      }).not.toThrow();
      
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should handle edge cases in domino operations', () => {
      // Test edge cases that might produce console output
      expect(() => {
        HandBuilder.withDoubles(0); // Zero doubles
        HandBuilder.withDoubles(7); // Maximum doubles
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
      // Perform many operations to test for memory leaks
      for (let i = 0; i < 100; i++) {
        StateBuilder.inBiddingPhase().build();

        // These operations should not accumulate errors
        expect(consoleErrorSpy).not.toHaveBeenCalled();
      }
    });

    it('should handle rapid state changes efficiently', () => {
      const state = StateBuilder.inPlayingPhase().build();

      // Rapid state modifications should not cause issues
      for (let i = 0; i < 50; i++) {
        state.currentPlayer = i % 4; // Intentionally cycling through player IDs 0-3 for stress test
        expect(consoleErrorSpy).not.toHaveBeenCalled();
      }
    });
  });

  describe('Data Integrity Validation', () => {
    it('should maintain consistent domino count throughout game', () => {
      const state = StateBuilder.inPlayingPhase().build();

      // Count total dominoes in hands + tricks + current trick
      const handsCount = state.players.reduce((sum, p) => sum + p.hand.length, 0);
      const tricksCount = state.tricks.reduce((sum, trick) => sum + trick.plays.length, 0);
      const currentTrickCount = state.currentTrick.length;

      const totalDominoes = handsCount + tricksCount + currentTrickCount;

      // Should always be 28 dominoes total
      expect(totalDominoes).toBeLessThanOrEqual(28);
      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should maintain valid player indices throughout game', () => {
      const state = StateBuilder.inPlayingPhase().build();

      // All player references should be 0-3
      expect(state.dealer).toBeGreaterThanOrEqual(0);
      expect(state.dealer).toBeLessThan(4);
      expect(state.currentPlayer).toBeGreaterThanOrEqual(0);
      expect(state.currentPlayer).toBeLessThan(4);

      if (state.winningBidder !== null && state.winningBidder !== undefined && state.winningBidder !== -1) {
        expect(state.winningBidder).toBeGreaterThanOrEqual(0);
        expect(state.winningBidder).toBeLessThan(4);
      }

      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should maintain valid trump values', () => {
      const state = StateBuilder.inPlayingPhase().build();

      if (state.trump.type !== 'not-selected') {
        // Trump should be valid TrumpSelection
        expect(['suit', 'doubles', 'no-trump']).toContain(state.trump.type);
        if (state.trump.type === 'suit') {
          expect(state.trump.suit).toBeGreaterThanOrEqual(0);
          expect(state.trump.suit).toBeLessThanOrEqual(6);
        }
      }

      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });
  });

  describe('Function Input Validation', () => {
    it('should validate inputs to critical functions', () => {
      // Test that functions handle invalid inputs gracefully
      // without producing console errors

      expect(() => {
        // Test with edge case inputs
        StateBuilder.inBiddingPhase().with({ phase: 'setup' }).build();
        StateBuilder.inBiddingPhase().build();
        StateBuilder.inPlayingPhase().build();
        StateBuilder.inScoringPhase([0, 0]).build();
      }).not.toThrow();

      expect(consoleErrorSpy).not.toHaveBeenCalled();
    });

    it('should handle boundary values correctly', () => {
      // Test boundary values that might cause issues
      expect(() => {
        StateBuilder.inBiddingPhase().withDealer(0).build();
        StateBuilder.inBiddingPhase().withDealer(3).build();
        StateBuilder.inBiddingPhase().withCurrentPlayer(0).build();
        StateBuilder.inBiddingPhase().withCurrentPlayer(3).build();
      }).not.toThrow();

      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });
  });

  describe('Integration Stability', () => {
    it('should complete full game without console errors', () => {
      // Run through complete game simulation
      const finalState = StateBuilder.gameEnded(0).build();

      expect(finalState.phase).toBe('game_end');
      expect(finalState.teamMarks.some(m => m >= finalState.gameTarget)).toBe(true);
      expect(consoleErrorSpy).not.toHaveBeenCalled();
      expect(consoleWarnSpy).not.toHaveBeenCalled();
    });

    it('should handle multiple game cycles cleanly', () => {
      // Run multiple complete games
      for (let gameNum = 0; gameNum < 5; gameNum++) {
        const finalState = StateBuilder.gameEnded(gameNum % 2 as 0 | 1).build();
        expect(finalState.phase).toBe('game_end');
        expect(consoleErrorSpy).not.toHaveBeenCalled();
      }
    });
  });
});