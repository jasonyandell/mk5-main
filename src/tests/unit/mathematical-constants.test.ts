import { describe, it, expect } from 'vitest';
import { MathematicalVerification } from '../helpers/mathematicalVerification';
import { createDominoes, getDominoPoints } from '../../game/core/dominoes';
import { createInitialState } from '../../game/core/state';

describe('Mathematical Constants Verification', () => {
  describe('Domino Set Constants', () => {
    it('should have exactly 35 total points (mk4 rules)', () => {
      expect(MathematicalVerification.verifyTotalPoints()).toBe(true);
    });

    it('should have exactly 28 dominoes', () => {
      expect(MathematicalVerification.verifyDominoCount()).toBe(true);
    });

    it('should have no duplicate dominoes', () => {
      expect(MathematicalVerification.verifyNoDuplicates()).toBe(true);
    });

    it('should have complete domino set (0-0 through 6-6)', () => {
      expect(MathematicalVerification.verifyCompleteSet()).toBe(true);
    });

    it('should have exactly 5 dominoes with points', () => {
      expect(MathematicalVerification.verifyPointDominoCount()).toBe(true);
    });

    it('should include all high-value dominoes', () => {
      expect(MathematicalVerification.verifyHighValueDominoes()).toBe(true);
    });
  });

  describe('Point Distribution Verification', () => {
    it('should have correct point values for specific dominoes', () => {
      const dominoSet = createDominoes();
      
      // Find and verify specific high-value dominoes
      const doubleBlank = dominoSet.find(d => d.high === 0 && d.low === 0);
      const doubleSix = dominoSet.find(d => d.high === 6 && d.low === 6);
      const sixFour = dominoSet.find(d => 
        (d.high === 6 && d.low === 4) || (d.high === 4 && d.low === 6)
      );
      const fiveBlank = dominoSet.find(d => 
        (d.high === 5 && d.low === 0) || (d.high === 0 && d.low === 5)
      );
      
      expect(doubleBlank).toBeDefined();
      expect(doubleSix).toBeDefined();
      expect(sixFour).toBeDefined();
      expect(fiveBlank).toBeDefined();
      
      if (doubleBlank) expect(getDominoPoints(doubleBlank)).toBe(0);
      if (doubleSix) expect(getDominoPoints(doubleSix)).toBe(0); // 6-6 = 0 points in mk4 rules
      if (sixFour) expect(getDominoPoints(sixFour)).toBe(10);
      if (fiveBlank) expect(getDominoPoints(fiveBlank)).toBe(5);
    });

    it('should have correct distribution of point values', () => {
      const distribution = MathematicalVerification.verifyPointDistribution();
      
      // Should have dominoes worth 0, 5, and 10 points only
      expect(distribution[0]).toBeGreaterThan(0); // Many 0-point dominoes
      expect(distribution[5]).toBe(3); // Three 5-point dominoes (5-0, 4-1, 3-2)
      expect(distribution[10]).toBe(2); // Two 10-point dominoes (5-5, 6-4)
      
      // Total should be 5 point-bearing dominoes (3 five-point + 2 ten-point)
      const totalPointDominoes = Object.entries(distribution)
        .filter(([points]) => parseInt(points) > 0)
        .reduce((sum, [, count]) => sum + count, 0);
      expect(totalPointDominoes).toBe(5);
    });

    it('should validate individual domino point calculations', () => {
      const testCases = [
        { high: 0, low: 0, expected: 0 },   // Double blank (0 total)
        { high: 6, low: 6, expected: 0 },   // Double six (12 total, no points in mk4)
        { high: 5, low: 5, expected: 10 },  // Double five (special 10 points)
        { high: 6, low: 4, expected: 10 },  // Six-four (special 10 points)
        { high: 5, low: 0, expected: 5 },   // Five-blank (5 total = 5 points)
        { high: 4, low: 1, expected: 5 },   // Four-one (5 total = 5 points)
        { high: 3, low: 2, expected: 5 },   // Three-two (5 total = 5 points)
        { high: 2, low: 3, expected: 5 },   // Two-three (5 total = 5 points)
        { high: 1, low: 2, expected: 0 },   // One-two (3 total, no points)
        { high: 3, low: 4, expected: 0 },   // Three-four (7 total, no points)
      ];
      
      testCases.forEach(({ high, low, expected }) => {
        const domino = { id: `${high}-${low}`, high, low };
        expect(getDominoPoints(domino)).toBe(expected);
      });
    });
  });

  describe('Game State Mathematical Validation', () => {
    it('should maintain mathematical consistency in initial state', () => {
      const state = createInitialState();
      expect(MathematicalVerification.verifyGameStateConsistency(state)).toBe(true);
    });

    it('should verify correct team assignments', () => {
      const state = createInitialState();
      
      // Players 0 and 2 on team 0
      expect(state.players[0]!.teamId).toBe(0);
      expect(state.players[2]!.teamId).toBe(0);
      
      // Players 1 and 3 on team 1  
      expect(state.players[1]!.teamId).toBe(1);
      expect(state.players[3]!.teamId).toBe(1);
    });

    it('should verify initial domino distribution', () => {
      const state = createInitialState();
      
      // Each player has 7 dominoes
      state.players.forEach(player => {
        expect(player.hand).toHaveLength(7);
      });
      
      // Total of 28 dominoes
      const totalDominoes = state.players.reduce(
        (sum, player) => sum + player.hand.length, 0
      );
      expect(totalDominoes).toBe(28);
      
      // All dominoes are unique
      const allDominoes = state.players.flatMap(player => player.hand);
      const dominoIds = allDominoes.map(d => d.id);
      const uniqueIds = new Set(dominoIds);
      expect(uniqueIds.size).toBe(28);
    });

    it('should verify point conservation', () => {
      const state = createInitialState();
      
      // Calculate total points in all hands
      let totalPoints = 0;
      state.players.forEach(player => {
        player.hand.forEach(domino => {
          totalPoints += getDominoPoints(domino);
        });
      });
      
      expect(totalPoints).toBe(35);
    });

    it('should verify mathematical constraints throughout gameplay', () => {
      const state = createInitialState();
      
      // Current trick starts empty
      expect(state.currentTrick).toHaveLength(0);
      
      // No completed tricks initially
      expect(state.tricks).toHaveLength(0);
      
      // Initial scores are zero
      expect(state.teamScores).toEqual([0, 0]);
      expect(state.teamMarks).toEqual([0, 0]);
      
      // Game target is reasonable
      expect(state.gameTarget).toBeGreaterThan(0);
      expect(state.gameTarget).toBeLessThanOrEqual(21); // Reasonable maximum
    });
  });

  describe('Comprehensive Mathematical Verification', () => {
    it('should pass all mathematical verification checks', () => {
      const results = MathematicalVerification.runFullVerification();
      
      expect(results.totalPoints).toBe(true);
      expect(results.dominoCount).toBe(true);
      expect(results.noDuplicates).toBe(true);
      expect(results.completeSet).toBe(true);
      expect(results.pointDominoCount).toBe(true);
      expect(results.highValueDominoes).toBe(true);
      expect(results.gameStateConsistency).toBe(true);
      expect(results.trickConsistency).toBe(true);
      expect(results.biddingConstraints).toBe(true);
    });

    it('should maintain mathematical invariants', () => {
      // Test multiple state generations for consistency
      for (let i = 0; i < 10; i++) {
        const state = createInitialState();
        expect(MathematicalVerification.verifyGameStateConsistency(state)).toBe(true);
        
        // Verify point total remains 42
        let totalPoints = 0;
        state.players.forEach(player => {
          player.hand.forEach(domino => {
            totalPoints += getDominoPoints(domino);
          });
        });
        expect(totalPoints).toBe(35);
      }
    });
  });
});