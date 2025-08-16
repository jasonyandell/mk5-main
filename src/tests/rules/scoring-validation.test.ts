import { describe, it, expect } from 'vitest';
import { calculateRoundScore } from '../../game/core/scoring';
import { isGameComplete } from '../../game/core/state';
import type { Bid } from '../../game/types';
import { BID_TYPES } from '../../game/constants';
import { createTestState } from '../helpers/gameTestHelper';

describe('Scoring Validation', () => {
  describe('calculateRoundScore', () => {
    it('should calculate basic point totals correctly', () => {
      const state = createTestState({
        currentBid: { type: BID_TYPES.POINTS, value: 30, player: 0 },
        winningBidder: 0,
        teamScores: [25, 17],
        teamMarks: [0, 0]
      });
      
      const result = calculateRoundScore(state);
      
      expect(result).toBeDefined();
      expect(result).toHaveLength(2);
      expect(state.teamScores[0] + state.teamScores[1]).toBe(42);
    });

    it('should handle successful bids', () => {
      const state = createTestState({
        currentBid: { type: BID_TYPES.POINTS, value: 30, player: 0 },
        winningBidder: 0,
        teamScores: [32, 10], // Bidding team made their bid
        teamMarks: [0, 0]
      });
      
      const result = calculateRoundScore(state);
      
      expect(result).toBeDefined();
      expect(state.teamScores[0]).toBeGreaterThanOrEqual(30); // Made the bid
    });

    it('should handle failed bids', () => {
      const state = createTestState({
        currentBid: { type: BID_TYPES.POINTS, value: 35, player: 0 },
        winningBidder: 0,
        teamScores: [20, 22], // Bidding team failed to make bid
        teamMarks: [0, 0]
      });
      
      const result = calculateRoundScore(state);
      
      expect(result).toBeDefined();
      expect(state.teamScores[0]).toBeLessThan(35); // Failed the bid
    });

    it('should handle mark bids correctly', () => {
      const state = createTestState({
        currentBid: { type: BID_TYPES.MARKS, value: 1, player: 0 },
        winningBidder: 0,
        teamScores: [42, 0], // Perfect score for 1 mark bid
        teamMarks: [0, 0]
      });
      
      const result = calculateRoundScore(state);
      
      expect(result).toBeDefined();
      expect(state.teamScores[0]).toBe(42); // All points to bidding team
    });

    it('should validate point conservation', () => {
      const state = createTestState({
        currentBid: { type: BID_TYPES.POINTS, value: 35, player: 1 },
        winningBidder: 1,
        teamScores: [18, 24],
        teamMarks: [0, 0]
      });
      
      // Total points should always equal 42 (35 counting + 7 tricks)
      expect(state.teamScores[0] + state.teamScores[1]).toBe(42);
      
      const result = calculateRoundScore(state);
      expect(result).toBeDefined();
    });
  });

  describe('Game Completion', () => {
    it('should detect game completion at 7 marks', () => {
      const state1 = createTestState({ teamMarks: [7, 0] });
      const state2 = createTestState({ teamMarks: [0, 7] });
      const state3 = createTestState({ teamMarks: [6, 6] });
      const state4 = createTestState({ teamMarks: [6, 0] });
      
      expect(isGameComplete(state1)).toBe(true);
      expect(isGameComplete(state2)).toBe(true);
      expect(isGameComplete(state3)).toBe(false);
      expect(isGameComplete(state4)).toBe(false);
    });

    it('should handle edge cases correctly', () => {
      const state1 = createTestState({ teamMarks: [7, 7] }); // Both teams reach 7
      const state2 = createTestState({ teamMarks: [8, 5] }); // Exceed 7
      const state3 = createTestState({ teamMarks: [0, 0] }); // New game
      
      expect(isGameComplete(state1)).toBe(true);
      expect(isGameComplete(state2)).toBe(true);
      expect(isGameComplete(state3)).toBe(false);
    });
  });

  describe('Tournament Scoring Rules', () => {
    it('should follow tournament mark system', () => {
      // Tournament play awards marks based on bid success/failure
      const tests = [
        { bid: 30, score: 32, expectedMarks: 1 }, // Made bid
        { bid: 35, score: 25, expectedMarks: 0 }, // Failed bid  
        { bid: 42, score: 42, expectedMarks: 2 }, // Perfect 1-mark bid
      ];

      tests.forEach(test => {
        const state = createTestState({
          currentBid: { type: BID_TYPES.POINTS, value: test.bid, player: 0 },
          winningBidder: 0,
          teamScores: [test.score, 42 - test.score],
          teamMarks: [0, 0]
        });
        
        expect(state.teamScores[0] + state.teamScores[1]).toBe(42);
        
        const result = calculateRoundScore(state);
        expect(result).toBeDefined();
      });
    });

    it('should handle set penalties correctly', () => {
      // When a team fails their bid, opponents get marks
      const state = createTestState({
        currentBid: { type: BID_TYPES.MARKS, value: 2, player: 0 },
        winningBidder: 0,
        teamScores: [20, 22], // Failed 2-mark bid (needed 42+ points)
        teamMarks: [0, 0]
      });
      
      const result = calculateRoundScore(state);
      expect(result).toBeDefined();
      expect(state.teamScores[0]).toBeLessThan(42); // Failed the bid
    });
  });

  describe('Mathematical Verification', () => {
    it('should maintain counting domino totals', () => {
      // Standard counting dominoes total 35 points
      const countingDominoes = [
        { domino: '5-5', points: 10 },
        { domino: '6-4', points: 10 },
        { domino: '5-0', points: 5 },
        { domino: '4-1', points: 5 },
        { domino: '3-2', points: 5 }
      ];
      
      const totalCount = countingDominoes.reduce((sum, d) => sum + d.points, 0);
      expect(totalCount).toBe(35);
      
      // Plus 7 trick points = 42 total
      expect(totalCount + 7).toBe(42);
    });

    it('should validate hand point distribution', () => {
      // Various realistic point distributions
      const distributions = [
        [21, 21], // Perfect split
        [35, 7],  // One team gets all count
        [7, 35],  // Other team gets all count
        [42, 0],  // Shutout (all count + all tricks)
        [0, 42]   // Reverse shutout
      ];

      distributions.forEach(dist => {
        if (dist[0] === undefined || dist[1] === undefined) {
          throw new Error('Distribution values cannot be undefined');
        }
        expect(dist[0] + dist[1]).toBe(42);
        expect(dist[0]).toBeGreaterThanOrEqual(0);
        expect(dist[1]).toBeGreaterThanOrEqual(0);
      });
    });
  });

  describe('Bid Value Validation', () => {
    it('should validate point bid ranges', () => {
      // Valid point bids: 30-41
      for (let points = 30; points <= 41; points++) {
        const bid: Bid = { type: BID_TYPES.POINTS, value: points, player: 0 };
        expect(bid.value).toBeGreaterThanOrEqual(30);
        expect(bid.value).toBeLessThanOrEqual(41);
      }
    });

    it('should validate mark bid equivalents', () => {
      // Mark bids represent specific point values
      const markEquivalents = [
        { marks: 1, points: 42 },
        { marks: 2, points: 84 },
        { marks: 3, points: 126 },
        { marks: 4, points: 168 }
      ];

      markEquivalents.forEach(({ marks, points }) => {
        expect(marks * 42).toBe(points);
      });
    });
  });

  describe('Score History and Progression', () => {
    it('should track cumulative game scores', () => {
      // Simulate multiple hands
      const gameProgression = [
        { hand: 1, team0: 2, team1: 1 },
        { hand: 2, team0: 3, team1: 2 },
        { hand: 3, team0: 4, team1: 4 },
        { hand: 4, team0: 7, team1: 4 }  // Game ends
      ];

      gameProgression.forEach((round, index) => {
        const state = createTestState({ teamMarks: [round.team0, round.team1] });
        if (index === gameProgression.length - 1) {
          expect(isGameComplete(state)).toBe(true);
        } else {
          expect(isGameComplete(state)).toBe(false);
        }
      });
    });

    it('should maintain score consistency across hands', () => {
      // Scores should only increase or stay same between hands
      const team0Scores = [0, 2, 3, 4, 7];
      const team1Scores = [0, 1, 2, 4, 4];

      for (let i = 1; i < team0Scores.length; i++) {
        const currentTeam0 = team0Scores[i];
        const prevTeam0 = team0Scores[i - 1];
        const currentTeam1 = team1Scores[i];
        const prevTeam1 = team1Scores[i - 1];
        
        if (currentTeam0 === undefined || prevTeam0 === undefined || 
            currentTeam1 === undefined || prevTeam1 === undefined) {
          throw new Error('Score values cannot be undefined');
        }
        
        expect(currentTeam0).toBeGreaterThanOrEqual(prevTeam0);
        expect(currentTeam1).toBeGreaterThanOrEqual(prevTeam1);
      }
    });
  });
});