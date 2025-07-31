import { describe, it, expect } from 'vitest';
import { createInitialState, calculateRoundScore } from '../../game';

describe('Feature: Hand Victory', () => {
  describe('Scenario: Defending Team Wins', () => {
    it('should set the bidders when bidding team fails to take enough points', () => {
      // Given a hand has been played
      const gameState = createInitialState();
      gameState.phase = 'scoring';
      gameState.winningBidder = 0; // Player 0 (team 0) bid
      gameState.currentBid = { type: 'points', value: 40, player: 0 };
      gameState.teamScores = [38, 4]; // Team 0: 38 points, Team 1: 4 points (42 - 38 = 4)
      gameState.teamMarks = [0, 0];
      
      // When calculating round score
      const newMarks = calculateRoundScore(gameState);
      
      // Then the defending team wins by "setting" the bidders
      const biddingTeamMadeBid = gameState.teamScores[0] >= 40;
      expect(biddingTeamMadeBid).toBe(false);
      expect(gameState.teamScores[0]).toBeLessThan(40);
      expect(gameState.teamScores[0] + gameState.teamScores[1]).toBe(42); // Total points always equals 42
      
      // In tournament play, the defending team receives marks equal to what was bid
      expect(newMarks[0]).toBe(0); // Bidding team gets 0
      expect(newMarks[1]).toBe(1); // Defending team gets 1 mark for setting 40 point bid
    });

    it('should award defending team marks when setting a mark bid', () => {
      // Given a hand has been played with a 2 mark bid (84 points)
      const gameState = createInitialState();
      gameState.phase = 'scoring';
      gameState.winningBidder = 1; // Player 1 (team 1) bid
      gameState.currentBid = { type: 'marks', value: 2, player: 1 };
      gameState.teamScores = [1, 41]; // Team 1 got 41 points (just missed the 42 needed for 1 mark)
      gameState.teamMarks = [0, 0];
      
      // When calculating round score
      const newMarks = calculateRoundScore(gameState);
      
      // Then the defending team wins and receives 2 marks
      const biddingTeamMadeBid = gameState.teamScores[1] >= 42; // Need at least 42 for 1 mark when bidding marks
      expect(biddingTeamMadeBid).toBe(false);
      expect(newMarks[1]).toBe(0); // Bidding team gets 0
      expect(newMarks[0]).toBe(2); // Defending team gets 2 marks
    });

    it('should correctly determine set on various bid levels', () => {
      const testCases = [
        { bid: { type: 'points' as const, value: 30 }, biddingTeamPoints: 29, expectedBidderMarks: 0, expectedDefenderMarks: 1 },
        { bid: { type: 'points' as const, value: 35 }, biddingTeamPoints: 34, expectedBidderMarks: 0, expectedDefenderMarks: 1 },
        { bid: { type: 'points' as const, value: 41 }, biddingTeamPoints: 40, expectedBidderMarks: 0, expectedDefenderMarks: 1 },
        { bid: { type: 'marks' as const, value: 1 }, biddingTeamPoints: 41, expectedBidderMarks: 0, expectedDefenderMarks: 1 },
        { bid: { type: 'marks' as const, value: 2 }, biddingTeamPoints: 41, expectedBidderMarks: 0, expectedDefenderMarks: 2 },
        { bid: { type: 'points' as const, value: 30 }, biddingTeamPoints: 30, expectedBidderMarks: 1, expectedDefenderMarks: 0 },
        { bid: { type: 'points' as const, value: 35 }, biddingTeamPoints: 40, expectedBidderMarks: 1, expectedDefenderMarks: 0 },
        { bid: { type: 'marks' as const, value: 1 }, biddingTeamPoints: 42, expectedBidderMarks: 1, expectedDefenderMarks: 0 },
      ];

      testCases.forEach(({ bid, biddingTeamPoints, expectedBidderMarks, expectedDefenderMarks }) => {
        const gameState = createInitialState();
        gameState.phase = 'scoring';
        gameState.winningBidder = 0; // Team 0 bid
        gameState.currentBid = { ...bid, player: 0 };
        gameState.teamScores = [biddingTeamPoints, 42 - biddingTeamPoints];
        gameState.teamMarks = [0, 0];
        
        const newMarks = calculateRoundScore(gameState);
        
        expect(newMarks[0]).toBe(expectedBidderMarks);
        expect(newMarks[1]).toBe(expectedDefenderMarks);
      });
    });
  });
});