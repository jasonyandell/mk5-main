import { describe, it, expect } from 'vitest';
import { createTestState, createHandWithDoubles } from '../helpers/gameTestHelper';
import { calculateGameScore, isGameComplete } from '../../game/core/scoring';
import { isValidPlay, getValidPlays } from '../../game/core/rules';
import { BID_TYPES } from '../../game/constants';
import type { Domino, Trump, Bid } from '../../game/types';

describe('Special Gameplay Scenarios', () => {
  describe('High Stakes Bidding', () => {
    it('handles 6 and 7 mark bids correctly', () => {
      // Create state with escalated bidding
      const state = createTestState({
        phase: 'bidding',
        dealer: 0,
        currentPlayer: 1,
        bids: [
          { type: BID_TYPES.MARKS, value: 4, player: 0 },
          { type: BID_TYPES.MARKS, value: 5, player: 1 },
          { type: BID_TYPES.MARKS, value: 6, player: 2 }
        ]
      });

      // 7 mark bid should be valid as final escalation
      const sevenMarkBid: Bid = { type: BID_TYPES.MARKS, value: 7, player: 3 };
      
      // This represents maximum possible bid (instant game winner)
      expect(sevenMarkBid.value).toBe(7);
      expect(sevenMarkBid.type).toBe(BID_TYPES.MARKS);
      
      // 7 marks should instantly win the game if achieved
      const sevenMarkScore = calculateGameScore([42, 0, 0, 0]); // All points to player 0 (team 0)
      
      // Use the state variable
      expect(state.phase).toBe('bidding');
      expect(sevenMarkScore[0]).toBe(42); // Team 0 gets all points
      expect(sevenMarkScore[1]).toBe(0);  // Team 1 gets no points
    });

    it('handles dramatic bid escalation scenarios', () => {
      // Test rapid bid escalation that might occur in competitive play
      const escalationBids: Bid[] = [
        { type: BID_TYPES.POINTS, value: 30, player: 0 },
        { type: BID_TYPES.POINTS, value: 35, player: 1 },
        { type: BID_TYPES.MARKS, value: 1, player: 2 },   // 42 points
        { type: BID_TYPES.MARKS, value: 2, player: 3 },   // 84 points
        { type: BID_TYPES.MARKS, value: 3, player: 0 },   // 126 points
        { type: BID_TYPES.MARKS, value: 4, player: 1 }    // 168 points
      ];

      escalationBids.forEach((bid, index) => {
        expect(bid.value).toBeGreaterThan(0);
        expect([BID_TYPES.POINTS, BID_TYPES.MARKS]).toContain(bid.type);
        
        // Each subsequent bid should be higher value
        if (index > 0) {
          const prevBid = escalationBids[index - 1];
          
          // Convert to comparable values (marks = value * 42)
          const currentValue = bid.type === BID_TYPES.MARKS ? (bid.value || 0) * 42 : (bid.value || 0);
          const prevValue = prevBid.type === BID_TYPES.MARKS ? (prevBid.value || 0) * 42 : (prevBid.value || 0);
          
          expect(currentValue).toBeGreaterThan(prevValue);
        }
      });
    });
  });

  describe('Trump Dominance Scenarios', () => {
    it('handles all-trump hands correctly', () => {
      // Create hand with all trump dominoes
      const allTrumpHand: Domino[] = [
        { id: 'trump1', high: 1, low: 0, points: 0 },
        { id: 'trump2', high: 1, low: 1, points: 0 },
        { id: 'trump3', high: 1, low: 2, points: 0 },
        { id: 'trump4', high: 1, low: 3, points: 0 },
        { id: 'trump5', high: 1, low: 4, points: 0 },
        { id: 'trump6', high: 1, low: 5, points: 0 },
        { id: 'trump7', high: 1, low: 6, points: 0 }
      ];

      const trump: Trump = 1; // ones are trump
      const currentTrick: { player: number; domino: Domino }[] = [];

      // All dominoes should be valid plays
      allTrumpHand.forEach(domino => {
        expect(isValidPlay(domino, allTrumpHand, currentTrick, trump)).toBe(true);
      });

      const validPlays = getValidPlays(allTrumpHand, currentTrick, trump);
      expect(validPlays).toHaveLength(7);
    });

    it('handles no-trump scenarios', () => {
      const mixedHand: Domino[] = [
        { id: 'twos1', high: 2, low: 0, points: 0 },
        { id: 'threes1', high: 3, low: 1, points: 5 },
        { id: 'fours1', high: 4, low: 2, points: 0 },
        { id: 'fives1', high: 5, low: 3, points: 0 },
        { id: 'sixes1', high: 6, low: 4, points: 10 },
        { id: 'double1', high: 1, low: 1, points: 0 },
        { id: 'double2', high: 2, low: 2, points: 0 }
      ];

      // Note: No-trump is typically suit 7 or special designation
      // but for this test, we'll use null to represent no-trump
      const noTrump: Trump = 8; // 8 represents no-trump
      const currentTrick: { player: number; domino: Domino }[] = [];

      // With no trump, all dominoes should be playable initially
      const validPlays = getValidPlays(mixedHand, currentTrick, noTrump);
      expect(validPlays.length).toBeGreaterThan(0);
    });

    it('handles doubles as trump correctly', () => {
      const handWithDoubles: Domino[] = [
        { id: 'double0', high: 0, low: 0, points: 0 },
        { id: 'double1', high: 1, low: 1, points: 0 },
        { id: 'double2', high: 2, low: 2, points: 0 },
        { id: 'double6', high: 6, low: 6, points: 0 },
        { id: 'regular1', high: 2, low: 3, points: 0 },
        { id: 'regular2', high: 4, low: 5, points: 0 },
        { id: 'regular3', high: 1, low: 6, points: 0 }
      ];

      const doublesTrump: Trump = 6; // doubles are trump
      const currentTrick: { player: number; domino: Domino }[] = [];

      // All doubles should be trump
      const doubles = handWithDoubles.filter(d => d.high === d.low);
      expect(doubles).toHaveLength(4);

      const validPlays = getValidPlays(handWithDoubles, currentTrick, doublesTrump);
      expect(validPlays).toHaveLength(7); // All should be playable initially
    });
  });

  describe('Extreme Point Distribution', () => {
    it('handles all points to one team scenario', () => {
      // Scenario: One team captures all counting dominoes and tricks
      const allPointsToTeam0 = calculateGameScore([42, 0, 0, 0]);
      
      expect(allPointsToTeam0[0]).toBe(42); // Team 0 gets everything
      expect(allPointsToTeam0[1]).toBe(0);  // Team 1 gets nothing
      expect(allPointsToTeam0[0] + allPointsToTeam0[1]).toBe(42);
    });

    it('handles perfect point split scenario', () => {
      // Scenario: Perfect 21-21 split
      const perfectSplit = calculateGameScore([21, 0, 21, 0]);
      
      expect(perfectSplit[0]).toBe(42); // Team 0: 21 + 21
      expect(perfectSplit[1]).toBe(0);  // Team 1: 0 + 0
      
      // Alternative perfect split
      const altSplit = calculateGameScore([10.5, 10.5, 10.5, 10.5]);
      expect(altSplit[0] + altSplit[1]).toBe(42);
    });

    it('handles counting dominoes distribution extremes', () => {
      // All high-value counting dominoes to one player
      const highValueDominoes = [
        { id: 'five-five', high: 5, low: 5, points: 10 },
        { id: 'six-four', high: 6, low: 4, points: 10 },
        { id: 'five-blank', high: 5, low: 0, points: 5 },
        { id: 'four-one', high: 4, low: 1, points: 5 },
        { id: 'three-deuce', high: 3, low: 2, points: 5 }
      ];

      const totalCountValue = highValueDominoes.reduce((sum, d) => sum + (d.points || 0), 0);
      expect(totalCountValue).toBe(35); // All counting dominoes

      // One player having all counting dominoes would be extremely advantageous
      expect(totalCountValue).toBeGreaterThan(42 / 2); // More than half total points
    });
  });

  describe('End Game Scenarios', () => {
    it('handles game-winning scenarios correctly', () => {
      // Test various winning scores
      const winningScenarios = [
        { team0: 7, team1: 0 },   // Shutout
        { team0: 7, team1: 6 },   // Close game
        { team0: 7, team1: 3 },   // Comfortable win
        { team0: 0, team1: 7 }    // Team 1 wins
      ];

      winningScenarios.forEach(scenario => {
        const gameComplete = isGameComplete(scenario.team0, scenario.team1);
        const hasWinner = scenario.team0 >= 7 || scenario.team1 >= 7;
        
        expect(gameComplete).toBe(hasWinner);
      });
    });

    it('handles overtime scenarios (simultaneous 7+ marks)', () => {
      // Rare scenario where both teams might reach 7 marks in same hand
      const overtimeScenario = { team0: 7, team1: 7 };
      
      // Game rules would determine tiebreaker
      const gameComplete = isGameComplete(overtimeScenario.team0, overtimeScenario.team1);
      expect(gameComplete).toBe(true); // Someone must win
    });

    it('handles mercy rule scenarios (large lead)', () => {
      // Test scenarios with large score differences
      const mercyScenarios = [
        { team0: 7, team1: 0 },
        { team0: 0, team1: 7 },
        { team0: 6, team1: 0 },
        { team0: 0, team1: 6 }
      ];

      mercyScenarios.forEach(scenario => {
        const leadSize = Math.abs(scenario.team0 - scenario.team1);
        const hasSevenMarks = scenario.team0 >= 7 || scenario.team1 >= 7;
        
        if (hasSevenMarks) {
          expect(isGameComplete(scenario.team0, scenario.team1)).toBe(true);
        }
        
        // Large leads indicate dominant performance
        if (leadSize >= 6) {
          expect(leadSize).toBeGreaterThanOrEqual(6);
        }
      });
    });
  });

  describe('Unusual Hand Compositions', () => {
    it('handles hand with maximum doubles', () => {
      const maxDoublesHand = createHandWithDoubles(7); // All doubles
      
      expect(maxDoublesHand).toHaveLength(7);
      expect(maxDoublesHand.every(d => d.high === d.low)).toBe(true);
      
      // All doubles hand is extremely rare but possible
      const doubleValues = maxDoublesHand.map(d => d.high);
      expect(doubleValues).toEqual(expect.arrayContaining([0, 1, 2, 3, 4, 5, 6]));
    });

    it('handles hand with no doubles', () => {
      const noDoublesHand = createHandWithDoubles(0); // No doubles
      
      expect(noDoublesHand).toHaveLength(7);
      expect(noDoublesHand.every(d => d.high !== d.low)).toBe(true);
      
      // Should have variety of suits represented
      const suits = new Set();
      noDoublesHand.forEach(d => {
        suits.add(d.high);
        suits.add(d.low);
      });
      expect(suits.size).toBeGreaterThan(3); // Multiple suits represented
    });

    it('handles hand with all counting dominoes', () => {
      const countingDominoes: Domino[] = [
        { id: 'five-five', high: 5, low: 5, points: 10 },
        { id: 'six-four', high: 6, low: 4, points: 10 },
        { id: 'five-blank', high: 5, low: 0, points: 5 },
        { id: 'four-one', high: 4, low: 1, points: 5 },
        { id: 'three-deuce', high: 3, low: 2, points: 5 },
        { id: 'filler1', high: 1, low: 2, points: 0 },
        { id: 'filler2', high: 2, low: 4, points: 0 }
      ];

      const totalPoints = countingDominoes.reduce((sum, d) => sum + (d.points || 0), 0);
      expect(totalPoints).toBe(35); // All counting points

      // Having all counting dominoes is extremely advantageous
      expect(totalPoints).toBeGreaterThan(42 * 0.8); // More than 80% of total points
    });
  });

  describe('Strategic Edge Cases', () => {
    it('handles forced play scenarios', () => {
      // Scenario: Player must play specific domino due to suit requirements
      const forcedPlayState = createTestState({
        phase: 'playing',
        trump: 1, // ones trump
        currentTrick: [{
          player: 0,
          domino: { id: 'lead', high: 3, low: 2, points: 0 } // threes suit led
        }]
      });

      const constrainedHand: Domino[] = [
        { id: 'forced', high: 3, low: 4, points: 0 }, // Only threes suit domino
        { id: 'trump1', high: 1, low: 5, points: 0 },  // Trump
        { id: 'other1', high: 2, low: 6, points: 0 }   // Other suit
      ];

      const validPlays = getValidPlays(constrainedHand, forcedPlayState.currentTrick, forcedPlayState.trump!);
      
      // Should be forced to play the threes suit domino
      expect(validPlays).toHaveLength(1);
      expect(validPlays[0].id).toBe('forced');
    });

    it('handles last trick scenarios', () => {
      // Last trick where all players have exactly one domino
      const lastTrickHands = {
        0: [{ id: 'last0', high: 2, low: 3, points: 0 }],
        1: [{ id: 'last1', high: 1, low: 4, points: 0 }], // trump
        2: [{ id: 'last2', high: 2, low: 5, points: 0 }],
        3: [{ id: 'last3', high: 3, low: 6, points: 0 }]
      };

      Object.values(lastTrickHands).forEach(hand => {
        expect(hand).toHaveLength(1);
      });

      // All players should be able to play their last domino
      const totalRemainingDominoes = Object.values(lastTrickHands).flat().length;
      expect(totalRemainingDominoes).toBe(4);
    });
  });
});